import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt


torch.set_printoptions(4, edgeitems=16 ,sci_mode=False)
torch.autograd.set_detect_anomaly(True)

def sharp_softmax(inputs, gamma):
    w_pow = torch.pow(inputs, gamma)
    w_soft = torch.divide(w_pow, (torch.sum(w_pow, dim = -1, keepdims = True)))
    return w_soft

def gumsharp_softmax(logits,num_filters,seq_len,  hard=False):

    y = sharp_softmax(logits, gamma = 4)
    
    if not hard:
        return y.view(-1, num_filters * seq_len)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    ## y_hard here is the output weights of gumsharping 
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(num_filters * seq_len, -1)

def sparsify(mask, seq_len, num_filters, level):
    window_size = 2 ** level
    num_of_windows = int(seq_len / window_size)
    shape = [ num_filters, num_of_windows, window_size]
    w = mask.view(*shape)
    w = gumsharp_softmax(w,num_filters,seq_len,  True)
    w = torch.reshape(w, (num_filters, seq_len))
    return w


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class SE_block(nn.Module):
    def __init__(self, num_channels, ratio = 4):
        super(SE_block, self).__init__()
        num_channels_reduced = num_channels // ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.activ_1 = nn.ReLU()
        self.activ_2 = nn.Sigmoid()
      
    def forward(self,input):
        """
        :param input_tensor: X, shape = (batch_size,  num_channels, seq_len)
        :return: output tensor
        """
        batch_size,  num_channels, seq_len = input.size()

        # Average along each channel
        squeeze_tensor = input.view(batch_size,  num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.activ_1(self.fc1(squeeze_tensor))
        fc_out_2 = self.activ_2(self.fc2(fc_out_1))
        out = torch.einsum('xyz,xy->xyz', input, fc_out_2)
        return  out

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, level, mask, gaw_1, gaw_2, skip_mask, seq_len,stride,dilation, skip = False,
                 gated_act = False,   dropout=0.2):   ###ADATCN
    #def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):    ###TCN
        super(TemporalBlock, self).__init__()
        self.level = level
        self.mask = mask
        self.gaw_1 = gaw_1
        self.gaw_2 = gaw_2
        self.skip_mask = skip_mask
        self.seq_len = seq_len
        self.skip = skip
        self.gated_act = gated_act
        self.conv1 = weight_norm(CausalConv1d(n_inputs, n_outputs, kernel_size,
                                              stride=stride, dilation=dilation))
        #self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.ninp = n_inputs
        self.nout = n_outputs
        self.BN1 = nn.BatchNorm1d(n_inputs)
        self.BN2 = nn.BatchNorm1d(n_inputs)
        self.conv2 = weight_norm(CausalConv1d(n_inputs, n_outputs, kernel_size,
                                              stride=stride, dilation=dilation))                              
        #self.chomp2 = Chomp1d(padding) 
        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)
        self.identity1 = nn.Identity()
        self.identity2 = nn.Identity()
        self.tanh1 = nn.Tanh()
        self.sig1 = nn.Sigmoid()
        self.tanh2 = nn.Tanh()
        self.sig2 = nn.Sigmoid()
        
        self.downsample = CausalConv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        self.skip_con = [None]
        self.layernorm = nn.LayerNorm([self.nout, self.seq_len], elementwise_affine = True)
        self.SE_block = SE_block(self.nout, 4)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #first block

        y = self.conv1(x)
        y = torch.mul(y, sparsify(self.mask[0], self.mask.shape[-1], self.nout, self.level)) if self.level > 0 else self.identity1(y) 
        #y = self.chomp1(y) #needed for baseline tcn
        y = y if self.downsample is None else self.downsample(y)
        #UNCOMMENT IF YOU WANT TO USE GATED ACTIVATIONS
        if self.gated_act:
            y = torch.mul(self.tanh1(torch.mul(y , self.gaw_1[0])) ,self.sig1(torch.mul(y , self.gaw_1[1]))) #Gated activations
        y = self.BN1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        
        #2nd block
        y = self.conv2(y)
        y = torch.mul(y, sparsify(self.mask[1], self.mask.shape[-1], self.nout, self.level)) if self.level > 0 else self.identity2(y)
        #y = self.chomp2(y) #needed for baseline TCN
        y = y if self.downsample is None else self.downsample(y)
        #UNCOMMENT IF YOU WANT TO USE GATED ACTIVATIONS
        if self.gated_act:
            y = torch.mul(self.tanh1(torch.mul(y , self.gaw_2[2])) ,self.sig1(torch.mul(y , self.gaw_2[3]))) #Gated Activation
        
        y = self.BN2(y)
        y = self.relu2(y)
        out = self.dropout2(y)       
        res = x if self.downsample is None else self.downsample(x)
        out = self.SE_block(out)
        
        #UNCOMMENT IF YOU WANT TO USE THE SKIP CONNECTIONS, AND COMMENT THE FOLLOWING RETURN LINE
        #return self.relu(out + res) * self.skip_mask #multiplying each block with its respective skip connection weight
        #return self.layernorm(self.relu(out + res)) * self.skip_mask #multiplying each block with its respective skip connection weight
        if self.skip:
            return self.relu(out + res) * self.skip_mask #multiplying each block with its respective skip connection weight
        else:
            return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, seq_len, num_inputs, num_channels,skip, gated_act, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = []
        num_levels = len(num_channels)
        self.num_filters = num_channels[-1]
        self.seq_len = seq_len
        num_conv_layers = 2
        self.downsamp1 = CausalConv1d(self.num_filters, self.num_filters, 1) 
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.downsamp2 = CausalConv1d(self.num_filters, self.num_filters, 1)
        self.dropout_final = nn.Dropout(dropout) 
        
        self.mask = torch.empty(size = (num_levels - 1, num_conv_layers,self.num_filters, self.seq_len)).cuda()      
        self.gaw_1 = torch.empty(size = (num_levels, num_conv_layers, self.num_filters, self.seq_len)).cuda()      #Gated_activation for the first conv layer
        self.gaw_2 = torch.empty(size = (num_levels, num_conv_layers * 2, self.num_filters, self.seq_len)).cuda()      #Gated activation for the second conv layer
        self.skip_mask = torch.empty(size = (num_levels,1)).cuda()   
        
        self.mask = torch.nn.parameter.Parameter(nn.init.kaiming_uniform_(self.mask, mode='fan_in', nonlinearity='relu'), requires_grad = True).cuda()
        self.gaw_1 = torch.nn.parameter.Parameter(nn.init.kaiming_uniform_(self.gaw_1, mode='fan_in', nonlinearity='relu'), requires_grad = True).cuda()
        self.gaw_2 = torch.nn.parameter.Parameter(nn.init.kaiming_uniform_(self.gaw_2, mode='fan_in', nonlinearity='relu'), requires_grad = True).cuda()
        self.skip_mask= torch.nn.parameter.Parameter(nn.init.kaiming_uniform_(self.skip_mask, mode = 'fan_in' ,nonlinearity='relu'), requires_grad = True).cuda()
        

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            self.layers += [TemporalBlock(in_channels, out_channels, dilation_size *kernel_size , i, self.mask[i - 1],
                            self.gaw_1[i], self.gaw_2[i], self.skip_mask[i][0], self.seq_len, stride=1, dilation= 1,skip = skip, gated_act = gated_act,
                            dropout=dropout)]

        self.network = nn.Sequential(*self.layers)
    
    def init_weights(self):
      torch.nn.init.kaiming_normal_(self.mask,'fan_out','relu')
      torch.nn.init.kaiming_normal_(self.gaw_1,'fan_out','relu')
      torch.nn.init.kaiming_normal_(self.gaw_2,'fan_out','relu')
      torch.nn.init.kaiming_normal_(self.skip_mask,'fan_out','relu')

    def forward(self, x):
        y = self.network(x) 
        y = self.downsamp1(y)
        y = self.relu1(y)
        y = self.downsamp2(y)
        y = self.relu2(y)
        y = self.dropout_final(y)
        return y

