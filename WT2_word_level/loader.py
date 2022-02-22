import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../../")
from utils import *
from utils import data_generator
from model import *
from model import TCN as TCN
import pickle
from random import randint
from torchinfo import summary
import os
import matplotlib.pyplot as plt
CUDA_AVAILABLE_DEVICES=1

import sys

#sys.stdout = open('stdout.txt', "w")

parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus (default: ./data/penn)')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings (default: 600)')
parser.add_argument('--levels', type=int, default=6,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=.001,
                    help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default= 128,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=2322,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='RAdam',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=1,
                    help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=128,
                    help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
args = parser.parse_args()

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()



# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
corpus = data_generator(args)
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, eval_batch_size, args)

n_words = len(corpus.dictionary)
print('vocab size: ', n_words)

num_chans = [int(args.nhid)] * (args.levels - 1) + [args.emsize]
print('num_chans', num_chans)
k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
tied = args.tied
print('args.emsize' , args.emsize)
device = "cuda:1"
#loading pre-trained model
model_path = 'model.pt'
model = torch.load(model_path)
"""
model_path = "model.pt"
model = TCN(args.seq_len,
 args.emsize,
 n_words,
 num_chans,
 dropout=dropout,
 emb_dropout=emb_dropout,
 kernel_size=k_size,
 tied_weights=tied)
"""
#print model summary
#print(summary(model,
#              dtype = [torch.long],
#              input_shape = (args.batch_size, args.seq_len, args.nhid)))
#print('for model')

#for params in model.state_dict():
#    print(params, '\t', model.state_dict()[params].size)
if args.cuda:
    model.to(device)

# May use adaptive softmax to speed up training
criterion = nn.CrossEntropyLoss()

lr = args.lr
optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, eps = 1e-3)

#print(ckpt_load)
"""
print('for loaded ckpt')
for params in ckpt_load.state_dict():
    print(params, '\t', ckpt_load.state_dict()[params].size)
optimizer.load_state_dict(ckpt_load['optimizer'])
model.load_state_dict(ckpt_load['model_state_dict'])
valid_loss_min = ckpt_load['valid_loss_min']
start_epoch = ckpt_load['epoch']

print('model= ', model)
print('optim= ', optimizer)
print('start epoch= ', start_epoch)
print('valid_loss_min=  = {:.6f}'.format(valid_loss_min))
"""
@torch.no_grad()
def evaluate(data_source):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    with torch.no_grad():
        for i in range(0, data_source.size(1) - args.seq_len - 1, args.validseqlen):
            if i + args.seq_len - args.validseqlen >= data_source.size(1) - 1:
                continue
            data, targets = get_batch(data_source, i, args, evaluation=True)
            if args.cuda == True:
                data, targets = data.to(device), targets.to(device)
            output = model(data)

            # Discard the effective history, just like in training
            eff_history = args.seq_len - args.validseqlen
            final_output = output[:, eff_history:].contiguous().view(-1, n_words)
            final_target = targets[:, eff_history:].contiguous().view(-1)

            loss = criterion(final_output, final_target)

            # Note that we don't add TAR loss here
            total_loss += (data.size(1) - eff_history) * loss.item()
            processed_data_size += data.size(1) - eff_history
        return total_loss / processed_data_size

def train():
    # Turn on training mode which enables dropout.
    global train_data
    model.train()
    total_loss = 0
    tr_loss = 0
    start_time = time.time()
    #print('train_data shize', train_data.size(1))
    #print('train_data size modulus seq_len', train_data.size(1) // args.seq_len - 1)
    batch_loss = []
    counter= 0
    for batch_idx, i in enumerate(range(0, train_data.size(1) - args.seq_len - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= train_data.size(1) - 1:
            continue
        data, targets = get_batch(train_data, i, args)
        if args.cuda == True:
            data, targets = data.to(device), targets.to(device)
        #print('data shape', data.shape)
        optimizer.zero_grad()
        output = model(data)

        # Discard the effective history part
        eff_history = args.seq_len - args.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        final_target = targets[:, eff_history:].contiguous().view(-1)
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        loss = criterion(final_output, final_target)
        #batch_loss.append(loss)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        tr_loss += loss.item()
        counter +=1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, train_data.size(1) // args.validseqlen, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            
            """
            print('sentence | target  | output ')
            print('{}  | {} | {}'.format(" ".join([words[w] for w in data[0]]),
                                         words[final_target[0]],
                                        words[torch.argmax(final_output, dim = -1)[0]]))
            
            """
    tr_loss_plot.append(tr_loss / counter)

if __name__ == "__main__":
    best_vloss = 1e-8

    tr_loss_plot = []
    val_loss_plot = []
    test_loss_plot = []
    all_vloss = []
    # At any point you can hit Ctrl + C to break out of training early.

    try:

        patience = 20
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            test_loss = evaluate(test_data)

            #train_loss_plot.append(ep_loss)
            val_loss_plot.append(val_loss)
            test_loss_plot.append(test_loss)
            
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            test_loss, math.exp(test_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss

            # Anneal the learning rate if the validation loss plateaus
            if epoch > 20 and val_loss >= max(all_vloss[-20:]):
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(val_loss)
             
            if epoch > patience and val_loss >= max(all_vloss[-patience:]):
                print('Early stopping the training as val_loss did not improve in the last {} epochs'.format(patience))
                break
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)
    
    # Run on test data.
    test_loss = evaluate(test_data)
    
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


    

    ##Plotting losses
    num_eps = len(tr_loss_plot)
    plt.figure()
    plt.plot(tr_loss_plot,color = 'r', label = "Train_loss")
    plt.plot(val_loss_plot,color = 'g', label = "Validation_loss")
    plt.plot(test_loss_plot, color = 'b', label = "Test_loss")
    plt.title('losses')
    plt.legend()
    plt.show()
    plt.savefig('losses.png')
#sys.stdout.close()