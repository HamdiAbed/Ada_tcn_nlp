import os
import torch
from torch.autograd import Variable
import pickle
import requests

"""
Note: The meaning of batch_size in PTB is different from that in MNIST example. In MNIST, 
batch_size is the # of sample data that is considered in each iteration; in PTB, however,
it is the number of segments to speed up computation. 

The goal of PTB is to train a language model to predict the next word.
"""
URL = {
    "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
    "valid": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
}

train = requests.get(URL["train"], allow_redirects= True)
valid = requests.get(URL["valid"], allow_redirects= True)
test = requests.get(URL["test"], allow_redirects= True)

open("penntrain.txt", 'wb').write(train.content)
open("pennvalid.txt", 'wb').write(valid.content)
open("penntest.txt", 'wb').write(test.content)

cwd = os.getcwd()
print('current dir, ',cwd)

def data_generator(args):
    cwd = os.getcwd()
    #print('current dir, ',cwd)
    if os.path.exists(os.path.join(cwd, r"penncorpus")) and not args.corpus:
        corpus = pickle.load(open(os.path.join(cwd, r"penncorpus"), 'rb'))
    else:
        corpus = Corpus(args.data)
        pickle.dump(corpus, open(os.path.join(cwd, r"penncorpus"), 'wb'))
    return corpus


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        cwd = os.getcwd()
        self.dictionary = Dictionary()
        self.train = self.tokenize(r"penntrain.txt")
        self.valid = self.tokenize(r"pennvalid.txt")
        self.test  = self.tokenize(r"penntest.txt")

    def tokenize(self, path):
        """Tokenizes a text file."""
        #assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding = 'utf8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding = 'utf8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.seq_len, source.size(1) - 1 - i)
    data = Variable(source[:, i:i+seq_len], volatile=evaluation)
    target = Variable(source[:, i+1:i+1+seq_len])     # CAUTION: This is un-flattened!
    return data, target
