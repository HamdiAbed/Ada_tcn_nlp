Adaptive Temporal Convolutional Neural Network for Language models

This repository contains the experiments that were proposed in the paper (AdaTCN)
In this repository, a modified Temporal Convolutional Network (TCN) is presented, where the selection criteria of dilated convolutions that are used in TCN are adaptive.
our code is trained and evaluated in a word-level language model using Penn TreeBank (PTB) and WikiText-02 datasets.
Experiments are coded in PyTorch

Usage:
To run the code, you need to install requirements which are found in requirements.txt

to run the code in a word-level scenario:
For PTB dataset, please use the following code: python3 -m /PTB_word_level/text_penn
For WT2 dataset, please use the following code: python3 -m /WT2_word_level/text_wt2

to use different hyperparameters, please use the args as presented in text_penn.py and text_wt2.py