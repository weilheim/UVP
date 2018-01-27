from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def Conv1d(in_channels, out_channels, kernel_size,
           stride=1, padding=0, dilation=1, bias=True):
    m = nn.Conv1d(in_channels, out_channels, kernel_size,
                  stride, padding, dilation, bias=bias)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # nn.init.xavier_uniform(param.data)
            param.data.uniform_(-0.05, 0.05)
        elif 'bias' in name:
            # nn.init.constant(param.data, 0.0)
            param.data.uniform_(-0.05, 0.05)
    return m

def LSTMCell(input_dim, hidden_dim, **kwargs):
    m = nn.LSTMCell(input_dim, hidden_dim, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.05, 0.05)
    return m

def LSTM(input_dim, hidden_dim,
         dropout=0.0, bias=False, bidirectional=False):
    # note this is always 1 layer LSTM, batch_first is always False.
    m = nn.LSTM(input_dim, hidden_dim, num_layers=1, dropout=dropout,
                bias=bias, batch_first=False, bidirectional=bidirectional)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.05, 0.05)
    return m

def Linear(in_features, out_features, bias=False, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.05, 0.05)
    if bias:
        m.bias.data.uniform_(-0.05, 0.05)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.05, 0.05)
    # take care of padding index after initialization
    if padding_idx is not None:
        m.weight.data[padding_idx, :] = 0.0
    return m