from __future__ import absolute_import
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import wrappers
import casual_conv


class CNNEncoder(nn.Module):
    def __init__(self, input_size=4096, hid_size=1024,
                 num_layers=3, enc_len=10, res_connection=False):
        super(CNNEncoder, self).__init__()
        self.enc_len = enc_len
        self.res_connection = res_connection

        self.cnn = nn.ModuleList([
            wrappers.Conv1d(input_size if layer == 0 else hid_size, hid_size,
                            kernel_size=3, stride=1, padding=1)
            for layer in range(num_layers)
        ])

        if input_size != hid_size and res_connection:
            self.fc_input = wrappers.Linear(input_size, hid_size)

    def forward(self, input):
        """
        :param input: bsz x vid_len x input_size or
        bsz x vid_len x image_size x image_size
        :return:
        """
        bsz, vid_len = input.size(0), input.size(1)
        assert vid_len >= self.enc_len
        x = input.view(bsz, vid_len, -1)
        x = x[:, :self.enc_len, :]

        # bsz x enc_len x input_size -> bsz x input_size x enc_len
        # B x T x C -> B x C x T
        x = x.transpose(1, 2).contiguous()
        for i, conv in enumerate(self.cnn):
            if i == 0:
                if hasattr(self, 'fc_input'):
                    residual = self.fc_input(x.transpose(2, 1))
                    residual = residual.transpose(1, 2).contiguous()
                else:
                    residual = x
                x = conv(x)
                x = x + residual if self.res_connection else x
                x = F.relu(x)
                # x = F.tanh(x)
            elif i != len(self.cnn) - 1:
                residual = x
                x = conv(x)
                x = x + residual if self.res_connection else x
                x = F.relu(x)
                # x = F.tanh(x)
            else:
                residual = x
                x = conv(x)
                x = x if self.res_connection else x
        # bsz x hid_size x enc_len -> bsz x enc_len x hid_size
        # B x C x T -> B x T x C
        x = x.transpose(2, 1)
        return x


class CNNDecoder(nn.Module):
    """1d cnn decoder"""
    def __init__(self, input_size, output_size=4096, hid_size=1024,
                 num_layers=3, dropout=0.2):
        super(CNNDecoder, self).__init__()
        self.dropout = dropout

        self.cnn = nn.ModuleList([
            wrappers.Conv1d(input_size if layer == 0 else hid_size, hid_size,
                            kernel_size=3, stride=1, padding=1)
            # casual_conv.CausalConv1d(input_size if layer == 0 else hid_size, hid_size,
            #                          kernel_size=3, stride=1, dilation=1)
            for layer in range(num_layers)
        ])
        self.fc_output = wrappers.Linear(hid_size, output_size, True)

    def forward(self, enc_out):
        """
        :param enc_out: bsz x enc_len x hid_size
        :return:
        """
        x = enc_out
        bsz, enc_len, _ = enc_out.size()
        for i, conv in enumerate(self.cnn):
            x = conv(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_output(x)
        # x = F.sigmoid(x)
        return x


class CNNPredictor(nn.Module):
    """1d causal cnn decoder"""
    def __init__(self, input_size, output_size=4096, hid_size=1024,
                 num_layers=3, dropout=0.2):
        super(CNNPredictor, self).__init__()
        self.dropout = dropout

        self.cnn = nn.ModuleList([
            # wrappers.Conv1d(input_size if layer == 0 else hid_size, hid_size,
            #                 kernel_size=3, stride=1, padding=1)
            casual_conv.CausalConv1d(input_size if layer == 0 else hid_size, hid_size,
                                     kernel_size=3, stride=1, dilation=1)
            for layer in range(num_layers)
        ])
        self.fc_output = wrappers.Linear(hid_size, output_size, True)

    def forward(self, enc_out):
        """
        :param enc_out: bsz x enc_len x hid_size
        :return:
        """
        pass