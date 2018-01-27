from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_size, out_size, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(in_size, out_size, kernel_size,
                                           stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

        self.left_padding = dilation * (kernel_size - 1)
        self._init()

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(CausalConv1d, self).forward(x)

    def _init(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.constant(param.data, 1.0)
                # nn.init.xavier_uniform(param.data)
                param.data.uniform_(-0.05, 0.05)
            elif 'bias' in name:
                nn.init.constant(param.data, 0.0)
                # param.data.uniform_(-0.05, 0.05)


if __name__ == "__main__":
    # (B, C, T): (1, 1, 8)
    x = Variable(torch.FloatTensor([[[1, 2, 3, 4, 5, 6, 7, 8]]]))
    causal_conv = CausalConv1d(in_size=1, out_size=2, kernel_size=3, stride=1)
    y = causal_conv(x)
    print('input:', x)
    print('output:', y)