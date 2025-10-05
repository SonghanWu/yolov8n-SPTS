import numpy as np
import torch
from torch import nn
from torch.nn import init


class CFFAttention(nn.Module):
    '''
    CFFAttention
    '''

    def __init__(self, c1, k_size,internal_neurons=64,input_channels=32):
        super(CFFAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=7, stride=1,
                             bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = torch.squeeze(input, dim=0)
        c, w, y = input.size()
        fhl = input[0:c / 2, :, :]
        flh = input[c / 2:c, :, :]
        fa = torch.cat([fhl,flh],dim=0)
        fa = self.con1(fa)
        f_mp = self.max_pool(fa)
        f_ap = self.avg_pool(fa)
        f_sta = torch.stack([f_mp,f_ap],dim=0)
        f_b = self.conv2(f_sta)
        weight = self.sigmoid(f_b)

        f_out = flh * weight + fhl * (1 - weight)
        return f_out