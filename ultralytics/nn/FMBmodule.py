import numpy as np
import torch
from torch import nn
from torch.nn import init



class FMBmodule(nn.Module):

    def __init__(self, k_size, channel_in=512,channel_out=512):
        super(FMBmodule, self).__init__()
        self.depthconv3 = nn.Conv2d(channel_in, channel_out, 3, padding=3 // 2, groups=channel_in)
        self.depthconv5 = nn.Conv2d(channel_in, channel_out, 5, padding=5 // 2, groups=channel_in)
        self.depthconv7 = nn.Conv2d(channel_in, channel_out, 7, padding=7 // 2, groups=channel_in)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, input):
        input = torch.squeeze(input, dim=0)
        fa = self.conv1(input)
        c,w,y = fa.size()
        f0 = input[0:c/4,:,:]
        f1 = input[c / 4:c / 2, :, :]
        f2 = input[c / 2:(3*c) / 4, :, :]
        f3 = input[(3 * c) / 4:c, :, :]
        f1_out = self.depthconv3(f1)
        f2_out = self.depthconv5(torch.cat((f1_out, f2), dim=0))
        f3_out = self.depthconv7(torch.cat((f2_out, f3), dim=0))
        f_out = self.conv1(torch.cat((f0, f1_out, f2_out, f3_out), dim=0))

        return f_out