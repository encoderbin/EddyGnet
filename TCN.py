import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation_factor, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_inputs,n_outputs,(1,kernel_size),dilation=(1,dilation_factor))
        self.relu = nn.ReLU()
        

    def forward(self, x):
        y=self.conv1(x)
        y = self.relu(y)

#         print(x.shape)
#         print(y.shape)
#         print(y)


        return y
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation_factor=dilation_size,
                                      dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,use_LayerNorm):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(0.2)
        self.norm_TCN = nn.LayerNorm(input_size)
    def forward(self, x):
        if self.use_LayerNorm:
            x=self.norm_TCN(x)
        TCN_input=torch.permute(x,(0,3,2,1))  # B-F-N-T
        y1 = self.tcn(TCN_input)  # B-F-N-T
        y2 = torch.permute(y1,(0,3,2,1)) # B-T-N-F
        return self.dropout(self.linear(y2))
