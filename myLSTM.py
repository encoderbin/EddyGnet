import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from torch import Tensor
import argparse
import datetime
import torch
from torch.nn.modules import Module
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import math
from torch.nn import init
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
import torch.optim as optim
import argparse
import datetime
import torch
import random
from torch.nn import Module
import torch
from geopy.point import Point
from geopy.units import radians
from torch.distributions import Normal
class TraceLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TraceLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Softsign = nn.Softsign()
        self.sigmoid = nn.Sigmoid()

        self._flat_weights_names = []

        w_ii = nn.Parameter(torch.Tensor(input_size, 6 * hidden_size))
        w_hh = nn.Parameter(torch.Tensor(hidden_size, 6 * hidden_size))
        w_cc = nn.Parameter(torch.Tensor(hidden_size, 3 * hidden_size))
        w_ee = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        w_rr = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        b_ih = nn.Parameter(torch.Tensor(6 * hidden_size))
        b_hh = nn.Parameter(torch.Tensor(6 * hidden_size))
        b_k = nn.Parameter(torch.Tensor(hidden_size))

        layer_param = (w_ii, w_hh, w_cc, w_ee, w_rr, b_k, b_ih, b_hh)

        param_names = ['weight_ii', 'weight_hh', 'weight_cc', 'weight_ee',
                       'weight_rr', 'weight_aa', 'bias_ih', 'bias_hh']

        for name, param in zip(param_names, layer_param):
            setattr(self, name, param)
        self._flat_weights_names.extend(param_names)

        self.param_length = len(param_names)
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in
                              self._flat_weights_names]
        self.flatten_parameters()
        self.reset_parameters()

    def reset_parameters(self) -> None:   #用于初始化权重和偏置参数。这里使用了均匀分布的方法来初始化，范围是 -stdv 到 +stdv
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def flatten_parameters(self) -> None: #优化在 GPU 上执行时的内存访问模式
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, num_weights,
                        self.input_size, rnn.get_cudnn_mode(self.mode),
                        self.hidden_size, self.proj_size, self.num_layers,  # type: ignore
                        self.batch_first, bool(self.bidirectional))  # type: ignore

    def forward(self, x, h ,c):
        """
        :param x: input
        :param h: hidden
        :param c: cell
        :return:
        """
        # the weight of input and hidden, bias
        # x = x.squeeze(1)
        bias = self._flat_weights[-1] + self._flat_weights[-2]
        weight_bias = torch.matmul(x, self._flat_weights[0]) + torch.matmul(h, self._flat_weights[1]) + bias #N-B-(6*256)
        weight_bias = torch.split(weight_bias, self.hidden_size, dim=2)

        # split other weight
        weight_cell = torch.split(self._flat_weights[2], self.hidden_size, dim=1)
        # the weight of eddy feature
        forget_gate = self.sigmoid(weight_bias[0] + torch.matmul(c, weight_cell[0]))
        input_gate = self.sigmoid(weight_bias[1] + torch.matmul(c, weight_cell[1]))
        c = forget_gate * c + input_gate * self.Softsign(weight_bias[2])
        output_gate = self.sigmoid(weight_bias[4] + torch.matmul(c, weight_cell[2]))
        h = output_gate * self.Softsign(c)

        return h, c
class TraceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, number_layers):
        super(TraceLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_layers = number_layers

        deep_list = []
        for i in range(number_layers):
            temp_input_size = self.input_size if i == 0 else self.hidden_size
            deep_list.append(TraceLSTMCell(
                input_size=temp_input_size, hidden_size=self.hidden_size
            ))

        self.deep_list = nn.ModuleList(deep_list)
        self.linear = torch.nn.Linear(hidden_size*number_layers, 2)

    def forward(self, x, h, c):# N- B-276,#N-层-B-256
        temp_input = x
#         print('temp_input',temp_input.shape)
        for layer in range(self.number_layers):
            hn = h[:,layer, :, :]
            cn = c[:,layer, :, :]  #N-B-256
            hn, cn = self.deep_list[layer](temp_input, hn, cn)# N- B-276   #N-B-256        
                    # return N-B-256 
            
            temp_input = hn
            out_hn = hn.unsqueeze(1) if layer == 0 else torch.cat((out_hn, hn.unsqueeze(1)),dim=1) #N -层 -B-256  
            out_cn = cn.unsqueeze(1) if layer == 0 else torch.cat((out_cn, cn.unsqueeze(1)),dim=1)
        hn = out_hn.permute(0,2, 1, 3) #N -B-层-256 
        N,B,Lay,F=hn.shape
        hn = hn.reshape(N,B,(Lay*F))
        output = self.linear(hn)   # N-B-2
        return output, out_hn, out_cn  # N-B-2 #N -层 -B-256  