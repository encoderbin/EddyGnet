import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  # T-F
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # B-T-F
        pe = pe.unsqueeze(2)  # B-T-N -F
        self.register_buffer('pe', pe)

    def forward(self, x):# param x: B-T-N-F
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        kernel_size=3
        stride=1
        padding = (1,1)
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):# param x: B-T-N-F
        x = self.tokenConv(x.permute(0,3, 1,2)) # B-F-T-N
        x=x.permute(0,2, 3,1)
        return x# param return: B-T-N-F



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)


        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):# param x: B-T-N-F
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x)