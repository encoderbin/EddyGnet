import torch
from torch import nn, Tensor
from torch.nn import functional as F


from .esg_utils import Dilated_Inception, MixProp, LayerNorm
from .graph import  NodeFeaExtractor, EvolvingGraphLearner




class Evolving_GConv(nn.Module):
    def __init__(self, conv_channels: int, residual_channels: int, gcn_depth: int,  st_embedding_dim: int, 
                dy_embedding_dim: int, dy_interval: int, dropout=0.3, propalpha=0.05):
        super(Evolving_GConv, self).__init__()
        self.linear_s2d = nn.Linear(st_embedding_dim, dy_embedding_dim)
        self.scale_spc_EGL = EvolvingGraphLearner(conv_channels, dy_embedding_dim)
        self.dy_interval = dy_interval         

        self.gconv = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)

    def forward(self, x, st_node_fea):

        b, _, n, t = x.shape 
        dy_node_fea = self.linear_s2d(st_node_fea).unsqueeze(0)  
        states_dy = dy_node_fea.repeat( b, 1, 1) #[B, N, C]

        x_out = []
    
       
        for i_t in range(0,t,self.dy_interval):     
            x_i =x[...,i_t:min(i_t+self.dy_interval,t)]

            input_state_i = torch.mean(x_i.transpose(1,2),dim=-1)
          
            dy_graph, states_dy= self.scale_spc_EGL(input_state_i, states_dy)  
            x_out.append(self.gconv(x_i, dy_graph))     # GCN
        
        x_out = torch.cat(x_out, dim= -1) #[B, c_out, N, T]      
        return x_out




        







    