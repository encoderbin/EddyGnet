import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
import logging
from model.TCN import TCN
from model.Attn import MultiHeadAttention,subsequent_mask
from model.embed import DataEmbedding
from model.myLSTM import TraceLSTM
from model.MAG import Evolving_GConv
from model.STEAG import TrajectoryModel   # B-T-N-F
from torch.distributions import Normal
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def loc_pos(seq_):   

    # seq_ [B，T N 2]

    obs_len = seq_.shape[1]
    num_ped = seq_.shape[2]

    pos_seq = torch.arange(1, obs_len + 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    pos_seq = pos_seq.repeat(1, 1, num_ped, 1) .to(device)

    result = torch.cat((pos_seq, seq_), dim=-1) 
    return result

def clones(module, N):   
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])







class EncoderLayer(nn.Module):
    def __init__(self, dim_hid,Encoder_ESG,Encoder_SGCN,layernorm_num,layernorm_size=1,dropout=0.3, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.Encoder_ESG=Encoder_ESG
        self.Encoder_SGCN=Encoder_SGCN
        self.ESG=Evolving_GConv(conv_channels=1, residual_channels=3, gcn_depth=2, st_embedding_dim=40, dy_embedding_dim=20,#B-C-F-T
                                     dy_interval=3, dropout=0.3, propalpha=0.05)
        self.ESG_linear=nn.Linear(dim_hid * 3, dim_hid)
        self.SGCN=TrajectoryModel(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                        obs_len=15, pred_len=7, n_tcn=5, out_dims=256)  
        # 
        self.short=0.1
    def forward(self, x):
        '''
        :param x: encoder_input: (batch_size, T, N, 256)
        :return: (batch_size, T, N, 256)
        '''
        residual = x
        if self.Encoder_ESG:
            x=torch.permute(x,(0,2,3,1))#B-N-F-T
            x=x.unsqueeze(2)  # B-N-C-F-T
            B,N,C_in,F,T=x.shape
            x=x.reshape((B*N),C_in,F,T).to(device)  # (B*N)-C-F-T
            my_first_state=torch.zeros(F, 40).to(device)
            ESG_output=self.ESG(x,my_first_state)  # (B*N)-C-F-T
            ESG_output=ESG_output.reshape(B,N,3,F,T)
            ESG_output=torch.permute(ESG_output,(0,4,1,3,2))  #B-T-N-F-C
            ESG_output=ESG_output.reshape(B,T,N,(F*3))
            output_esg=self.short*self.ESG_linear(ESG_output)+residual  #B-T-N-256
    #         print('encoder_output',output.shape)
            
        else :
            output_esg = x
            
            
        if self.Encoder_SGCN:  
            SGCN_in=loc_pos(output_esg).to(device)  #B-T-N-257
#             print('SGCN_in',SGCN_in.shape)
            identity_spatial = torch.ones((SGCN_in.shape[1], SGCN_in.shape[2], SGCN_in.shape[2]), device='cuda') * \
                               torch.eye(SGCN_in.shape[2], device='cuda')  # [obs_len N N]
            identity_temporal = torch.ones((SGCN_in.shape[2], SGCN_in.shape[1], SGCN_in.shape[1]), device='cuda') * \
                                torch.eye(SGCN_in.shape[1], device='cuda')  # [N obs_len obs_len]
            identity = [identity_spatial, identity_temporal]
            
            
            output_sgcn=1*self.SGCN(SGCN_in,identity)+residual
            return output_sgcn
        else:
            return output_esg        
class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
#         self.norm = nn.LayerNorm(layer.size)  

    def forward(self, x):
        '''
        :param x: encoder_input: (batch_size, T, N, 256)
        :return: (batch_size, T, N, 256)
        '''
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_hid=8,obs_len=20,pred_len=7,dilation_factor=2,output_TCN=False,output_informer=False,output_LSTM=False,layernorm_size=8, dropout=0.3, residual_connection=True, use_LayerNorm=True,nb_head=2,d_ff=None):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*dim_hid
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.output_TCN=output_TCN        
        self.output_informer=output_informer 
        self.output_LSTM=output_LSTM
        self.TCN = TCN(input_size=dim_hid, output_size=dim_hid, num_channels=[25, 25, 25], kernel_size=2, dropout=dropout,use_LayerNorm=True)
        self.norm_TCN = nn.LayerNorm(dim_hid)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(nb_head, dim_hid, dropout=dropout) # 注意头数必须整除维度--------------------
        self.cross_attn = MultiHeadAttention(nb_head, dim_hid, dropout=dropout) # 注意头数必须整除维度--------------------
        self.norm_selfatt = nn.LayerNorm(dim_hid)
        self.norm_crossatt = nn.LayerNorm(dim_hid)
        self.norm_out = nn.LayerNorm(dim_hid)
        self.conv1 = nn.Conv2d(in_channels=dim_hid, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=dim_hid, kernel_size=1)
        self.activation = F.relu
        self.size=dim_hid
        self.memory_TF_linear = nn.Linear(dim_hid * obs_len, dim_hid)
        self.teacher_length=3
        self.LSTM_numlayer=1
        self.myLSTM=TraceLSTM(input_size=dim_hid+self.teacher_length*2, hidden_size=dim_hid,
                           number_layers=self.LSTM_numlayer)
        self.pred_length=pred_len
    def forward(self, x, memory,labels,Teacher,flag):
  

# return: B-T-N-F

#         print('dec_input.shape',x.shape)
#         print('memory.shape',memory.shape)
        if self.output_TCN:
            # param memory:  B-T-N-F
            output=self.TCN(memory) #  B-T-N-F
            #         if self.residual_connection:  
#             output=output+memory
            return output 

        if self.output_informer:
        # param x: B-T-N-F
        # param memory:  B-T-N-F
            x=torch.permute(x,(0,2,1,3)) #  B-N-T-F
            memory=torch.permute(memory,(0,2,1,3))#  B-N-T-F

            tgt_mask = subsequent_mask(x.size(-2)).to(memory.device)  # 1-T-T
    #         print('dec input',x[0][0])
    #         print('memory',memory[0][0])

            output_selfatt=self.self_attn(x,x,x,tgt_mask)#  B-N-T-F
    # ------------------------------------------------------------------------------------------------
            output_selfatt=x+self.dropout(output_selfatt)
            output_selfatt=self.norm_selfatt(output_selfatt)


            output_crossatt=self.cross_attn(output_selfatt,memory,memory)
    #-----------------------------------------------------------------------------------
            output_crossatt=output_selfatt+self.dropout(output_crossatt)
            output_crossatt=self.norm_crossatt(output_crossatt)


            input_conv1=torch.permute(output_crossatt,(0,3,1,2))  # B-F-N-T
            output_conv1=self.conv1(input_conv1)
            output_conv1=self.dropout(self.activation(output_conv1))

            output_conv2=self.conv2(output_conv1)
            output_conv2=self.dropout(output_conv2)

            out_conv=torch.permute(output_conv2,(0,2,3,1))  #  B-N-T-F

            output=output_crossatt+out_conv
            
            output=self.norm_out(output)

            output=torch.permute(output,(0,2,1,3))# B-T-N-F
#         print('output',output[0])
            return output 


        if self.output_LSTM:
        # param x: B-T_teacher-N-2
        # param memory:  B-T-N-F
            if flag=='train' or flag=='val':
                labels=torch.permute(labels,(2,0,1,3))  #  N-B-7-2
                x=torch.permute(x,(2,0,1,3)) #  N-B-T_teacher-2
                memory=torch.permute(memory,(2,0,1,3))#  N-B-T-F
                N,B,T,F=memory.shape
                memory=memory.reshape(N,B,(T*F))
                memory=self.memory_TF_linear(memory)   #N-B-F
    #             print('memory.shape',memory.shape)




                h0 = torch.zeros( N,self.LSTM_numlayer, B, self.size).to(device) 
                c0 = torch.zeros( N,self.LSTM_numlayer, B, self.size).to(device)   
                previous_hn = h0
                previous_cn = c0
                output_list=torch.randn(self.pred_length, N,B, 2)

                previous_site=torch.randn(N,B,self.teacher_length+self.pred_length, 2).to(device)  # N-B-17-2
                previous_site[:,:,:self.teacher_length,:]=x         # N-B-17-2  
                previous_site[:,:,self.teacher_length:,:]=labels   
                output=x[:,:, -1, ...]  

                for j in range(self.pred_length):
                    if Teacher:
                        previous_site[:,:,j+self.teacher_length-1,...]=output    
                    temp_previous_site = previous_site[:, :,j:j+self.teacher_length, ...].clone()
                    N,B,T_teacher,F_spa=x.shape
                    temp_previous_site = temp_previous_site.reshape(N,B,(T_teacher*F_spa)) 

                    output, hn, cn = self.myLSTM(torch.cat((temp_previous_site, memory), dim=2), previous_hn,
                                               previous_cn)

                    output_list[j,:,:,:]=output  # (pred_len, N,B, 2)
                    previous_hn = hn
                    previous_cn = cn





    #             print('output_list.shape',output_list.shape)
                return output_list    # (pred_len, N,B, 2)
            else:
                labels=torch.permute(labels,(2,0,1,3))  #  N-B-7-2
                labels=labels.repeat(1, 10000, 1, 1)
                x=torch.permute(x,(2,0,1,3)) #  N-B-T_teacher-2
                x=x.repeat(1, 10000, 1, 1)
                memory=torch.permute(memory,(2,0,1,3))#  N-B-T-F
                memory=memory.repeat(1, 10000, 1, 1)
                N,B,T,F=memory.shape
                memory=memory.reshape(N,B,(T*F))
                memory=self.memory_TF_linear(memory)   #N-B-F
    #             print('memory.shape',memory.shape)


    #             B_new=10000

                h0 = torch.zeros( N,self.LSTM_numlayer, B, self.size).to(device) 
                c0 = torch.zeros( N,self.LSTM_numlayer, B, self.size).to(device)   
                previous_hn = h0
                previous_cn = c0
                output_list=torch.randn(self.pred_length, N,B, 2)
                log_prob_list=torch.randn(self.pred_length, N,B, 1)

                previous_site=torch.randn(N,B,self.teacher_length+self.pred_length, 2).to(device)  # N-B-17-2
                previous_site[:,:,:self.teacher_length,:]=x         # N-B-17-2  
                previous_site[:,:,self.teacher_length:,:]=labels   
                output=x[:,:, -1, ...]  

                for j in range(self.pred_length):
                    if Teacher:
                        previous_site[:,:,j+self.teacher_length-1,...]=output    
                    temp_previous_site = previous_site[:, :,j:j+self.teacher_length, ...].clone()
                    N,B,T_teacher,F_spa=x.shape
                    temp_previous_site = temp_previous_site.reshape(N,B,(T_teacher*F_spa)) 

                    LSTM_input=torch.cat((temp_previous_site, memory), dim=2)   #N-B-198
                    myvar=torch.ones(198)/100
                    myvar=myvar.to(device)
                    LSTM_input_new=Normal(LSTM_input, myvar).sample()
                    output, hn, cn = self.myLSTM(LSTM_input_new, previous_hn,
                                               previous_cn)
                    print('LSTM_input_new',LSTM_input_new)
                    print('LSTM_input_new.shape',LSTM_input_new.shape)
                    print('output',output)
                    print('output.shape',output.shape)
                    log_prob=Normal(LSTM_input, myvar).log_prob(LSTM_input_new)
    #                 print('log_prob',log_prob)
                    print('log_prob.shape',log_prob.shape)   
                    log_prob_sum = torch.sum(log_prob, dim=-1)[..., None]
    #                 print('log_prob_sum',log_prob_sum)
                    print('log_prob_sum.shape',log_prob_sum.shape)    
                    output_list[j,:,:,:]=output  # (pred_len, N,B, 2)
                    log_prob_list[j,:,:,:]=log_prob_sum  # (pred_len, N,B, 1)
                    previous_hn = hn
                    previous_cn = cn
    

#             print('output_list.shape',output_list.shape)
                return output_list,log_prob_list    # (pred_len, N,B, 2)
        else :
            return memory

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory,labels,Teacher,flag):    #---------------------------------------------------------------------
        #                       
        '''

        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:    
            x = layer(x, memory,labels,Teacher,flag)
#         x = self.norm(x)
        return x            

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, encoder_embedding, decoder_embedding, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding
        self.prediction_generator = generator
        self.to(DEVICE)

    def forward(self, encoder_raw_input, decoder_raw_input):
        '''
        encoder_input:  (batch_size, N, T_in, F_in)
        decoder_input: (batch, N, T_out, F_out)
        '''
        ZC=self.my_channel_learn(encoder_raw_input)
        encoder_output = self.encode(ZC)  # (batch_size, N, T_in, d_model)
        decoder_output=self.decode(decoder_raw_input, encoder_output)
        my_output=self.prediction_generator(decoder_output)
        return my_output

 

    def my_channel_learn(self, encoder_raw_input):            
        ZC=self.encoder_embedding(encoder_raw_input)
        return ZC
    
    def encode(self, ZC):
        '''
        encoder_input: (batch_size, N, T_in, F_in)
        '''
        
        return self.encoder(ZC)


    def decode(self, decoder_raw_input, encoder_output,labels,Teacher,flag):
        return self.decoder(self.decoder_embedding(decoder_raw_input), encoder_output,labels,Teacher,flag)

def make_model(device,dilation_factor,Encoder_SGCN,Encoder_ESG,output_TCN,output_informer,output_LSTM,encoder_embedding,decoder_embedding,
               dec_in,dim_hid,dim_out,obs_len,pred_len,layernorm_size=1,num_layers_E=1, num_layers_d=2,dropout=0.3,residual_connection=True, use_LayerNorm=True,use_generator=True):




    encoderLayer = EncoderLayer(dim_hid,Encoder_ESG,Encoder_SGCN,layernorm_size,dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    encoder = Encoder(encoderLayer, num_layers_E)

    decoderLayer = DecoderLayer(dim_hid,obs_len,pred_len,dilation_factor,output_TCN,output_informer,output_LSTM,layernorm_size,dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    decoder = Decoder(decoderLayer, num_layers_d)
    
    
    if encoder_embedding:
        encoder_embedding = DataEmbedding(dec_in, dim_hid,dropout)
    else:
        encoder_embedding = nn.Sequential()  
        
    if decoder_embedding:
        decoder_embedding = DataEmbedding(dec_in, dim_hid,dropout)
    else:
        decoder_embedding = nn.Sequential()
#     generator = nn.Sequential()
    
    if use_generator:
        generator = nn.Linear(dim_hid, dim_out)
    else:
        generator = nn.Sequential()

    model = EncoderDecoder(encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           generator,
                           device)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model