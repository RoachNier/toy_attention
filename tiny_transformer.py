# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:50:44 2022

@author: 14935
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math


class PositionalEmbedding(nn.Module):
    #实验：根据工序调整Position的幅值
    
    def __init__(self,d_model,max_len,device = 'cuda:0'):
        super(PositionalEmbedding,self).__init__()
        self.embedding = torch.zeros(size = (max_len,d_model),device = device,requires_grad=False)
        pos = torch.arange(0,max_len,device = device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0,d_model,step = 2, device=device).float()
        self.embedding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.embedding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))
    def forward(self,x,x_tag):
        #x:[bs,max_len,d_model]
        #x_tag:[bs,max_len,1],包含(1,2,3...)，对应着不同的工序
        bs,max_len,d_model = x.shape
        pos_emb = self.embedding.unsqueeze(0).tile(bs,1,1) #[bs,max_len,d_model]
        #print(pos_emb.shape)
        #print(x_tag.tile(1,1,d_model).shape)
        #print(self.embedding.shape)
        pos_emb  =pos_emb * x_tag.tile(1,1,d_model)
        return pos_emb   #带有工序信息的PE
#********************位置编码测试*********************
# pe = PositionalEmbedding(4, 4, 'cpu')
# x = torch.randn(size = (2,4,4))
# x_tag = torch.tensor([1,1,1,1,1,1,1,1]).reshape(2,4,1)
# embedding = pe(x,x_tag)
# print(embedding)
#******************************************************************
class TransformerEmbedding(nn.Module):
    def __init__(self,input_dim,d_model,max_len,drop_prob = 0.1, device = 'cuda:0'):
        super(TransformerEmbedding,self).__init__()
        self.positionembedding= PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(p = drop_prob)
        self.word_embedding = nn.Linear(input_dim,d_model)
    def forward(self,x,x_tag):
        x = self.word_embedding(x)
        pos_emb = self.positionembedding(x,x_tag)
        return self.dropout(x+pos_emb)
        

class LayerNorm(nn.Module):
    #层归一化引入了各个传感器在同一时刻变化的分布
    
    def __init__(self,d_model,eps = 1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        out = (x-mean)/(std+self.eps)
        return out
    
class ScaleDotProductAttention(nn.Module):
    #经典点积注意力
    
    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
        self.softmax = nn.Softmax(dim = -1)
    def forward(self,q,k,v,mask = None, e = -1e12):
        bs,num_head,max_len,d_head = q.shape
        #q,k,v : [bs,num_head,max_len,d_model//num_head]
        score = q @ k.transpose(-1,-2)/math.sqrt(d_head)
        #print(score.shape)
        #print(mask.shape)
        #assert 1==2
        if mask is not None:
            mask = mask.unsqueeze(1).tile(1,num_head,1,1) #需要传入一个四维张量[bs,1,len_q,len_k]
            score = score.masked_fill(mask ==0,e)  #对False进行mask_fill
            #print(mask.shape)
        score = self.softmax(score)
        v = score @ v
        return v, score

class MultiHeadSelfAttention(nn.Module):
    #进一步封装点积注意力类
    
    def __init__(self,d_model,num_head):
        super(MultiHeadSelfAttention,self).__init__()
        self.num_head= num_head
        self.attention = ScaleDotProductAttention()
        self.wq = nn.Linear(d_model,d_model)  #维度不变
        self.wk = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)
        self.w_concat= nn.Linear(d_model,d_model)
    def forward(self,q,k,v,mask = None):
        q,k,v = self.wq(q), self.wk(k), self.wv(v)
        q,k,v = self.split(q),self.split(k),self.split(v)
        out, attention = self.attention(q,k,v,mask = mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out
    def split(self,x):
        bs,x_len,d_model = x.shape
        d_head = d_model//self.num_head
        return x.view(bs,x_len,self.num_head,d_head).transpose(1,2)
    def concat(self,x):
        bs,num_head,x_len,d_head = x.shape
        return x.transpose(1,2).contiguous().view(bs,x_len,-1)
    
class PositionwiseFeedForward(nn.Module):
    #其实就是MLP
    def __init__(self,d_model,hidden_dim,drop_prob =0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.leakrelu = nn.LeakyReLU()
        self.dropout= nn.Dropout(p = drop_prob)
    def forward(self,x):
        return self.linear2(self.dropout(self.leakrelu(self.linear1(x))))
    
class EncoderLayer(nn.Module):
    #对MSHA和FFN以及LN进行集成
    def __init__(self,d_model = 8,hidden_dim = 16,num_head =1,drop_prob = 0.1):
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p = drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, hidden_dim)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)
    def forward(self,x,s_mask):
        residual = x
        x = self.attention(q = x, k = x, v = x, mask = s_mask)
        x = self.norm1(x+residual)
        x = self.dropout1(x)
        residual = x
        x = self.ffn(x)
        x = self.norm2(x+residual)
        x = self.dropout2(x)
        return x

class Encoder(nn.Module):
    #封装EncoderLayer
    def __init__(self,input_dim,num_layer,max_len,d_model,num_head,hidden_dim,drop_prob = 0.1,device = 'cuda:0'):
        super(Encoder,self).__init__()
        self.embedding = TransformerEmbedding(input_dim,d_model, max_len,drop_prob = drop_prob,device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model,hidden_dim,num_head,drop_prob)
                                     for i in range(num_layer)])

    def forward(self,x,x_tag,s_mask):
        x = self.embedding(x,x_tag)
        for layer in self.layers:
            x = layer(x,s_mask)
        return x

    
class DecoderLayer(nn.Module):
    #Decoder比Encoder要多一个MHSA
    def __init__(self,d_model=8,hidden_dim = 16,num_head =1,drop_prob = 0.1):
        super(DecoderLayer,self).__init__()
        self.attention1 = MultiHeadSelfAttention(d_model, num_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p = drop_prob)
        
        self.attention2 = MultiHeadSelfAttention(d_model, num_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, hidden_dim)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p = drop_prob)
    def forward(self,dec_x,enc_x,t_mask,s_mask):
        residual = dec_x
        dec_x = self.attention1(q = dec_x, k = dec_x, v = dec_x, mask = t_mask)
        dec_x = self.dropout1(self.norm1(dec_x+residual))
        
        residual = dec_x
        dec_x = self.attention2(q = dec_x, k = enc_x, v = enc_x, mask = s_mask)
        dec_x = self.dropout2(self.norm2(dec_x+ residual))
        
        residual = dec_x
        dec_x = self.ffn(dec_x)
        dec_x = self.dropout3(self.norm3(self.ffn(dec_x)))
        
        return dec_x
    
class Decoder(nn.Module):
    #封装decoderlayer
    def __init__(self,out_dim,d_model,hidden_dim,num_head,num_layer,drop_prob = 0.1):
        super(Decoder,self).__init__()
        self.in_linear = nn.Linear(out_dim,d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model,hidden_dim,num_head,drop_prob) for i in range(num_layer)])
        self.linear = nn.Linear(d_model,out_dim*2)
        self.out_linear = nn.Linear(out_dim*2,out_dim)
        self.sigmoid  = nn.Sigmoid()
    def forward(self,tgt,enc_x,t_mask,s_mask):
        tgt = self.in_linear(tgt)
        for layer in self.layers:
            x = layer(tgt,enc_x,t_mask,s_mask)
        #print(x.shape)
       #print(f'decoder x shape:{x.shape}')
        out = self.sigmoid(self.out_linear(self.linear(x)))
        #print(out.shape)
        #print(out)
        #print(f'decoder out shape:{out.shape}')
        return out

class Transformer(nn.Module):
    #总模型集成
    def __init__(self,input_dim,max_len,d_model,num_head,hidden_dim,s_pad_idx,t_pad_idx,num_layer,out_dim,drop_prob = 0.1, device = 'cuda:0'):
        super(Transformer,self).__init__()
        self.s_pad_idx = s_pad_idx
        self.t_pad_idx = t_pad_idx
        self.device = device
        self.encoder = Encoder(input_dim,num_layer, max_len, d_model, num_head, hidden_dim,drop_prob=drop_prob,device = device)
        self.decoder = Decoder(out_dim,d_model,hidden_dim,num_head,num_layer,drop_prob = drop_prob)
        
    def forward(self,src,src_tag,tgt):
        src_mask = self.make_pad_mask(src,src)  #还没有经过embedding，因此是原序列，[bs,max_len,d_feature]
        src_tgt_mask = self.make_pad_mask(tgt,src)
        tgt_mask = self.make_pad_mask(tgt, tgt) * self.make_seq_mask(tgt, tgt)
        #print(tgt_mask)
        #print(f'tgt_mask{tgt_mask.shape}')
        enc_src = self.encoder(src,src_tag,src_mask)
        #print(enc_src)
        output = self.decoder(tgt,enc_src,tgt_mask,src_tgt_mask)
        return output
        
    def make_pad_mask(self,q,k):
        #对query和key进行pad，但结构化数据src不需要pad，自回归的Decoder需要pad
        #Encoder的max_len与每个样本的实际长度不一样，提前进行了padding,因此需要mask每个样本的padding部分
        #Decoder为了实现并行计算，对自回归进行mask剔除不应该被提前感知的时序段
        #len_q,len_k = q.size(1),k.size(1)
        #k = k.ne(self.s_pad_idx).unsqueeze(1).unsqueeze(2)
        #k = k.repeat(1,1,len_q,1)
        #q = q.ne(self.s_pad_idx).unsqueeze(1).unsqueeze(2)
        #mask = k&q
        
        #q [bs,max_len_q,feature_dim_q]
        #k [bs,max_len_k,feature_dim_k]
        bs_q,len_q,fdim_q = q.shape
        bs_k,len_k,fdim_k = k.shape
        assert bs_q == bs_k, print(f'bs_q:{bs_q},bs_k:{bs_k}')
        q = q[:,:,0:1]
        q = q.ne(self.s_pad_idx)
        k = k[:,:,0:1].transpose(-1,-2)
        k = k.ne(self.s_pad_idx)
        mask = k & q
        #这次重庆的数据是结构化的，不需要padding, 因此直接返回False的mask tensor即可
        # bs_q,len_q = q.size(0),q.size(1)
        # bs_k,len_k = k.size(0),k.size(1)
        # assert bs_q == bs_k, print(f'bs_q:{bs_q},bs_k:{bs_k}')
        # mask = torch.tensor([True]).tile(len_k).unsqueeze(0).tile(len_q,1).unsqueeze(0).tile(bs_q,1,1).to(self.device)
        #print(f'*****mask_shape{mask.shape}')
        return mask # [bs,1,len_q,len_k]
    def make_seq_mask(self,q,k):
        len_q,len_k = q.size(1),k.size(1)
        mask = torch.tril(torch.ones(len_q,len_k)).type(torch.BoolTensor).to(self.device)
        return mask
        
        
        
    
    


class TransformerEncoder(nn.Module):
    #只包含Encoder部分，目的是在训练后对d_model所包含的semantic feature进行分析
    def __init__(self,input_dim,max_len,d_model,num_head,hidden_dim,s_pad_idx,t_pad_idx,num_layer,out_dim,drop_prob = 0.1, device = 'cuda:0'):
        super(TransformerEncoder,self).__init__()
        self.s_pad_idx = s_pad_idx
        self.t_pad_idx = t_pad_idx
        self.device = device
        self.encoder = Encoder(input_dim,num_layer, max_len, d_model, num_head, hidden_dim,drop_prob=drop_prob,device = device)
        #self.decoder = Decoder(out_dim,d_model,hidden_dim,num_head,num_layer,drop_prob = drop_prob)
        
    def forward(self,src,src_tag):
        src_mask = self.make_pad_mask(src,src)  #还没有经过embedding，因此是原序列，[bs,max_len,d_feature]
       # src_tgt_mask = self.make_pad_mask(tgt,src)
        #tgt_mask = self.make_pad_mask(tgt, tgt) * self.make_seq_mask(tgt, tgt)
        #print(tgt_mask)
        #print(f'tgt_mask{tgt_mask.shape}')
        enc_src = self.encoder(src,src_tag,src_mask)
        #print(enc_src)
        #output = self.decoder(tgt,enc_src,tgt_mask,src_tgt_mask)
        return enc_src
        
    def make_pad_mask(self,q,k):
        #对query和key进行pad，但结构化数据src不需要pad，自回归的Decoder需要pad
        #Encoder的max_len与每个样本的实际长度不一样，提前进行了padding,因此需要mask每个样本的padding部分
        #Decoder为了实现并行计算，对自回归进行mask剔除不应该被提前感知的时序段
        #len_q,len_k = q.size(1),k.size(1)
        #k = k.ne(self.s_pad_idx).unsqueeze(1).unsqueeze(2)
        #k = k.repeat(1,1,len_q,1)
        #q = q.ne(self.s_pad_idx).unsqueeze(1).unsqueeze(2)
        #mask = k&q
        
        #q [bs,max_len_q,feature_dim_q]
        #k [bs,max_len_k,feature_dim_k]
        bs_q,len_q,fdim_q = q.shape
        bs_k,len_k,fdim_k = k.shape
        assert bs_q == bs_k, print(f'bs_q:{bs_q},bs_k:{bs_k}')
        q = q[:,:,0:1]
        q = q.ne(self.s_pad_idx)
        k = k[:,:,0:1].transpose(-1,-2)
        k = k.ne(self.s_pad_idx)
        mask = k & q
        #这次重庆的数据是结构化的，不需要padding, 因此直接返回False的mask tensor即可
        # bs_q,len_q = q.size(0),q.size(1)
        # bs_k,len_k = k.size(0),k.size(1)
        # assert bs_q == bs_k, print(f'bs_q:{bs_q},bs_k:{bs_k}')
        # mask = torch.tensor([True]).tile(len_k).unsqueeze(0).tile(len_q,1).unsqueeze(0).tile(bs_q,1,1).to(self.device)
        #print(f'*****mask_shape{mask.shape}')
        return mask # [bs,1,len_q,len_k]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        