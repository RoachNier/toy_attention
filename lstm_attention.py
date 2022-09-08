# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:27:52 2022

@author: 14935
"""

import torch
import torch.nn as nn
import math

#lstm用来做效果对比
class LSTM(nn.Module):
    def __init__(self,num_layers,input_size,hidden_size,output_size,batch_first=True):
        super(LSTM,self).__init__()
        self.fc = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        output, (hidden,cell) = self.lstm(x)
        output = self.fc(output)
        output = self.sigmoid(x)
        
#lstm+attention用来做进一步效果对比
class LSTMAttention(nn.Module):
    def __init__(self,num_layers, input_size, hidden_size, output_size,batch_first=True):
        super(LSTMAttention,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=batch_first)
        self.fc = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.wq = nn.Linear(hidden_size,hidden_size)
        self.wk = nn.Linear(hidden_size,hidden_size)
        self.wv = nn.Linear(hidden_size,hidden_size)
        
    def DotProductAttention(self,q,k,v):
        #经典单头点积注意力机制，如果想用加性注意力机制，需要配合lstm层取得与此方法不同的hidden_state
        #说实话感觉lstm本身很鸡肋，哪怕是加了attention。就不实现加性注意力了。
        q,k,v = self.wq(q),self.wk(k),self.wv(v)
        score = q @ k.transpose(-1,-2) / math.sqrt(q.size(-1))
        score = torch.softmax(score, dim = -1)
        output = score @ v
        return output, score
        
    def forward(self,x):
        #x: [bs,seq_len,input_size]
        output, (hidden, cell) = self.lstm(x)
        output,score = self.DotProductAttention(output,output,output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output
    

        




























