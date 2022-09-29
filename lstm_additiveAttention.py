# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:35:42 2022

@author: 14935
"""
import torch
import torch.nn as nn
import math
#simplified additive attention实现
class LSTMAttention(nn.Module):
    def __init__(self,num_layers, input_size, hidden_size, output_size,batch_first=True):
        super(LSTMAttention,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=batch_first)
        self.fc = nn.Linear(hidden_size,output_size)
        self.fc_shortcut = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
        self.wh = nn.Linear(hidden_size,hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        
    def AdditiveAttention(self,output):
        score = self.v(torch.tanh(self.wh(output))).transpose(-1,-2).contiguous()
        output = torch.bmm(score, output)
        return output, score
        
    def forward(self,x):
        #x: [bs,seq_len,input_size]
        B, L, H = x.shape
        output, (hidden, cell) = self.lstm(x)
        outputs = []
        for i in range(L):
            outputs_item,score = self.AdditiveAttention(output)
            outputs_item = self.fc(outputs_item)
            outputs_item = self.sigmoid(outputs_item)
            outputs.append(outputs_item)
        outputs = torch.cat(outputs, dim = 1)
        # shortcut
        output = self.fc(output)
        output = self.sigmoid(output)
        outputs = outputs + output
        return outputs
    
    
if __name__ =='__main__':
    x = torch.randn(2,8,4)
    model = LSTMAttention(1,4,5,6)
    y = model(x)
    print(y)
    
    
    
    
    
    
    
    
    
    
    
    
    