# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:08:07 2022

@author: 14935
"""

import torch
import torch.nn as nn

def dw_pw_block(in_channels,out_channels,dw_stride,pw_stride):
    #生成mobilenet中基本的block，避免类里重复写
    #首先是dw
    conv1 = nn.Conv2d(in_channels,in_channels,3,1,padding=1,groups = in_channels)
    conv2 = nn.Conv2d(in_channels,out_channels,1,1,padding=0)
    return nn.Sequential(conv1,conv2)

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,2,padding = 1) #output: 32*112*112
        self.block1 = dw_pw_block(32,64,1,1)
        self.block2 = dw_pw_block(64,128,2,1)
        self.block3 = dw_pw_block(128,128,1,1)
        self.block4 = dw_pw_block(128,256,2,1)
        self.block5 = dw_pw_block(256,256,1,1)
        self.block6 = dw_pw_block(256,512,2,1)
        self.List7 = nn.ModuleList([dw_pw_block(512,512,1,1) for i in range(5)])
        self.block8 = dw_pw_block(512,1024,2,1)
        self.block9 = dw_pw_block(1024,1024,1,1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024,1000)
        self.softmax = nn.Softmax(dim = -1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        for layer in self.List7:
            x = layer(x)
        x = self.block8(x)
        x = self.block9(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.permute(0,2,3,1).contiguous()
        x = self.fc(x)
        x = self.softmax(x)
        return x
        
if __name__ =='__main__':
    x = torch.randn(1,3,224,224)
    model = MobileNet()
    y = model(x)
        
        
        
        