# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:25:51 2022

@author: 14935
"""

# classic Resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d

def conv3x3(in_channel,out_channel,stride = 1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3,
                     stride=stride, padding=1, bias= False) #保持特征图尺寸不变

class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride = 1):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel,stride)
        self.bn1 = BatchNorm2d(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = None
        if in_channel != out_channel or stride !=1:  #如果fm的尺寸或通道改变
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=1
                          ,stride = stride,bias = False),
                BatchNorm2d(out_channel))
    def forward(self,x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.conv2(res)
        res = self.bn2(res)
        
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = shortcut+res
        out = self.relu(out)
        return out
    
def create_layer_basic(in_channel,out_channel,bnum,stride = 1):
    layers = [BasicBlock(in_channel, out_channel,stride = stride) ]
    for i in range(bnum-1): #离谱，只能把循环写在外面，写在列表里面就成了生成器
        layers.append(BasicBlock(out_channel, out_channel,stride = 1))
    print(layers)
    return nn.Sequential(*layers)

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3,bias = False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride = 2, padding = 1)
        self.layer1 = create_layer_basic(64, 64, bnum = 2, stride = 1)
        self.layer2 = create_layer_basic(64, 128, bnum = 2, stride = 2)
        self.layer3 = create_layer_basic(128, 256, bnum =2 ,stride = 2)
        self.layer4 = create_layer_basic(256, 512, bnum = 2, stride  = 2)
        #self.init_weight()
        
    def init_weight(self):
        import modelzoo
        resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k,v in state_dict.items():
            if 'fc' in k : continue
            self_state_dict.update({k:v})
        self.load_state_dict(self_state_dict)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #1/4
        
        x = self.layer1(x) # 1/4
        feat8 = self.layer2(x) #1/8  128
        feat16 = self.layer3(feat8) # 1/16  256 
        feat32 = self.layer4(feat16) # 1/32  512
        return feat8, feat16, feat32
        
if __name__ == '__main__':
    net = Resnet18()
    x = torch.randn(16, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())

        
        
        
        
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


