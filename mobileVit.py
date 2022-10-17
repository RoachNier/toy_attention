# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:28:58 2022

@author: 14935

注意:在看Attention和MobileViTBlock的时候，观察维度变化与ViT的差异；
ViT是将一个patch直接展平，即一张三维feature map展平成二维，
而MobileViT仍旧保留了最后一维的维度为channel的数目，同时将每个patch的不同位置拆成
独立的二维张量，让不同位置的点进行Attention；
由于Transformer的FLOPS基本取决于Attention的d_model数目，而MobileViT相比于ViT，d_model降低了(4*patch_size)倍，
因而其速度快也在意料之中。但唯独一点，论文中也提到，边缘端显卡资源很少有支持transformer的模块，相比于N卡，其推理速度还是会变慢很多。
整个模型的亮点就在于此⬆，其他的都是玄学实验。
"""

import torch
import torch.nn as nn


class conv_1x1_bn(nn.Module):
    def __init__(self,inp,oup):
        super().__init__()
        self.net =  nn.Sequential(
            nn.Conv2d(inp,oup,1,1,0,bias=False),
            nn.BatchNorm2d(oup),
            nn.SiLU())
    def forward(self, x):
        return self.net(x)
    
class conv_nxn_bn(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size,stride,1,bias=False),
            nn.BatchNorm2d(oup),
            nn.SiLU())
    def forward(self, x):
        return self.net(x)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn #多传进来一个函数,ViT开头就是这么写的
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x),**kwargs)
    
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, num_head=8,dim_head=64,dropout=0.):
        super().__init__()
        self.d_model = num_head * dim_head
        self.project_out = nn.Linear(self.d_model, dim)
        self.num_head, self.dim_head = num_head, dim_head
        self.scale = dim_head ** -0.5
        self.w = nn.Linear(dim, self.d_model*3, bias = False)
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        qkv = self.w(x)
        q, k, v = qkv.chunk(3, dim = -1)
        q, k ,v = self.DimtoMultiHead(q), self.DimtoMultiHead(k), self.DimtoMultiHead(v)
        score = q @ k.transpose(-1,-2) ** self.scale
        score = self.softmax(score)
        #print(score.shape, v.shape)
        attn = torch.matmul(score, v) #超过3D就不能用bmm，而需要用matmul
        attn = self.MultiHeadtoDim(attn)
        attn = self.project_out(attn)
        return attn
    def DimtoMultiHead(self,x):
        B, P, nP, L = x.shape
        return x.reshape(B, P, nP, self.num_head,self.dim_head).transpose(-2,-3).contiguous()
    def MultiHeadtoDim(self, x):
        B, P, num_head, nP, dim_head = x.shape
        return x.transpose(-2,-3).contiguous().reshape(B,P,nP,-1)
        
class Transformer(nn.Module):
    def __init__(self,depth,dim,hidden_dim,num_head=8,dim_head=64,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim,num_head,dim,dropout)),
                PreNorm(dim,FFN(dim,hidden_dim,dropout))]))
    def forward(self,x):
        #print(x.shape)
        for Atten, feedforward in self.layers:
            x = Atten(x) + x
            x = feedforward(x) + x
        return x

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1,2], "stride should be in [1,2]"
        hidden_dim = inp * expansion
        self.use_residual = self.stride == 1 and inp == oup
        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim,hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
    def forward(self, x):
        if self.use_residual:
            x = x + self.conv(x)
        else:
            x = self.conv(x)
        return x
            
class MobileVitBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(depth, dim, mlp_dim, 4, 8, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2*channel, channel, kernel_size)
    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).contiguous()
        #如果觉得麻烦可以用einops.rearrange,但不清楚哪个方法会使得系统调用时间更少
        x = x.reshape(B, H//self.ph, self.ph, W//self.pw, self.pw, C).permute(0,2,4,1,3,5).contiguous().reshape(B, self.pw*self.ph, -1, C)
        print(f'进入transformer的张量维度:{x.shape}')
        x = self.transformer(x)
        x = x.reshape(B,self.ph, self.pw, H//self.ph, W//self.pw, C).permute(0,5,3,1,4,2).contiguous().reshape(B,C,H,W)
        x = self.conv3(x)
        x = torch.cat((x,y),1)
        x = self.conv4(x)
        return x
    
class MobileVit(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2,2)):
        super().__init__()
        ih, iw = image_size
        assert ih == iw, "input image must be square size"
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        L = [2, 4, 3]
        #下面就是玄学组装看实验效果的
        self.conv1 = conv_nxn_bn(3,channels[0],stride=2) #MV22 -> 尺寸/2
        self.mv2 = nn.ModuleList()
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion)) #尺寸不变
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[3], channels[4], 1, expansion))
        self.mv2.append(MV2Block(channels[4], channels[5], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[6], channels[7], 2, expansion)) #尺寸共下降32倍
        self.mvit = nn.ModuleList()
        self.mvit.append(MobileVitBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileVitBlock(dims[1], L[1], channels[6], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileVitBlock(dims[2], L[2], channels[7], kernel_size, patch_size, int(dims[2]*4)))
        self.conv2 = conv_1x1_bn(channels[-1],channels[-1])
        self.pool = nn.AvgPool2d(ih//32, 1) #输入的图片是方形的才能直接用ih
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)
        x = self.mv2[4](x)
        
        x = self.mvit[0](x)
        
        x = self.mv2[5](x)
        
        x = self.mvit[1](x)
        
        x = self.mv2[6](x)
        
        x = self.mvit[2](x)
        
        x = self.conv2(x)
        x = self.pool(x).reshape(-1, x.shape[1]) #reshape成channel数
        x = self.fc(x)
        return x
    
def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 24, 24, 48, 48, 64, 80, 320]
    return MobileVit((256,256), dims, channels, num_classes=1000, expansion=2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    img = torch.randn(size=(1,3,256,256)) #尺寸是32的倍数
    model = mobilevit_xxs()
    pred = model(img)
    print(pred.shape)
    print(count_parameters(model))
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
