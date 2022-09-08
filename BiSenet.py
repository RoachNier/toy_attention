# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:33:59 2022

@author: 14935
"""

from resnet import Resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import torchvision

class ConvBNReLU(nn.Module):
    def __init__(self, in_channel,out_channel, ks=3, stride=1,padding=1,*args,**kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace = True)
        self.init_weight()
    def init_weight(self):
        for ly in nn.children():
            if isinstance(ly,nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight,a = 1)  ####也一并实现以下kaiming_normal_
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class UpSample(nn.Module):
    def __init__(self,in_channel,factor=2):
        super(UpSample,self).__init__()
        out_channel = in_channel*factor*factor
        self.proj = nn.Conv2d(in_channel,out_channel,1,1,0)
        self.up= nn.PixelShuffle(factor)
        self.init_weight()
    def init_weight(self):
        nn.init.xavier_normal_((self.proj.weight),gain = 1.0)  ###实现一下xavier
    def forward(self,x):
        x = self.proj(x)
        x = self.up(x)
        return x

class BiSeNetOutput(nn.Module):
    def __init__(self,in_channel,mid_channel,n_classes,up_factor=32,*args,**kwargs):
        super(BiSeNetOutput,self).__init__()
        self.up_factor = up_factor
        out_channel = n_classes
        self.conv = ConvBNReLU(in_channel, out_channel, ks=3,stride=1,padding=1)
        self.conv_out = nn.Conv2d(mid_channel,out_channel,kernel_size=1,bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,mode = 'bilinear',align_corners=False)
        self.init_weight()
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly,nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight,a = 1)
                if not ly.bias is None: nn.init.constant_(ly.bias,0)
    def forward(self,x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

class AttentionRefineModule(nn.Module):
    def __init(self,in_channel,out_channel,*args,**kwargs):
        super(AttentionRefineModule,self).__init__()
        self.conv = ConvBNReLU(in_channel, out_channel, ks=3,stride=1,padding=1)
        self,conv_atten = nn.Conv2d(out_channel,out_channel,kernel_size=1,bias=False)
        self.bn_attn = BatchNorm2d(out_channel)
        self.init_weight()
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly,nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight,a = 1)
                if not ly.bias is None: nn.init.constant_(ly.bias,0)
    def forward(self,x):
        feat = self.conv(x)
        atten = torch.mean(feat,dim = (2,3),keepdim = True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(atten,feat)
        return out  #通道注意力机制，不改变原featuremap 尺寸，但可能改变其通道数

class ContextPath(nn.Module):
    def __init__(self,*args,**kwargs):
        super(ContextPath,self).__init__()
        self.resnet = Resnet18() #1/8,1/16,1/32
        self.arm16 = AttentionRefineModule(256,128)
        self.arm32 = AttentionRefineModule(512,128)
        self.conv_head32 = ConvBNReLU(128,128,ks=3,stride=1,padding=1)
        self.conv_head16 = ConvBNReLU(128,128,ks=3,stride=1,padding=1)
        self.conv_avg = ConvBNReLU(512,128,ks=1,stride=1,padding=0)
        self.up32 = nn.Upsample(scale_factor=2.0)
        self.up16 = nn.Upsample(scale_factor=2.0)
        self.init_weight()
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self,x):
        feat8, feat16, feat32 = self.resnet(x)
        avg = torch.mean(feat32,dim = (2,3),keepdim = True)
        avg = self.conv_avg(avg)
        
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg #如同residual+shortcut
        feat32_up = self.up(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)
        
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)
        
        return feat16_up, feat32_up #1/8,1/16
    
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath,self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7,stride=2,padding=3)
        self.conv2 = ConvBNReLU(64,64, ks=3,stride=2,padding=1)
        self.conv3 = ConvBNReLU(64,64, ks=3,stride=2,padding=1)
        self.conv_out = ConvBNReLU(64,128,ks=1,stride=1,padding=0)
        self.init_weight(self)
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x
        
class FeatureFusionModule(nn.Module):
    def __init__(self,in_channel,out_channel,*args,**kwargs):
        super(FeatureFusionModule,self).__init__()
        self.convblk = ConvBNReLU(in_channel, out_channel, ks=1,stride=1,padding=0)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1,padding=0,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.init__weight()
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self,fsp,fcp):
        fcat = torch.cat([fsp,fcp],dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat,dim=(2,3),keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(atten,feat)
        feat_out = feat_atten+feat
        return feat_out
    
class BiSeNet(nn.Module):
    def __init__(self,n_classes,aux_mode='train',*args,**kwargs):
        super(BiSeNet,self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath() #两个路是定死的
        self.ffm = FeatureFusionModule(256,256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128,64,n_classes,up_factor=8)
            self.conv_out32 = BiSeNetOutput(128,64,n_classes,up_factor=16)
        self.init_weight()
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self,x):
        H,W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp,feat_cp8)
        
        feat_out = self.conv_out(feat_fuse)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            return feat_out, feat_out16, feat_out32
        elif self.aux_mode == 'eval':
            return feat_out
        elif self.aux_mode == 'pred':
            return feat_out.argmax(dim = 1)
        else:
            raise NotImplementedError
        
    
        
        
        
        
        

            
            
            
            
            
            
            
            
            
            
            
                
        
    
    

            




























