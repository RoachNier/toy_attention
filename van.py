# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:10:33 2022

@author: 14935
"""

import torch
import torch.nn as nn
from layers import DropPath, trunc_normal_ #一个是对shortcut的dropout,另一个是初始化权重的方法
from torch import Tensor
class DWConv(nn.Module):
    def __init__(self,dim=768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim,dim,3,1,1,groups=dim)
    def forward(self,x) ->Tensor:
        return self.dwconv(x)
    
class MLP(nn.Module):
    '''先做pw，再做dw，再做pw的macron结构'''
    def __init__(self,dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim if out_dim is not None else dim
        self.fc1 = nn.Conv2d(dim,hidden_dim,1) #pwconv
        self.dw_con = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim,out_dim,1)
    def forward(self,x) -> Tensor:
        x = self.fc1(x)
        x = self.dw_con(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class LKA(nn.Module):
    '''
    求attn时通道数目和尺寸数目都不变，先做空间融合再做深度融合；求得的attn与x按位相乘
    '''
    def __init__(self,dim) ->None:
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2,groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9,dilation=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x) -> Tensor:
        u = x.clone()
        attn = self.conv1(self.conv_spatial(self.conv0(x)))
        return attn * u 
    
class Attention(nn.Module):
    '''注意力前后都有pwConv的macron结构'''
    def __init__(self,dim) -> None:
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim ,1)
        self.act = nn.GELU()
        self.spatial_gating_unit = LKA(dim)
        self.proj_2 = nn.Conv2d(dim, dim ,1)
    def forward(self,x) -> Tensor:
        shortcut = x.clone()
        x = self.proj_2(self.spatial_gating_unit(self.act(self.proj_1(x))))
        x = x + shortcut
        return x
    
class Block(nn.Module):
    '''基本的模块：先attn，再mlp'''
    def __init__(self, dim, mlp_ratio=4, dpr=0., init_value=1e-2):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(dpr) if dpr>0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim,int(dim*mlp_ratio))
        self.layer_scale_1 = nn.Parameter(init_value*torch.ones(dim),requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value*torch.ones(dim),requires_grad=True)
    def forward(self,x) -> Tensor:
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x
    
class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch,embed_dim,patch_size,stride,patch_size//2)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self,x):
        x = self.proj(x)
        x = self.norm(x)
        B,C,H,W = x.shape
        return x, H, W

van_settings = {'S':[[2,2,4,2],[64,128,320,512]]}

class VAN(nn.Module):
    def __init__(self,model_name = 'S', num_classes = 1000, pretrained = None, *args, **kwargs):
        super().__init__()
        assert model_name in van_settings.keys(),'我们暂时只用van-S'
        depths, embed_dims = van_settings[model_name]
        dpr = 0.
        mlp_ratios = [8,8,4,4]
        dpr = [x.item() for x in torch.linspace(0, dpr, sum(depths))]
        cur = 0
        for i in range(4):
            if i == 0:
                patch_embed = OverlapPatchEmbed(7,4,3,embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(3,2,embed_dims[i-1],embed_dims[i])
            block = nn.Sequential(
                *[Block(embed_dims[i],mlp_ratios[i],dpr[cur+j]) for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i],eps = 1e-6)
            cur += depths[i]
            setattr(self, f'patch_embed{i+1}',patch_embed)
            setattr(self, f'block{i+1}', block)
            setattr(self, f'norm{i+1}', norm)
        self.head = nn.Linear(embed_dims[-1],num_classes)
        
        self._init_weight(pretrained)
        
    def forward(self,x):
        B = x.shape[0]
        for i in range(4):
            x, H, W = getattr(self,f'patch_embed{i+1}')(x)
            x = getattr(self,f'block{i+1}')(x)
            x = x.flatten(2).transpose(1,2) # [B,C,HW] -> [B,HW,C]
            x = getattr(self, f'norm{i+1}')(x) #单纯为了做一个LN要进行这样的转置必要吗
            
            if i != 3 : #最后一轮进行分类输出
                x = x.reshape(B, H, W, -1).permute(0,3,1,2).contiguous()
        x = x.mean(dim = 1)
        x = self.head(x)
        return x
    def _init_weight(self,pretrained):
        if pretrained:
            try:
                self.load_state_dict(torch.load(pretrained,map_location='cpu')['state_dict'])
            except RuntimeError:
                pretrained_dict = torch.load(pretrained,map_location='cpu')['state_dict']
                pretrained_dict.popitem()
                pretrained_dict.popitem() #把分类头删掉
                self.load_state_dict(pretrained_dict,strict = False)
        else:
            raise NotImplementedError('请用预训练的权重来fine-tune，别自己从头训练')
    

    def extract_feature(self,x):
        # grad-CAM也是需要梯度的
        B = x.shape[0]
        outs = []
        for i in range(4):
            x, H, W = getattr(self, 'patch_embed{i+1}')(x)
            x = getattr(f'block{i+1}')(x)
            x = x.flatten(2).transpose(1,2)
            x = getattr(f'norm{i+1}')(x)
            x = x.reshape(B,H,W,-1).permute(0,3,1,2).contiguous()
            outs.append(x)
        return outs
            
        
if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    model = VAN(num_classes=7,pretrained='D:/grad1/internship/van_small_811.pth.tar')
    y = model(x)
    print(y.shape)
        
        
        
        
        
        
        
        
        
        
        
        


    
    
    
    
    
    
    


        
        
        