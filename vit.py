# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:21:34 2022

@author: 14935
"""

import torch
import torch.nn as nn

def pair(t):
    return t if isinstance(t, tuple) else (t,t) #将输入改为元组输出,ViT类中判断图片尺寸输入的是int还是tuple

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super(PreNorm,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn # atten 和 ffn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)
    
class FeedForwardModule(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super(FeedForwardModule,self).__init__()
        self.net = nn.Sequential(
                                nn.Linear(dim,hidden_dim),
                                nn.GELU(),  #gelu与tanh近似
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim,dim),
                                nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self,dim,heads = 8, dim_head = 64, dropout = 0.):
        super(Attention,self).__init__()
        inner_dim = dim_head * heads
        project_out = not(heads==1 and dim_head ==dim) #如果是单头，project_out=False,对应着MHSA升维的情况，再给降回去
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim,inner_dim*3,bias = False) #用chunk拆开，属于是炫技了，定义三个linear就行
        self.to_out = nn.Sequential(
                                    nn.Linear(inner_dim,dim),
                                    nn.Dropout(dropout)) if project_out else nn.Identity()
    def forward(self,x):
        '''还是自注意力'''
        q,k,v = self.to_qkv(x).chunk(3,dim = -1) #在dim的维度将tensor进行切分，并返回tuple
        #分割为多个头
        bs, max_len, d_model = x.shape
        assert d_model % self.heads ==0, print('''输入特征维度必须被头的个数整除！，
                                               请调整你的输入维度，或者调整头的个数''')
        #print(q.shape,d_model,self.heads)
        q = q.reshape(bs,max_len,self.heads,-1).transpose(1,2)
        k = k.reshape(bs,max_len,self.heads,-1).transpose(1,2)
        v = v.reshape(bs,max_len,self.heads,-1).transpose(1,2)
        score = self.attend(q @ k.transpose(-1,-2))/self.scale
        score = self.dropout(score)
        output = score @ v
        output = output.transpose(1,2).reshape(bs,max_len,-1)
        return self.to_out(output) #如果是多头的话

class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,dim_head,mlp_dim,dropout = 0.):
        super(Transformer,self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                                nn.ModuleList([
                                    PreNorm(dim, Attention(dim,heads,dim_head,dropout = dropout)),
                                    PreNorm(dim,FeedForwardModule(dim, mlp_dim,dropout = dropout))
                             ]))
    def forward(self,x):
        for attn, ffn in self.layers: #遍历加解包
            x = x + attn(x)
            x = x + ffn(x)
        return x
    
class ViT(nn.Module):
    def __init__(self,image_size,patch_size,num_classes,dim,depth,heads,mlp_dim,
               pool='cls',channels = 3, dim_head= 64,dropout= 0., emb_dropout=0.):
        super(ViT,self).__init__()
        self.imh,self.imw = pair(image_size)
        self.ph,self.pw = pair(patch_size) #二维数据
        self.channels = channels
        assert self.imh % self.ph ==0 and self.imw % self.pw ==0, print('图片的尺寸必须是图片快尺寸的整数倍，请调整你的patch_size')
        self.num_patches = (self.imh//self.ph) * (self.imw//self.pw)
        self.patch_dim = self.channels * self.ph * self.pw #对patch进行attention其实是对图片进行缩小后，将w h channel展平，再施加多头注意力，此时一张图片如同一句话，而深度方向是多张图片
        assert pool in {'cls','mean'}, print('pool必须是cls token 或者 mean pooling')
        self.to_patch_embedding = nn.Linear(self.patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patches+1,dim)) #多一个cls token, 并且pe是需要训练的
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim,depth,heads,dim_head,mlp_dim,dropout=dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
                                    nn.LayerNorm(dim),
                                    nn.Linear(dim, num_classes))
    def forward(self,img):
        #对patch进行embedding,[bs,channel,imh,imw] -> [bs,num_patches,ph*pw*channels]
        bs= img.shape[0]
        patch = img.permute(0,2,3,1)
        patch = patch.reshape(bs,self.imh//self.ph,self.ph,-1,self.channels)
        patch = patch.transpose(-3,-2).reshape(bs,self.imh//self.ph,self.ph,self.imw//self.pw,self.pw,self.channels).transpose(-2,-3)
        patch = patch.reshape(bs,(self.imh//self.ph)*(self.imw//self.pw),self.patch_dim)
        patch = self.to_patch_embedding(patch)
        
        #上面 已经形成了patch_embedding:[bs,num_patches,dim]
        bs, num_patches,dim = patch.shape
        cls_tokens = self.cls_token.repeat(bs,1,1)
        patch = torch.cat([cls_tokens,patch],dim = 1)
        patch += self.pos_embedding
        patch = self.dropout(patch)
        patch = self.transformer(patch) 
        patch = patch.mean(dim =1) if self.pool =='mean' else patch[:,0] #patch变成二维，[bs,dim_after_pool]
        patch = self.to_latent(patch)
        return self.mlp_head(patch) #之后就是分类头
        
        
        
if __name__ =='__main__':
    bs = 2
    channel = 3
    image_size = 224
    patch_size = 7
    num_classes = 4
    dim = 512
    depth = 3
    heads = 8
    mlp_dim = 64
    model = ViT(image_size,patch_size,num_classes,dim,depth,heads,mlp_dim,)
    images = torch.randn(bs,channel,image_size,image_size)
    
    output = model(images)
    
    
    
    
    
    
    
    
    














