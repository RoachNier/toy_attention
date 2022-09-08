# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:15:19 2022
只关注swinT的骨干核心，快速复现一下，真想要用还是去Github下一个，还有pretrain的模型呢
@author: 14935
"""

import torch
import torch.nn as nn
from torch.utils import checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_




class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features = None, out_features = None, act_layer= nn.GELU,dropout = 0.):
        super().__init__()
        hidden_features = hidden_features if hidden_features else in_features
        out_features = out_features if out_features else in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(dropout)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x,window_size):
    '''
        这里就暗含了window_size必须能整除x的高宽，不然有数据剩余凑不成一个完整的维度，就会报错
        和ViT的做法有点类似，但处理的结果不同:
        ViT:[B,C,H,W] -> [B,#P,DP],中间也经历了六维的变化
        ViT的patchnify处理方法是：
            1、不管batch维，首先将channel permute到最后,然后contiguous()
            2、对imh//ph进行拆分，用reshape
            3、对-2,-3维进行转置，然后对imw//pw用reshape进行拆分
            4、对-2 -3维进行转置
            5、reshape成想要的三维的形状
        SwinT:[B,H,W,C] -> [B*#Window,window_size,window_size,C]
        SwinT的处理方法更简洁一些：
            1、输入的x已经将C排在了最后一维，不需要再permute了
            2、直接对H W维进行拆分，一次拆两个
            3、permute 2 3 维
            4、reshape成为想要的四维形状
        花了两天时间，苦思冥想，终于想出来这种六维tensor的处理方法，如果求方便，就用爱因斯坦表示法
        
    '''
    B,H,W,C = x.shape
    windows = x.reshape(B,H//window_size,window_size,W//window_size,window_size,C)
    windows = windows.permute(0,1,3,2,4,5).contiguous().reshape(-1,window_size,window_size,C)
    return windows

def window_reverse(windows,window_size,H,W):
    '''是window_partition的反函数'''
    B = int(windows.shape[0]/(H/window_size)/(W/window_size))
    x = windows.reshape(B,H//window_size,W//window_size,window_size,window_size,-1)
    x = windows.permute(0,1,3,2,4,5).contiguous().reshape(B,H,W,-1) #一起拆的就一起合起来
    return x

class WindowAttention(nn.Module):
    '''我只能说太离谱了'''
    def __init__(self,dim,window_size,num_heads,qkv_bias=True,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        self.scale = qk_scale if qk_scale else self.head_dim** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1),num_heads)) #构造的相对位置表是相对位置索引的2倍+1，因此随便取，不会越界
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h,coords_w]))#对应一个window的编码，但是有两个张量
        #coords存的是每一个patch(或者说等效的像素)在图片中的位置，理解为绝对位置，以左上角为(0,0)原点
        coords_flatten = torch.flatten(coords,1)#按行展平，其实是将图片的绝对位置编码拉平，和传统AlexNet最后分类头一样，只不过将feature map换成了位置编码而已
        relative_coords = coords_flatten[:,:,None]-coords_flatten[:,None,:]#竖的减横的，是(xi-xj,yi-yj),相当于以每一个patch为基准点，计算其他patch的相对位置
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += self.window_size[0] -1
        relative_coords[:,:,1] += self.window_size[1] -1 #从0开始
        relative_coords[:,:,0] *= 2*self.window_size[1]-1 #上面展平了，下面又要加起来，如果不对(x,y)坐标有所区分的话，基准点斜侧经过相加后的相对位置坐标就一样了，这样是为了在二维上区分不同的点
        relative_position_index = relative_coords.sum(-1) #把相对坐标的x和y加起来
        self.register_buffer('relative_position_index',relative_position_index)
        #有一说一，这relative_position_index和吹一样，我也能吹个完全不一样的
        self.qkv = nn.Linear(dim,dim*3,bias = qkv_bias) #ViT就是False，这里就是True，真的让我很难做人啊大哥
        self.atten_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table,std = .02)
        self.softmax = nn.Softmax(dim = -1)
    def forward(self,x,mask = None):
        B_,N,C = x.shape #[#windows*B,ws*ws,C]
        
        qkv = self.qkv(x).reshape(B_,N,3,self.num_heads,C//self.num_heads)#qkv之后C->3C
        qkv = qkv.permute(2,0,3,1,4) #这也太玄乎了我操
        q,k,v = qkv[0],qkv[1],qkv[2]
        
        #######如果换种方式实现##########
        #self.wq, self.wk, self.wv = nn.Linear(dim,dim),nn.Linear(dim,dim),nn.Linear(dim,dim)
        #q,k,v = self.wq(q), self.wk(k), self.wv(v) #[B_,N,C]
        #B_,N,C = q.shape
        #q = q.reshape(B_,N,self.num_heads,C//self.num_heads).transpose(1,2)
        #k = k.reshape(B_,N,self.num_heads,C//self.num_heads).transpose(1,2)
        #v = v.reshape(B_,N,self.num_heads,C//self.num_heads).transpose(1,2)
        ##############################
        
        q = q * self.scale
        attn = q @ k.transpose(-1,-2)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
            self.window_size[0]*self.window_size[1],self.window_size[0]*self.window_size[1],-1)
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_//nW,nW,self.num_heads,N,N)+mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1,self.num_heads,N,N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.atten_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    '''里面就包含了shifted window和 window attention的部分'''
    def __init__(self,dim,input_resolution,num_heads,window_size=7,shift_size=0,
                 mlp_ratio=4,qkv_bias=True,qk_scale=None,proj_drop=0.,attn_drop=0.,
                 drop_path=0.,fused_window_process=False,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size <= self.window_size, 'torch.roll的行列个数得在0 ~ window size 之间'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size, num_heads,
                                    qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,dropout=proj_drop)
        if self.shift_size>0:
            H,W = self.input_resolution
            img_mask = torch.zeros((1,H,W,1))
            h_slices = (slice(0, -self.window_size), #妈的，花了半个小时在疑惑负号是啥意思上，其实就是把feature map分成了三份，和论文里的示意图一样
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt #这个生成后，就已经对应着roll之后的attn位置了
                    cnt += 1
            mask_windows =  window_partition(img_mask, self.window_size)#[1*nW,ws,ws,1]
            mask_windows = mask_windows.reshape(-1,self.window_size*self.window_size) #[num_windows,ws*ws]
            attn_mask = mask_windows[:,None,:] - mask_windows[:,:,None] #unsqueeze也行
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100)).masked_fill(attn_mask==0,float(0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask',attn_mask)
        self.fused_window_process = fused_window_process
    def forward(self,x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, '输入的featuremap高宽必须和输入的分辨率参数匹配'
        
        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B,H,W,C)
        if self.shift_size >0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims = (1,2))
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                print('没装windowprocess,如果要融合窗口操作的话，请去github上跟着install.md安装swintransformer')
                return NotImplementedError
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1,self.window_size*self.window_size,C)
        attn_windows = self.attn(x_windows,mask = self.attn_mask)
        attn_windows = attn_windows.reshape(-1,self.window_size,self.window_size,C)
        #上面对shift_size>0的情况进行了roll，接下来要将图片还原
        if self.shift_size>0:
            attn_windows = window_reverse(attn_windows, self.window_size, H, W)
            attn_windows = attn_windows.roll(shifts = (-self.shift_size,-self.shift_size),dim = (1,2))
        else:
            attn_windows = window_reverse(attn_windows, self.window_size,H,W)
        x = attn_windows
        x = x.resahpe(B,-1,C)
        x = shortcut + self.drop_path(self.mlp(self.norm2(x)))
        return x
        
        
        
class PatchEmbed(nn.Module):
    def __init__(self,img_size=224,patch_size=4,in_chans=3,embed_dim=96,norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) #搞成tuple
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] #把patch当作一个像素单位，计算高宽上的patch个数
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)#竟然是卷积直接到新维度，爷刚学的patchnify啊
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    def forward(self,x):
        B,C,H,W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], '把模型定义的图片尺寸和实际输入搞对，射射'
        x = self.proj(x).flatten(2).transpose(1,2) #展平h w维度并交换后两维位置，这他妈的也太简洁了
        if self.norm is not None:                  #vit白学了，还是这个简洁，而且参数量会多num_patches倍
            x = self.norm(x)
        return x
    def flops(self):
        '''flops计算的仅仅是乘法的次数'''
        return NotImplementedError
    
class PatchMerging(nn.Module):
    def __init__(self,input_resolution,dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim,2*dim,bias=False)
        self.norm = norm_layer(4*dim)
    def forward(self,x):
        #x:[B,H*W,C]
        H, W = self.input_resolution 
        B, L, C = x.shape
        assert L == H * W, '每一个PacthMerging的feature map分辨率应该一样'
        assert H % 2 == 0 and W % 2 == 0, 'feature map 的高宽必须为偶数，才能1/2降采样'
        x = x.permute(0,2,3,1).contiguous()
        x0 = x[:,0::2,0::2,:]
        x1 = x[:,1::2,0::2,:]
        x2 = x[:,0::2,1::2,:]
        x3 = x[:,1::2,1::2,:]
        x = torch.cat([x0,x1,x2,x3],dim = -1) #x:[B,H/2,W/2,4*C]
        x = x.reshape(B,-1,4*C)
        x = self.norm(x)
        x = self.reduction(x)  #想要控制特征维度，加Linear就好了
        return x
        
class BasicLayer(nn.Module):
    def __init__(self,dim,input_resolution,depth,num_heads,window_size,
                 mlp_ratio=4.,qkv_bias=True,qk_scale=None,drop=0.,attn_drop=0.,
                 drop_path=0.,norm_layer=nn.Layernorm,downsample=None,use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim,input_resolution,num_heads,window_size=window_size,
                                 shift_size=0 if (_ %2 ==0) else window_size //2,
                                 mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale = qk_scale,
                                 drop=drop, attn_drop=attn_drop,drop_path=drop_path[_] if isinstance(drop_path,list) else drop_path,
                                 norm_layer = norm_layer,fused_window_process=fused_window_process
                            ) 
            for _ in range(depth)])    
        #和原论文不很匹配，这里patch_merging是放在swintransformer block之后的
        if downsample is not None:
            self.downsample = downsample(input_resolution,dim=dim,norm_layer=norm_layer)
        else:
            downsample = None
    def forward(self,x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, )
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
            
class SwinTransformer(nn.Module):
    def __init__(self,img_size=224,patch_size=4,in_chans=3,num_classes=1000,
                     embed_dim=96,depths=[2,2,6,2],num_heads=[3,6,12,24],
                     window_size=7,mlp_ratio=4.,qkv_bias=True,qk_scale=None,
                     drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,
                     norm_layer=nn.LayerNorm,ape=False,patch_norm=True,
                     use_checkpoint=False,fused_window_process=False,**kwargs):
        super().__init__()
        self.num_classes = num_classes #还是图像分类问题，但应该是可以将swinT当作backbone适用于不同的任务，但爷要先去看detr了
        self.num_layer = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim*2**(self.num_layer-1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=in_chans,
                                      embed_dim=embed_dim,norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_position_embedding = nn.Parameter(torch.zeros(1,num_patches,embed_dim))
            trunc_normal_(self.absolute_position_embedding,std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.items() for x in torch.linspace(0, drop_path_rate,sum(depths))]
        self.layers = nn.ModuleList()
        for ith_layer in range(self.num_layer):
            layer = BasicLayer(dim=int(embed_dim*2**ith_layer),
                               input_resolution=(patches_resolution[0]//(2**ith_layer),patches_resolution[1]//(2**ith_layer)),
                               depth = depths[ith_layer],num_heads=num_heads[ith_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:ith_layer]):sum(depths[:ith_layer+1])],
                               norm_layer=norm_layer,downsample=PatchMerging if ith_layer< self.num_layer-1 else None,
                               use_checkpoint=use_checkpoint,fused_window_process=fused_window_process
                               )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features) #最后一个transformer block输出后，要接检测头或分类头部分
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Indentity()
    def forward_features(self,x):
        x = self.patch_embed(x)
        if self.ape:
            x = x+ self.absolute_position_embedding
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1,2)) #本来x是[B,L,C],transpose后是[B,C,L],再做1d的avgpool相当于是对每个channel都取了2davgpool
        x = torch.flatten(x,1) #[B,C]
        return x
    def forward(self,x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
#复现完SwinT，有种酣畅淋漓的快感，再搞完detr，transformer这一块就暂时收工
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    