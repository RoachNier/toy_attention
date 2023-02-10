# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:42:22 2023

@author: 14935
精简了nanogpt，保留了骨干部分，去除了load_weight的部分，仅供学习
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size:int = 1024
    vocab_size:int = 50304 #为什么非要64的倍数
    n_layer:int = 12
    n_head:int = 12 #head_dim = 64
    n_embd:int = 768
    dropout:float = 0.0
    bias:bool = True

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Paramter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    
class SelfAttention(nn.Module):
    def __init__(self, config):
        # 直接传入config类
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout) # residual identity dropout
        self.n_head = config.n_head                    
        self.n_embd = config.n_embd
        self.dropout = config.dropout                  # 原nanogpt中还有flashattention(2022),但torch1.10版本没有，就略过
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size).reshape(1, 1, config.block_size, config.block_size)))
    def forward(self, x):
        B, L, C = x.size()
        q, k, v = self.c_atten(x).split(self.n_embd, dim=2) 
        k = k.reshape(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.reshape(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        q = v.reshape(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        atten = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(k.size(-1)))
        atten = atten.masked_fill(self.bias[:,:,:L,:L] == 0, float('-inf'))
        atten = self.atten_dropout(atten)
        y = atten @ v
        y = y.transpose(1, 2).reshape(B, L, C)
        y = self.resid_dropout(y)
        return y

class MLP(nn.module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.MouduleList([Block(config) for layer in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias)
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weight)
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6))
        
    def get_num_params(self, non_embedding=True):
        n_params = sum(param.numel() for param in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, target=None):
       device = idx.device
       b, l = idx.size() #batch, sentence_len
       assert l <= self.config.block_size, f"最大的长度是{self.config.block_size},请减小句子长度"
       pos = torch.arange(0, l, dtype=torch.long, device=device).unsqueeze(0)
       tok_emb = self.transformer.wte(idx) #B L C
       pos_emb = self.transformer.wpe(pos) #1 L C
       x = self.transformer.drop(tok_emb + pos_emb)
       for block in self.transformer.h:
           x = block(x)
       x = self.transformer.ln_f(x)
       
       if target is not None:
           logits = self.lm_head(x)
           loss = F.cross_entropy(logits.reshape(-1), target.reshape(-1), ignore_index=-1)
       else:
           logits = self.lm_head(x[:,[-1],:]) #只取最后时间点的logits
           loss = None
       return logits, loss
   
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
            
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # 每次拿到下一个元素,并cat在时间维度上，不断进行生成
        return idx
        
           
          
            
          
           
        
        
                      































