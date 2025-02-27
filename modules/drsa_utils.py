import math
import torch
import numbers
import torch.nn as nn

from einops import rearrange
from torch.nn import functional as F

## ------------------------------------    
class TransformerBlock(nn.Module):
    def __init__(self, dim=None, head_size=10, channels=None, num_heads=None, ffn_expansion_factor=None, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.head_size = head_size
        self.channels = channels
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)', h = h, w = w)
        x = self.norm1(x)
        x = x + self.attn(x)
        x = x + self.ffn(self.norm2(x))
        x = rearrange(x, 'b c (h w) -> b c h w', h = h, w = w)
        return x

## ------------------------------------    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, device):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, device=device))

        self.q = nn.Linear(dim, dim, bias=False).to(device)
        self.k = nn.Linear(dim, dim, bias=False).to(device)
        self.v = nn.Linear(dim, dim, bias=False).to(device)
        self.project_out = nn.Linear(dim, dim, bias=False).to(device)

    def forward(self, x):
        q = self.q(x) # (bs, C, HW)
        k = self.k(x)
        v = self.v(x) 
        
        q = rearrange(q, 'b feature seq -> b seq feature') # (bs, HW, C)
        k = rearrange(k, 'b feature seq -> b seq feature')
        v = rearrange(v, 'b feature seq -> b seq feature')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b seq feature -> b feature seq ')
        out = self.project_out(out)
        return out