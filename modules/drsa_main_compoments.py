import math
import torch
import numbers
import torch.nn as nn

from einops import rearrange
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # h, w = x.shape[-2:]
        return self.body(x)
    
## ------------------------------------
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
## ------------------------------------
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-5):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape, device='cuda'))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, device='cuda'))
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x_normalize = (x - mu) / torch.sqrt(sigma + self.epsilon)

        # scale and shift
        out = x_normalize * self.weight + self.bias
        return  out


## ------------------------------------
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, hw, c = x.shape
        x = rearrange(x, 'b c (h w) -> b c h w', h=int(math.sqrt(c)), w = int(math.sqrt(c)))
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = rearrange(x, 'b c h w -> b c (h w)', h=int(math.sqrt(c)), w= int(math.sqrt(c)))
        return x
    
## Baseline -----------------------------
class BaselineFeedForwardNet(nn.Module):
    """
    Position-wise feed-forward network for Transformer blocks.
    
    Args:
        d_model (int): Dimension of the input embeddings.
        d_ff (int): Hidden dimension for the feed-forward layer.
    """
    def __init__(self, d_model, ffn_expansion_factor=2, dropout=0.1, bias=False):
        super().__init__()
        hidden_features = int(d_model*ffn_expansion_factor)

        self.conv1 = nn.Conv2d(d_model, hidden_features*2, kernel_size=1, bias=bias)
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv2d(hidden_features*2, d_model, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        
        Returns:
            A tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear layer + ReLU
        x = self.conv1(x)
        x = self.gelu(x) 
        x = self.dropout(x)

        # Second linear layer
        x = self.conv2(x)
        x = self.dropout(x)
        return x

## ------------------------------------
class DRSATransformerBlock(nn.Module):
    def __init__(self, cfg, dim=None, channels=None, bias=False, LayerNorm_type='WithBias'):
        super(DRSATransformerBlock, self).__init__()

        self.dim = dim
        self.cfg = cfg
        self.channels = channels
        low_dim = cfg.drsa.lr_dim
        num_heads = cfg.drsa.num_head
        n, m = cfg.drsa.downsample_factor
        self.head_size = cfg.drsa.head_size    # specify the window-size

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DRSA_Attention(cfg, dim, num_heads, channels, self.head_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.dropout = nn.Dropout(0.1)
        self.norm3 = LayerNorm(low_dim*low_dim, LayerNorm_type)

        self.upsample = nn.ConvTranspose2d(channels, channels, n, n) 
        self.upsample3 = nn.ConvTranspose2d(channels, channels, m, m)
        if cfg.drsa.maxpool:
            self.downsample = nn.MaxPool2d(kernel_size=n, stride=n, padding=0) 
            self.downsample3 = nn.MaxPool2d(kernel_size=m, stride=m, padding=0)  
        else:
            self.downsample = nn.AvgPool2d(kernel_size=n, stride=n, padding=0)
            self.downsample3 = nn.AvgPool2d(kernel_size=m, stride=m, padding=0)

        self.aggregate = nn.Conv2d(channels, 1, kernel_size=(1, 1), stride=1)      
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        # self.project_out = nn.Linear(dim,dim, bias=False)

        ffn_expansion_factor = cfg.drsa.reduction_factor
        if cfg.drsa.gdfn_layer:            
            self.ffn = FeedForward(channels, ffn_expansion_factor, bias)
        else:
            self.ffn = BaselineFeedForwardNet(channels, ffn_expansion_factor)
        
    def forward(self, x, res=0):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)', h = h, w = w)  # (b, c, h*w) 
        if res==0:
            x = self.norm1(x)
        elif res==1:
            x = self.norm3(x)
        else:
            x = self.norm4(x)
        x = rearrange(x, 'b c (h w)-> b c h w ', h = h, w = w)  # (b, c, h, w)
        x = self.attn(x, res)
        return x
    
    def final_attention(self, xxlow, xlow, xhigh, residual, res):

        if res==0:
            out = xhigh
        elif res==1:
            xl = self.upsample(xlow)   # (b, c, h, w)
            out  = xl + xhigh  
        else:
            xl = self.upsample(xlow)
            xxl = self.upsample3(xxlow)
            out = xxl + xl + xhigh

        out = self.project_out(out)
        out = self.dropout(out)

        b, c, h, w = out.shape
        out = out + residual
        out = rearrange(out, 'b c h w-> b c (h w) ', h = h, w = w)  # residual + DRSA
        out = self.norm2(out)

        if self.cfg.drsa.gdfn_layer:
            out = out + self.ffn(out) 
            out = rearrange(out, 'b c (h w)-> b c h w ', h = h, w = w)  # (b, c, h, w)
        else:
            out = rearrange(out, 'b c (h w)-> b c h w', h = h, w = w)
            out1 = self.ffn(out)
            out = out + out1
            
        return out

## ------------------------------------    
## Multi-DConv Head Transposed Self-Attention (MDTA)
class DRSA_Attention(nn.Module):
    #DRSA_Attention(cfg, dim, num_heads, channels, self.head_size)
    def __init__(self, cfg, dim, num_heads, channels, head_size):
        super(DRSA_Attention, self).__init__()

        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.cfg = cfg
        scale = (channels // num_heads)
        self.nc = channels // num_heads
        self.temperature = nn.Parameter(torch.tensor(scale ** -0.5))
        
        self.head_size = head_size
        self.num_heads = num_heads

        self.q_Hi = nn.Linear(head_size, head_size, bias=False)
        self.k_Hi = nn.Linear(head_size ,head_size, bias=False)
        self.v_Hi = nn.Linear(head_size, head_size, bias=False)

        if cfg.drsa.with_drsa:
            self.q_Lo = nn.Linear(head_size, head_size, bias=False)
            self.k_Lo = nn.Linear(head_size, head_size, bias=False)
            self.v_Lo = nn.Linear(head_size, head_size, bias=False)
        
    def forward(self, x, res=0):
        b, c, h, w = x.shape

        if res==0:
            head_size = self.head_size
        elif res==1:
            head_size = self.cfg.drsa.ld_head_size 
        else:
            head_size = self.cfg.drsa.ld_head_size3
        
        # window SA is used, the input is split on the spatial direction and not on the channel as the normal MHSA
        # this create n small windows  ~ self.head_size vs self.num_heads
        x = rearrange(x, 'b c h w -> b c (h w)')    # (b, c, h*w)    
        x = rearrange(x, 'b c (hw head) -> b hw c  head', head=head_size) # (b, h*w / hsize, c, hsize)
        
        # Compute qkv
        #qkv_Hi = self.to_qkv_Hi(x_Hi)  # [b, c*3, h, w]
        #q_Hi,k_Hi,v_Hi = qkv_Hi.chunk(3, dim=1) # ~ qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        
        if res==0:
            q = self.q_Hi(x) # (b, h*w / hsize, c, hsize)
            k = self.k_Hi(x) # ~
            v = self.v_Hi(x) # ~            
        elif res==1:
            q = self.q_Lo(x)  # (b, h*w / hs, c, hs)
            k = self.k_Lo(x)
            v = self.v_Lo(x)
        else:
            q = self.q_Lo3(x) 
            k = self.k_Lo3(x) 
            v = self.v_Lo3(x)             
        
        q = rearrange(q, 'b head c  hw -> b head hw c')  # (b, h*w / hsize, hsize, c)
        k = rearrange(k, 'b head c  hw -> b head hw c')  # ~
        v = rearrange(v, 'b head c  hw -> b head hw c')  # ~

        if self.cfg.drsa.is_mhsa:
            q = rearrange(q, 'b head hw (nc final) -> b head hw nc final', nc=self.num_heads)
            k = rearrange(k, 'b head hw (nc final) -> b head hw nc final', nc=self.num_heads)
            v = rearrange(v, 'b head hw (nc final) -> b head hw nc final', nc=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (n, h*w / hsize, hsize, hsize)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)  # (n, h*w / hsize, hsize, c)

        if self.cfg.drsa.is_mhsa: # convert back
            out = rearrange(out, 'b head hw nc final -> b head hw (nc final)') 
        
        out = rearrange(out, 'b head hw c -> b head c hw') # (n, h*w / hsize, c, hsize)
        out = rearrange(out, 'b hw c head -> b c (hw head) ', head = head_size) # (n, c, h*w)
        out= rearrange(out, 'b c (h w) -> b c h w ', h = h, w = w) # (n, c, h, w)

        return out 
    
##############################
##############################

## ------------------------------------
class DRSAConvTransformerBlock(nn.Module):
    def __init__(self, cfg, dim=None, channels=None, bias=False, LayerNorm_type='WithBias'):
        super(DRSAConvTransformerBlock, self).__init__()

        self.dim = dim
        self.cfg = cfg
        self.channels = channels
        low_dim = cfg.drsa.lr_dim
        
        num_heads = cfg.drsa.num_head
        n, m = cfg.drsa.downsample_factor # n=2
        self.head_size = cfg.drsa.head_size    # specify the window-size
 
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DRSA_ConvAttention(cfg, dim, num_heads, channels, self.head_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.dropout = nn.Dropout(0.1)
        self.norm3 = LayerNorm(low_dim*low_dim, LayerNorm_type)

        self.upsample = nn.ConvTranspose2d(channels, channels, n, n) 
        self.upsample3 = nn.ConvTranspose2d(channels, channels, m, m)

        if cfg.drsa.maxpool:
            self.downsample = nn.MaxPool2d(kernel_size=n, stride=n, padding=0)  
        else:
            self.downsample = nn.AvgPool2d(kernel_size=n, stride=n, padding=0)

        self.aggregate = nn.Conv2d(channels, 1, kernel_size=(1,1), stride=1)      
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

        ffn_expansion_factor = cfg.drsa.reduction_factor
        if cfg.drsa.gdfn_layer:            
            self.ffn = FeedForward(channels, ffn_expansion_factor, bias)
        else:
            self.ffn = BaselineFeedForwardNet(channels, ffn_expansion_factor)
    
    def forward(self, x, res=0):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)', h = h, w = w)  # (b, c, h*w) 
        if res==0:
            x = self.norm1(x)
        else:
            x = self.norm3(x)

        x = rearrange(x, 'b c (h w)-> b c h w ', h = h, w = w)  # (b, c, h, w)
        x = self.attn(x, res)
        return x
    
    def final_attention(self, xxlow, xlow, xhigh, residual, res):
        if res==0:
            out = xhigh
        else:
            xl = self.upsample(xlow)   # (b, c, h, w)
            out  = xl + xhigh 

        out = self.project_out(out)
        out = self.dropout(out)

        b, c, h, w = out.shape
        out = out + residual
        out = rearrange(out, 'b c h w-> b c (h w) ', h = h, w = w)  # residual + DRSA
        out = self.norm2(out)

        if self.cfg.drsa.gdfn_layer:
            out = out + self.ffn(out) 
            out = rearrange(out, 'b c (h w)-> b c h w ', h = h, w = w)  # (b, c, h, w)
        else:
            out = rearrange(out, 'b c (h w)-> b c h w', h = h, w = w)
            out1 = self.ffn(out)
            out = out + out1
            
        return out

## ------------------------------------
class DRSA_ConvAttention(nn.Module):
    def __init__(self, cfg, dim, num_heads, channels, head_size):
        super(DRSA_ConvAttention, self).__init__()

        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.cfg = cfg
        scale = (channels // num_heads)
        self.nc = channels // num_heads
        self.temperature = nn.Parameter(torch.tensor(scale ** -0.5))

        self.head_size = head_size
        in_dim = int(dim/head_size)
        self.num_heads = num_heads

        self.q_Hi = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.k_Hi = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.v_Hi = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)

        if cfg.drsa.with_drsa:
            head_size = cfg.drsa.ld_head_size
            dim = cfg.drsa.lr_dim * cfg.drsa.lr_dim
            in_dim = int(dim/head_size)
            self.q_Lo = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
            self.k_Lo = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
            self.v_Lo = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        
    def forward(self, x, res=0):
        b, c, h, w = x.shape   

        if res==0:
            head_size = self.head_size
        else res==1:
            head_size = self.cfg.drsa.ld_head_size   
        
        # window SA is used, the input is split on the spatial direction and not on the channel as the normal MHSA
        # this create n small windows  ~ self.head_size vs self.num_heads
        x = rearrange(x, 'b c h w -> b c (h w)')    # (b, c, h*w)    
        x = rearrange(x, 'b c (hw head) -> b hw c  head', head=head_size) # create windows ~ (b, h*w / hsize, c, hsize)
        
        # Compute qkv
        #qkv_Hi = self.to_qkv_Hi(x_Hi)  # [b, c*3, h, w]
        #q_Hi,k_Hi,v_Hi = qkv_Hi.chunk(3, dim=1) # ~ qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        if res==0:
            q = self.q_Hi(x) # (b, h*w / hsize, c, hsize)
            k = self.k_Hi(x) # ~
            v = self.v_Hi(x) # ~
        else:            
            q = self.q_Lo(x)  # (b, h*w / hs, c, hs)
            k = self.k_Lo(x)
            v = self.v_Lo(x) 
        
        q = rearrange(q, 'b head c  hw -> b head hw c')  # (b, h*w / hsize, hsize, c) 
        k = rearrange(k, 'b head c  hw -> b head hw c')  # ~
        v = rearrange(v, 'b head c  hw -> b head hw c')  # ~
        
        if self.cfg.drsa.is_mhsa:
            q = rearrange(q, 'b head hw (nc final) -> b head hw nc final', nc=self.num_heads)
            k = rearrange(k, 'b head hw (nc final) -> b head hw nc final', nc=self.num_heads)
            v = rearrange(v, 'b head hw (nc final) -> b head hw nc final', nc=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        #attn = (q @ k.transpose(-2, -1)) * self.temperature  # (n, h*w / hsize, hsize, hsize)
        attn = (q @ k.transpose(-2, -1)) 
        attn = attn * self.temperature  # (n, h*w / hsize, hsize, hsize)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)  # (n, h*w / hsize, hsize, c)
        
        if self.cfg.drsa.is_mhsa: # convert back
            out = rearrange(out, 'b head hw nc final -> b head hw (nc final)') 
        
        out = rearrange(out, 'b head hw c -> b head c hw') # (n, h*w / hsize, c, hsize)
        out = rearrange(out, 'b hw c head -> b c (hw head) ', head = head_size) # (n, c, h*w)
        out= rearrange(out, 'b c (h w) -> b c h w ', h = h, w = w) # (n, c, h, w)
        
        return out