import math
import torch
import numbers
import torch.nn as nn

from einops import rearrange
from torch.nn import functional as F


## Custom DRSA
class LinearAttention(nn.Module):
    def __init__(self, cfg, in_dim, heads=8):
        super(LinearAttention, self).__init__()
        assert in_dim % heads == 0, "Embedding dimension must be divisible by number of heads"

        self.cfg = cfg
        self.num_heads = heads
        self.head_dim = in_dim // heads
        self.scale = (in_dim // heads) ** -0.5
        self.window_size = cfg.drsa.window_size

        # Linear projections for query, key, and value
        self.to_qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        #self.q_proj = nn.Linear(in_dim, in_dim)
        #self.k_proj = nn.Linear(in_dim, in_dim)
        #self.v_proj = nn.Linear(in_dim, in_dim)

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.to_out = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(0.1)  # Optional dropout

    def forward(self, x):
        b, c, h, w = x.shape              # (b, 2048, 60, 60)
        x = x.flatten(2).transpose(1, 2)  # [b, c, h*w] => [b, h*w, c]
        residual = x
        x = self.norm1(x)

        # Linear projection + Split into multiple heads
        qkv = self.to_qkv(x).chunk(3, dim=-1) # chunk create tuple with Q, K V
        q, k, v = map(lambda t: t.reshape(b, h*w, self.num_heads, -1).transpose(1, 2), qkv)
        # q.shape = (b, nh, h*w, c/nh)  nh = num_heads
        #q = self.q_proj(x)
        #k = self.k_proj(x)
        #v = self.v_proj(x)
        #q = q.reshape(b, h*w, self.heads, -1).transpose(1, 2)
        #k = k.reshape(b, h*w, self.heads, -1).transpose(1, 2)
        #v = v.reshape(b, h*w, self.heads, -1).transpose(1, 2)      

        # Apply windowed attention  
        if self.cfg.drsa.window:
            ss = (b, self.num_heads, h, w, self.head_dim)
            out = self.windowed_attention(q, k, v, ss)  # [B, H, L, D/H]
            out = out.reshape(b, h*w, -1)
        else:        
            dots = (q @ k.transpose(-1, -2)) * self.scale           # (b, nh, h*w, h*w)
            attn = dots.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, h*w, -1)    # (b, h*w, c)

        # Final linear projection    
        out = self.to_out(out)

        out = self.dropout(out)  # Optional dropout
        out = out + residual  # First residual connection

        residual = out
        out = self.norm2(out)

        out = out + residual  # Second residual connection
        return out.transpose(1, 2).reshape(b, c, h, w)
    

    def windowed_attention(self, q, k, v, hh):
        """
        Compute attention with a sliding 2D window approach.
        """
        Q = q.reshape(hh[0], hh[1], hh[2], hh[3], hh[4])
        K = q.reshape(hh[0], hh[1], hh[2], hh[3], hh[4])
        V = q.reshape(hh[0], hh[1], hh[2], hh[3], hh[4])

        batch_size, num_heads, height, width, head_dim = Q.size()
        pad_h, pad_w = self.window_size[0] // 2, self.window_size[1] // 2

        # Pad K and V for sliding window
        K_padded = F.pad(K, (0, 0, pad_w, pad_w, pad_h, pad_h), value=0)  # [B, H, H+2*pad_h, W+2*pad_w, D/H]
        V_padded = F.pad(V, (0, 0, pad_w, pad_w, pad_h, pad_h), value=0)

        outputs = []
        for i in range(height):
            row_outputs = []
            for j in range(width):
                # Extract 2D window
                K_window = K_padded[:, :, i:i+self.window_size[0], j:j+self.window_size[1], :]  # [B, H, W_H, W_W, D/H]
                V_window = V_padded[:, :, i:i+self.window_size[0], j:j+self.window_size[1], :]  # [B, H, W_H, W_W, D/H]

                # Compute attention scores
                Q_ij = Q[:, :, i, j, :].unsqueeze(2).unsqueeze(2)  # [B, H, 1, 1, D/H]
                scores = torch.einsum("bhijc,bhklc->bhijkl", Q_ij, K_window)  # [B, H, 1, 1, W_H, W_W]
                scores = scores / (head_dim ** 0.5)  # Scale scores
                attn_weights = F.softmax(scores.view(batch_size, num_heads, -1), dim=-1).view(scores.size())  # [B, H, 1, 1, W_H, W_W]

                # Weighted sum of values
                attn_output = torch.einsum("bhijkl,bhklc->bhijc", attn_weights, V_window)  # [B, H, 1, 1, D/H]
                row_outputs.append(attn_output.squeeze(3).squeeze(2))  # [B, H, D/H]

            outputs.append(torch.stack(row_outputs, dim=2))  # [B, H, W, D/H]
        outputs = torch.stack(outputs, dim=2)  # [B, H, H', W', D/H]
        outputs = outputs.reshape(hh[0], hh[1], hh[2]*hh[3], hh[4])
        return outputs


