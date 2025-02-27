import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientMultiHeadWSA(nn.Module):
    def __init__(self, cfg, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.cfg = cfg
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.window_size = cfg.drsa.window_size  # (H, W)
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores

        # Linear layers for queries, keys, and values
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, H, W, C], where H and W are spatial dimensions, and C is the embedding dimension.
        Returns:
            Tensor of shape [B, H, W, C] after applying windowed multi-head self-attention.
        """
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        window_h, window_w = self.window_size

        # Ensure dimensions are divisible by window size (pad if needed)
        pad_h = (window_h - H % window_h) % window_h
        pad_w = (window_w - W % window_w) % window_w
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # Padding along H and W
        H_pad, W_pad = x.shape[1:3]

        # Divide into windows
        x = x.view(B, H_pad // window_h, window_h, W_pad // window_w, window_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_h * window_w, C)  # [num_windows*B, window_size, C]

        # Linear projections for Q, K, V
        qkv = self.qkv(x).reshape(-1, window_h * window_w, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each is [num_windows*B, window_size, num_heads, head_dim]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [num_windows*B, num_heads, window_size, window_size]
        attn = F.softmax(attn, dim=-1)

        # Compute attention-weighted values
        attn_output = (attn @ v)  # [num_windows*B, num_heads, window_size, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(-1, window_h * window_w, C)  # Combine heads

        # Project output back to embedding dimension
        attn_output = self.out_proj(attn_output)  # [num_windows*B, window_size, C]

        # Reshape back to original spatial dimensions
        attn_output = attn_output.view(B, H_pad // window_h, W_pad // window_w, window_h, window_w, C)
        attn_output = attn_output.permute(0, 1, 3, 2, 4, 5).reshape(B, H_pad, W_pad, C)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            attn_output = attn_output[:, :H, :W, :]

        attn_output = attn_output.permute(0, 3, 2, 1)
        return attn_output