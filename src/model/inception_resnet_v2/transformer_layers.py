import math

import torch
from torch import nn
from einops import rearrange

INPLACE = False


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, 1)
        pe[:, 0::2, 0] = torch.sin(position * div_term)
        pe[:, 1::2, 0] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inner_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # qkv.shape : [B N dim_head * 3] =>
        # qkv.shape : [3 B N dim_head]
        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        # qkv.shape = [3 B num_head N dim]
        # q.shape = [B num_head N dim]
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # q.shape [B num_head N dim]
        # k.shape [B num_head dim N]
        # dots.shape [B num_head N N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        # attn.shape [B num_head N N]
        # v.shape [B num_head N dim]
        # out.shape [B num_head N dim]
        out = torch.matmul(attn, v)
        # out.shape [B N (num_head * dim)]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = heads * dim_head
        self.attn = SelfAttention(heads, dim_head, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(inner_dim, eps=1e-6)
        self.ffpn_dense_1 = nn.Linear(inner_dim, inner_dim * 4, bias=False)
        self.ffpn_act_1 = nn.ReLU6(inplace=INPLACE)
        self.ffpn_dropout_1 = nn.Dropout(dropout)
        self.ffpn_dense_2 = nn.Linear(inner_dim * 4, inner_dim, bias=False)
        self.ffpn_act_2 = nn.ReLU6(inplace=INPLACE)
        self.ffpn_dropout_2 = nn.Dropout(dropout)
        self.ffpn_norm = nn.LayerNorm(inner_dim, eps=1e-6)

    def forward(self, x):

        attn = self.attn(x)
        attn = self.attn_dropout(attn)
        attn = self.attn_norm(x + attn)

        out = self.ffpn_dense_1(attn)
        out = self.ffpn_act_1(out)
        out = self.ffpn_dropout_1(out)
        out = self.ffpn_dense_2(out)
        out = self.ffpn_act_2(out)
        out = self.ffpn_dropout_2(out)
        out = self.ffpn_norm(attn + out)

        return out


class TransformerEncoder2D(TransformerEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        attn = self.attn(x)
        attn = self.attn_dropout(attn)
        attn = self.attn_norm(x + attn)

        out = self.ffpn_dense_1(attn)
        out = self.ffpn_act_1(out)
        out = self.ffpn_dropout_1(out)
        out = self.ffpn_dense_2(out)
        out = self.ffpn_act_2(out)
        out = self.ffpn_dropout_2(out)
        out = self.ffpn_norm(attn + out)
        out = rearrange(out, 'b (h w) c -> b c h w',
                        h=H, w=W)
        return out


class TransformerEncoder3D(TransformerEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        _, _, Z, H, W = x.shape
        x = rearrange(x, 'b c z h w-> b (z h w) c')
        attn = self.attn(x)
        attn = self.attn_dropout(attn)
        attn = self.attn_norm(x + attn)

        out = self.ffpn_dense_1(attn)
        out = self.ffpn_act_1(out)
        out = self.ffpn_dropout_1(out)
        out = self.ffpn_dense_2(out)
        out = self.ffpn_act_2(out)
        out = self.ffpn_dropout_2(out)
        out = self.ffpn_norm(attn + out)
        out = rearrange(out, 'b (z h w) c -> b c z h w',
                        z=Z, h=H, w=W)
        return out
