import math

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SelfAttention(nn.Module):
    def __init__(self, dim: int,
                 heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # qkv.shape : [B N dim_head * 3] =>
        # qkv.shape : [3 B N dim_head]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
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
        return self.to_out(out)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CNNFeatureTransformer(nn.Module):
    def __init__(self, feature_model, feature_model_dim,
                 attn_dim_list, num_head_list, num_class,
                 dropout_proba=0.):
        super().__init__()
        inner_dim = None

        self.feature_model = feature_model
        self.positional_encoding = PositionalEncoding(
            dropout=dropout_proba)
        attn_layer_sequence = []
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            attn_layer = SelfAttention(inner_dim,
                                       heads=num_head, dim_head=attn_dim, dropout=dropout_proba)
            attn_layer_sequence.append(attn_layer)
            inner_dim = attn_dim * num_head
        self.attn_layer_sequence = nn.Sequential(*attn_layer_sequence)

        self.final_linear_sequence = nn.Sequential(
            nn.Linear(inner_dim, 2048),
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(2048, 1024),
            nn.ReLU6(),
            nn.Linear(1024, num_class),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        # input_tensor.shape: [B, C, Z, H, W]
        # feature_tensor.shape: [B*Z, C, H, W]
        feature_tensor = rearrange(input_tensor, 'b c z h w-> (b z) c h w')
        # feature_tensor.shape: [B*Z, C, H, W]
        feature_tensor = self.feature_model(feature_tensor)
        # feature_tensor.shape: [B, Z, H, W, C]
        feature_tensor = rearrange(
            input_tensor, '(b z) c h w-> b z h w c', b=batch_size)
        # feature_tensor.shape: [B, (H*W*Z), C]
        feature_tensor = torch.flatten(feature_tensor,
                                       start_dim=1, end_dim=3)
        # transfomer_tensor.shape: [B, H*W*Z, C]
        transfomer_tensor = self.attn_layer_sequence(feature_tensor)
        # transfomer_tensor.shape: [B, num_class]
        output_tensor = self.final_linear_sequence(transfomer_tensor)

        return output_tensor


class CNNFeatureTransformer2D(CNNFeatureTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        # input_tensor.shape: [B, 3*Z, H, W]
        # feature_tensor.shape: [B, C, H, W]
        feature_tensor = self.feature_model(input_tensor)
        feature_tensor = rearrange(feature_tensor, 'b c h w-> b (h w) c')
        # transfomer_tensor.shape: [B, H * W, C]
        transfomer_tensor = self.attn_layer_sequence(feature_tensor)
        # output_tensor.shape: [B, H*W*C]
        output_tensor = torch.flatten(transfomer_tensor,
                                      start_dim=1, end_dim=-1)
        # output_tensor.shape: [B, num_class]
        output_tensor = self.final_linear_sequence(output_tensor)

        return output_tensor
