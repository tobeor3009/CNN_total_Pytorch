import math

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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
        self.ffpn_dense_2 = nn.Linear(inner_dim * 4, inner_dim, bias=False)
        self.ffpn_dropout = nn.Dropout(dropout)
        self.ffpn_norm = nn.LayerNorm(inner_dim, eps=1e-6)

    def forward(self, x):
        attn = self.attn(x)
        attn = self.attn_dropout(attn)
        attn = self.attn_norm(x + attn)

        out = self.ffpn_dense_1(attn)
        out = self.ffpn_dense_2(out)
        out = self.ffpn_dropout(out)
        out = self.ffpn_dropout(out)
        out = self.ffpn_norm(attn + out)

        return out


class CNNFeatureTransformer(nn.Module):
    def __init__(self, feature_model, feature_model_output_shape,
                 attn_dim_list, num_head_list, num_class,
                 dropout_proba=0.):
        super().__init__()
        feature_model_dim = feature_model_output_shape[0]
        inner_dim = attn_dim_list[0] * num_head_list[0]
        self.feature_model = feature_model
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.positional_encoding = PositionalEncoding(d_model=inner_dim,
                                                      dropout=dropout_proba)

        # encoder_layers = nn.TransformerEncoderLayer(d_model=feature_model_dim,
        #                                             nhead=num_head_list[0], dim_feedforward=attn_dim_list[0],
        #                                             dropout=0.1)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 6)
        transformer_layer_list = []
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                            dropout=dropout_proba)
            transformer_layer_list.append(attn_layer)
            inner_dim = attn_dim * num_head
        self.transformer_encoder = nn.Sequential(*transformer_layer_list)

        self.final_linear_sequence = nn.Sequential(
            nn.Linear(inner_dim, 512),  # 512?
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(512, 256),
            nn.ReLU6(),
            nn.Linear(256, num_class),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        # input_tensor.shape: [B, C, Z, H, W]
        # feature_tensor.shape: [B*Z, C, H, W]
        feature_tensor = rearrange(input_tensor, 'b c z h w-> (b z) c h w')
        # feature_tensor.shape: [B*Z, C, H, W]
        # use inceptionv4 last output tensor
        feature_tensor = self.feature_model(feature_tensor)[-1]
        # feature_tensor.shape: [B*Z, C, H, W]
        feature_tensor = rearrange(
            input_tensor, '(b z) c h w-> b z h w c', b=batch_size)
        # feature_tensor.shape: [B, (H*W*Z), C]
        feature_tensor = torch.flatten(feature_tensor,
                                       start_dim=1, end_dim=3)
        feature_tensor = self.positional_encoding(feature_tensor)
        # transfomer_tensor.shape: [B, H*W*Z, C]
        transfomer_tensor = self.transformer_encoder(feature_tensor)
        transfomer_tensor = transfomer_tensor.mean(1)
        # transfomer_tensor.shape: [B, num_class]
        output_tensor = self.final_linear_sequence(transfomer_tensor)

        return output_tensor

# feature_model: pytorch_model, feature_model_output_shape: tuple,
# attn_dim_list: int_list, num_head_list:int_list, num_class: int,
# dropout_proba: float=0.


class CNNFeatureTransformer2D(CNNFeatureTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_tensor):
        # input_tensor.shape: [B, 3*Z, H, W]
        # feature_tensor.shape: [B, C, H, W]
        # use inceptionv4 last output tensor
        feature_tensor = self.feature_model(input_tensor)[-1]
        feature_tensor = self.pixel_shuffle(feature_tensor)
        # feature_tensor.shape: [B, H*W, C]
        feature_tensor = rearrange(feature_tensor, 'b c h w-> b (h w) c')
        feature_tensor = self.positional_encoding(feature_tensor)
        # transfomer_tensor.shape: [B, H * W, C]
        transfomer_tensor = self.transformer_encoder(feature_tensor)
        transfomer_tensor = transfomer_tensor.mean(1)
        # output_tensor.shape: [B, num_class]
        output_tensor = self.final_linear_sequence(transfomer_tensor)

        return output_tensor


# feature_model: pytorch_model, feature_model_output_shape: tuple,
# attn_dim_list: int_list, num_head_list:int_list, num_class: int,
# dropout_proba: float=0.
class CNNFeatureTransformer3D(CNNFeatureTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_tensor):
        # input_tensor.shape: [B, 3, Z, H, W]
        # feature_tensor.shape: [B, feature_dim, Z, H, W]
        # use resnet3d last output tensor
        feature_tensor = self.feature_model(input_tensor)
        # feature_tensor.shape: [B, Z*H*W, C]
        feature_tensor = rearrange(feature_tensor, 'b c z h w-> b (z h w) c')
        feature_tensor = feature_tensor.reshape(
            feature_tensor.size(0), 2048, 512)
        feature_tensor = self.positional_encoding(feature_tensor)
        # transfomer_tensor.shape: [B, H * W, C]
        transfomer_tensor = self.transformer_encoder(feature_tensor)
        transfomer_tensor = transfomer_tensor.mean(1)
        # output_tensor.shape: [B, num_class]
        output_tensor = self.final_linear_sequence(transfomer_tensor)

        return output_tensor
