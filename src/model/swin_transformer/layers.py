import math
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils import spectral_norm

INPLACE = False
DEFAULT_ACT = "relu6"


class SwishBeta(nn.Module):
    def __init__(self):
        super(SwishBeta, self).__init__()
        self.beta = nn.Parameter(torch.as_tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def get_act(act):
    if isinstance(act, nn.Module) or callable(act):
        act = act
    elif act == 'relu6':
        act = nn.ReLU6(inplace=INPLACE)
    elif act == 'relu':
        act = nn.ReLU(inplace=INPLACE)
    elif act == "leakyrelu":
        act = nn.LeakyReLU(0.1, inplace=INPLACE)
    elif act == "gelu":
        act = nn.GELU()
    elif act == "mish":
        act = nn.Mish(inplace=INPLACE)
    elif act == "silu":
        act = nn.SiLU(inplace=INPLACE)
    elif act == "silu-beta":
        act = SwishBeta()
    elif act == "sigmoid":
        act = torch.sigmoid
    elif act == "tanh":
        act = torch.tanh
    elif act == "softmax":
        act = partial(torch.softmax, dim=1)
    elif act is None:
        act = nn.Identity()
    else:
        act = nn.Identity()
    return act


def get_norm(norm, shape, mode="2d"):
    if isinstance(shape, int):
        shape = [shape]
    if isinstance(norm, nn.Module) or callable(norm):
        norm_layer = norm
    elif norm == 'layer':
        if len(shape) == 1:
            shape = shape[0]
        norm_layer = nn.LayerNorm(normalized_shape=shape,
                                  elementwise_affine=False)
    elif norm == 'instance':
        if mode == "2d":
            norm_layer = nn.InstanceNorm2d(num_features=shape[0])
        elif mode == "3d":
            norm_layer = nn.InstanceNorm3d(num_features=shape[0])
        elif mode == "1d":
            norm_layer = nn.InstanceNorm1d(num_features=shape[0])
    elif norm == 'batch':
        if mode == "2d":
            norm_layer = nn.BatchNorm2d(num_features=shape[0],
                                        affine=False)
        elif mode == "3d":
            norm_layer = nn.BatchNorm3d(num_features=shape[0],
                                        affine=False)
        elif mode == "1d":
            norm_layer = nn.BatchNorm1d(num_features=shape[0],
                                        affine=False)
    elif norm is None:
        norm_layer = nn.Identity()
    else:
        norm_layer = nn.Identity()
    return norm_layer


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2,
                           h // block_size, w // block_size)


class LinearAct(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, act=None):
        super().__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=bias)
        self.act = get_act(act)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False, channel_last=False):
        super().__init__()
        self.channel_last = channel_last
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups,  bias=bias)
        if norm == "spectral":
            self.conv = spectral_norm(self.conv)
        if not bias or norm != "spectral":
            self.norm_layer = get_norm(norm, out_channels, mode="1d")
        else:
            self.norm_layer = nn.Identity()
        self.act_layer = get_act(act)

    def forward(self, x):
        if self.channel_last:
            x = x.permute(0, 2, 1)
        conv = self.conv(x)
        norm = self.norm_layer(conv)
        act = self.act_layer(norm)
        if self.channel_last:
            act = act.permute(0, 2, 1)
        return act


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False, name=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups,  bias=bias)
        if norm == "spectral":
            self.conv = spectral_norm(self.conv)
        if not bias or norm != "spectral":
            self.norm_layer = get_norm(norm, out_channels, mode="2d")
        else:
            self.norm_layer = nn.Identity()
        self.act_layer = get_act(act)

    def forward(self, x):
        conv = self.conv(x)
        norm = self.norm_layer(conv)
        act = self.act_layer(norm)
        return act


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 act=DEFAULT_ACT, norm="batch", groups=1, bias=False, name=None):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        if norm == "spectral":
            self.conv = spectral_norm(self.conv)
        if not bias or norm != "spectral":
            # in keras, scale=False
            self.norm_layer = get_norm(norm, out_channels, mode="3d")
        else:
            self.norm_layer = nn.Identity()

        self.act = get_act(act)

    def forward(self, x):
        conv = self.conv(x)
        norm = self.norm_layer(conv)
        act = self.act(norm)
        return act


class PixelShuffle3D(nn.Module):
    '''
    Source: https://github.com/kuoweilai/pixelshuffle3d/blob/master/pixelshuffle3d.py
    This class is a 3d version of pixelshuffle.
    '''

    def __init__(self, upscale_factor):
        '''
        :param scale: upsample scale
        '''
        super().__init__()

        if isinstance(upscale_factor, int):
            upscale_factor = (upscale_factor, upscale_factor, upscale_factor)
        self.scale = upscale_factor

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // np.prod(self.scale)

        out_depth = in_depth * self.scale[0]
        out_height = in_height * self.scale[1]
        out_width = in_width * self.scale[2]

        input_view = input.view(batch_size, nOut, self.scale[0], self.scale[1],
                                self.scale[2], in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4)
        output = output.contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class AttentionPool1d(nn.Module):
    def __init__(self, sequence_length: int, embed_dim: int,
                 num_heads: int, output_dim: int = None,
                 channel_first: bool = True):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(sequence_length + 1,
                                                             embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x = x.permute(2, 0, 1)
        else:
            x = x.permute(1, 0, 2)  # NCL -> LNC

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias,
                                    self.k_proj.bias,
                                    self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

# Code from: https://github.com/openai/CLIP/blob/main/clip/model.py


class AttentionPool(nn.Module):
    def __init__(self, feature_num: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(np.prod(feature_num) + 1,
                                                             embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias,
                                    self.k_proj.bias,
                                    self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ZAttentionPooling(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ZAttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.qkv = torch.nn.Linear(input_dim, 3 * hidden_dim)

    def forward(self, x):
        # x: [B, patch_num, N, C]

        B, patch_num, N, C = x.shape

        # Query, Key, Value 동시에 계산
        qkv = self.qkv(x.reshape(-1, C)).reshape(B, patch_num, N,
                                                 3 * self.hidden_dim)
        Q, K, V = torch.split(qkv, self.hidden_dim, dim=-1)

        # Attention scores 계산
        attention_scores = torch.einsum(
            "bpni,bpnj->bpin", Q, K) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted sum 계산
        weighted_sum = torch.einsum("bpin,bpnc->bpic", attention_weights, V)

        # 평균 계산
        pooled_output = weighted_sum.mean(dim=2)

        return pooled_output  # [B, patch_num, hidden_dim]
