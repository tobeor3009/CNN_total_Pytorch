import math
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

INPLACE = False
DEFAULT_ACT = "relu6"


def get_act(activation):
    if isinstance(activation, nn.Module) or callable(activation):
        act = activation
    elif activation == 'relu6':
        act = nn.ReLU6(inplace=INPLACE)
    elif activation == 'relu':
        act = nn.ReLU(inplace=INPLACE)
    elif activation == "leakyrelu":
        act = nn.LeakyReLU(0.1)
    elif activation == "gelu":
        act = nn.GELU()
    elif activation == "mish":
        act = nn.Mish()
    elif activation == "sigmoid":
        act = torch.sigmoid
    elif activation == "tanh":
        act = torch.tanh
    elif activation == "softmax":
        act = partial(torch.softmax, dim=1)
    elif activation is None:
        act = nn.Identity()
    return act


def get_norm(norm, shape, mode="2d"):
    if isinstance(shape, int):
        shape = [shape]
    if isinstance(norm, nn.Module) or callable(norm):
        norm_layer = norm
    if norm == 'layer':
        norm_layer = nn.LayerNorm(normalized_shape=shape,
                                  elementwise_affine=False)
    elif norm == 'instance':
        if mode == "2d":
            norm_layer = nn.InstanceNorm2d(num_features=shape[0])
        elif mode == "3d":
            norm_layer = nn.InstanceNorm3d(num_features=shape[0])

    elif norm == 'batch':
        if mode == "2d":
            norm_layer = nn.BatchNorm2d(num_features=shape[0],
                                        affine=False)
        elif mode == "3d":
            norm_layer = nn.BatchNorm3d(num_features=shape[0],
                                        affine=False)
    elif norm is None:
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


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm="batch", act=DEFAULT_ACT, bias=False, name=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        if not bias:
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
                 norm="batch", act=DEFAULT_ACT, bias=False, name=None):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        if not bias:
            self.norm_layer = get_norm(norm, out_channels, mode="3d")
        else:
            self.norm_layer = nn.Identity()
        self.act_layer = get_act(act)

    def forward(self, x):
        conv = self.conv(x)
        norm = self.norm_layer(conv)
        act = self.act_layer(norm)
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

        input_view = input.contiguous().view(batch_size, nOut, self.scale[0], self.scale[1],
                                             self.scale[2], in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


# Code from: https://github.com/openai/CLIP/blob/main/clip/model.py
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1,
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
