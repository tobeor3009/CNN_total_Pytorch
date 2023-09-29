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
    if activation == 'relu6':
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
