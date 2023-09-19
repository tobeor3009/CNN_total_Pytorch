import math
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .layers import DEFAULT_ACT, HighwayLayer, get_act, PixelShuffle3D


class MultiDecoder2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=DEFAULT_ACT, kernel_size=2, use_highway=True):
        super().__init__()
        self.use_highway = use_highway
        pixel_shuffle_layer = nn.PixelShuffle(upscale_factor=kernel_size)
        conv_after_pixel_shuffle = nn.Conv2d(in_channels=in_channels // (kernel_size ** 2),
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            pixel_shuffle_layer,
            conv_after_pixel_shuffle
        )
        upsample_layer = nn.Upsample(scale_factor=kernel_size)
        conv_after_upsample = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1)
        self.upsample = nn.Sequential(
            upsample_layer,
            conv_after_upsample
        )
        if self.use_highway:
            self.highway = HighwayLayer(in_channels=out_channels,
                                        mode="2d")
        self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=False)
        self.act = get_act(activation)

    def forward(self, x):
        pixel_shuffle = self.pixel_shuffle(x)
        upsample = self.upsample(x)
        if self.use_highway:
            out = self.highway(pixel_shuffle, upsample)
        else:
            out = (pixel_shuffle + upsample) / math.sqrt(2)
        out = self.norm(out)
        out = self.act(out)
        return out


class MultiDecoder3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=DEFAULT_ACT, kernel_size=2, use_highway=True):
        super().__init__()
        self.use_highway = use_highway
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        pixel_shuffle_layer = PixelShuffle3D(upscale_factor=kernel_size)
        conv_after_pixel_shuffle = nn.Conv3d(in_channels=in_channels // np.prod(kernel_size),
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            pixel_shuffle_layer,
            conv_after_pixel_shuffle
        )
        upsample_layer = nn.Upsample(scale_factor=kernel_size)
        conv_after_upsample = nn.Conv3d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1)
        self.upsample = nn.Sequential(
            upsample_layer,
            conv_after_upsample
        )
        if self.use_highway:
            self.highway = HighwayLayer(in_channels=out_channels,
                                        mode="3d")
        self.norm = nn.InstanceNorm3d(num_features=out_channels, affine=False)
        self.act = get_act(activation)

    def forward(self, x):
        pixel_shuffle = self.pixel_shuffle(x)
        upsample = self.upsample(x)
        if self.use_highway:
            out = self.highway(pixel_shuffle, upsample)
        else:
            out = (pixel_shuffle + upsample) / math.sqrt(2)
        out = self.norm(out)
        out = self.act(out)
        return out


class HighwayOutput2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_highway=True, activation="tanh", init_bias=-3.0):
        super().__init__()
        self.use_highway = use_highway
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1)
        self.conv_3x3 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3, padding="same")
        if self.use_highway:
            self.highway = HighwayLayer(in_channels=out_channels,
                                        mode="2d", init_bias=init_bias)
        self.act = get_act(activation)

    def forward(self, x):
        conv_1x1 = self.conv_1x1(x)
        conv_3x3 = self.conv_3x3(x)
        if self.use_highway:
            output = self.highway(conv_1x1, conv_3x3)
        else:
            output = (conv_1x1 + conv_3x3) / math.sqrt(2)
        output = self.act(output)
        return output


class HighwayOutput3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_highway=True, activation="tanh", init_bias=-3.0):
        super().__init__()
        self.use_highway = use_highway
        self.conv_1x1 = nn.Conv3d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1)
        self.conv_3x3 = nn.Conv3d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1)
        if self.use_highway:
            self.highway = HighwayLayer(in_channels=out_channels,
                                        mode="3d", init_bias=init_bias)
        self.act = get_act(activation)

    def forward(self, x):
        conv_1x1 = self.conv_1x1(x)
        conv_3x3 = self.conv_3x3(x)
        if self.use_highway:
            output = self.highway(conv_1x1, conv_3x3)
        else:
            output = (conv_1x1 + conv_3x3) / math.sqrt(2)
        output = self.act(output)
        return output
