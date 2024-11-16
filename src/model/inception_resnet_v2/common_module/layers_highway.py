import math
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .layers import DEFAULT_ACT, INPLACE, HighwayLayer, PixelShuffle3D
from .layers import ConvBlock2D, ConvBlock3D
from .layers import get_act, get_norm

conv_transpose_kwarg_dict = {
    "kernel_size":3,
    "padding":1,
    "output_padding":1,
    "bias":True
}
class MultiDecoder2D(nn.Module):
    def __init__(self, input_hw, in_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, dropout_proba=0.0, kernel_size=2,
                 use_highway=False, use_pixelshuffle_only=False):
        super().__init__()
        h, w = input_hw
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.use_highway = use_highway
        self.use_pixelshuffle_only = use_pixelshuffle_only
        upsample_shape = (out_channels,
                          kernel_size[0] * h,
                          kernel_size[1] * w)
        conv_before_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                              out_channels=in_channels *
                                              np.prod(kernel_size),
                                              kernel_size=1)
        pixel_shuffle_layer = nn.PixelShuffle(upscale_factor=(kernel_size
                                                              if isinstance(kernel_size, int)
                                                              else kernel_size[0]))
        conv_after_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            conv_before_pixel_shuffle,
            pixel_shuffle_layer,
            conv_after_pixel_shuffle
        )
        if not self.use_pixelshuffle_only:
            upsample_layer = nn.Upsample(scale_factor=kernel_size,
                                         mode='bilinear')
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
            else:
                self.concat_conv = nn.Conv2d(in_channels=out_channels * 2,
                                             out_channels=out_channels,
                                             kernel_size=3, padding=1)
        self.norm = get_norm(norm, upsample_shape, mode="2d")
        self.act = get_act(act)
        self.dropout = nn.Dropout2d(p=dropout_proba, inplace=INPLACE)
    def forward(self, x):
        pixel_shuffle = self.pixel_shuffle(x)
        if not self.use_pixelshuffle_only:
            upsample = self.upsample(x)
            if self.use_highway:
                out = self.highway(pixel_shuffle, upsample)
            else:
                out = torch.cat([pixel_shuffle, upsample], dim=1)
                out = self.concat_conv(out)
        else:
            out = pixel_shuffle
        out = self.dropout(out)
        out = self.norm(out)
        out = self.act(out)
        return out

class MultiDecoder2D_V2(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, dropout_proba=0.0, kernel_size=2,
                 use_highway=False, use_pixelshuffle_only=False, include_conv_transpose=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.use_highway = use_highway
        self.use_pixelshuffle_only = use_pixelshuffle_only
        self.include_conv_transpose = include_conv_transpose
        conv_before_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                              out_channels=in_channels *
                                              np.prod(kernel_size),
                                              kernel_size=1)
        pixel_shuffle_layer = nn.PixelShuffle(upscale_factor=(kernel_size
                                                              if isinstance(kernel_size, int)
                                                              else kernel_size[0]))
        conv_after_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            conv_before_pixel_shuffle,
            pixel_shuffle_layer,
            conv_after_pixel_shuffle
        )
        middle_channels = out_channels
        if not self.use_pixelshuffle_only:
            upsample_layer = nn.Upsample(scale_factor=kernel_size,
                                         mode='bilinear')
            conv_after_upsample = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)
            self.upsample = nn.Sequential(
                upsample_layer,
                conv_after_upsample
            )
            middle_channels += out_channels
        if self.include_conv_transpose:
            self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                                     stride=kernel_size,
                                                    **conv_transpose_kwarg_dict)
            middle_channels += out_channels
            
        self.concat_conv = ConvBlock2D(middle_channels, out_channels,
                                kernel_size=3, stride=1, padding=1,
                                norm=norm, act=act, bias=True, dropout_proba=dropout_proba)
        self.skip_conv = ConvBlock2D(out_channels + skip_channels, out_channels,
                                kernel_size=3, stride=1, padding=1,
                                norm=norm, act=act, bias=True, dropout_proba=dropout_proba)
    def forward(self, x, skip):
        pixel_shuffle = self.pixel_shuffle(x)
        if not self.use_pixelshuffle_only:
            upsample = self.upsample(x)
            out = torch.cat([pixel_shuffle, upsample], dim=1)
        else:
            out = pixel_shuffle
        if self.include_conv_transpose:
            conv_transpose = self.conv_transpose(x)
            out = torch.cat([out, conv_transpose], dim=1)
        out = self.concat_conv(out)
        out = torch.cat([out, skip], dim=1)
        out = self.skip_conv(out)
        return out

class ConvTransposeDecoder2D_V2(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, norm="layer",
                 act=DEFAULT_ACT, dropout_proba=0.0, kernel_size=2):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                                 stride=kernel_size,
                                                **conv_transpose_kwarg_dict)
        self.conv = ConvBlock2D(skip_channels + out_channels, out_channels,
                                kernel_size=3, stride=1, padding=1,
                                norm=norm, act=act, bias=True, dropout_proba=dropout_proba)
    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class MultiDecoder3D(nn.Module):
    def __init__(self, input_zhw, in_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, dropout_proba=0.0, kernel_size=2,
                 use_highway=False, use_pixelshuffle_only=False):
        super().__init__()
        self.use_highway = use_highway
        self.use_pixelshuffle_only = use_pixelshuffle_only
        z, h, w = input_zhw
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        upsample_shape = (out_channels,
                          kernel_size[0] * z,
                          kernel_size[1] * h,
                          kernel_size[2] * w)
        conv_before_pixel_shuffle = nn.Conv3d(in_channels=in_channels,
                                              out_channels=(in_channels *
                                                            np.prod(kernel_size)) // 4,
                                              kernel_size=1)
        pixel_shuffle_layer = PixelShuffle3D(upscale_factor=kernel_size)
        conv_after_pixel_shuffle = nn.Conv3d(in_channels=in_channels // 4,
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            conv_before_pixel_shuffle,
            pixel_shuffle_layer,
            conv_after_pixel_shuffle
        )
        if not self.use_pixelshuffle_only:
            upsample_layer = nn.Upsample(scale_factor=kernel_size,
                                         mode='trilinear')
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
            else:
                self.concat_conv = nn.Conv3d(in_channels=out_channels * 2,
                                             out_channels=out_channels,
                                             kernel_size=1, padding=0)
        self.norm = get_norm(norm, upsample_shape, mode="3d")
        self.act = get_act(act)
        self.dropout = nn.Dropout3d(p=dropout_proba, inplace=INPLACE)

    def forward(self, x):
        pixel_shuffle = self.pixel_shuffle(x)
        if not self.use_pixelshuffle_only:
            upsample = self.upsample(x)
            if self.use_highway:
                out = self.highway(pixel_shuffle, upsample)
            else:
                out = torch.cat([pixel_shuffle, upsample], dim=1)
                out = self.concat_conv(out)
        else:
            out = pixel_shuffle
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out

class MultiDecoder3D_V2(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, dropout_proba=0.0, kernel_size=2,
                 use_highway=False, use_pixelshuffle_only=False, include_conv_transpose=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.use_highway = use_highway
        self.use_pixelshuffle_only = use_pixelshuffle_only
        self.include_conv_transpose = include_conv_transpose
        conv_before_pixel_shuffle = nn.Conv3d(in_channels=in_channels,
                                              out_channels=(in_channels *
                                                            np.prod(kernel_size)) // 4,
                                              kernel_size=1)
        pixel_shuffle_layer = PixelShuffle3D(upscale_factor=kernel_size)
        conv_after_pixel_shuffle = nn.Conv3d(in_channels=in_channels // 4,
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            conv_before_pixel_shuffle,
            pixel_shuffle_layer,
            conv_after_pixel_shuffle
        )
        middle_channels = out_channels
        if not self.use_pixelshuffle_only:
            upsample_layer = nn.Upsample(scale_factor=kernel_size,
                                         mode='trilinear')
            conv_after_upsample = nn.Conv3d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)
            self.upsample = nn.Sequential(
                upsample_layer,
                conv_after_upsample
            )
            middle_channels += out_channels
        if self.include_conv_transpose:
            self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                    stride=kernel_size,
                                                    **conv_transpose_kwarg_dict)
            middle_channels += out_channels

        self.concat_conv = ConvBlock3D(middle_channels, out_channels,
                                kernel_size=3, stride=1, padding=1,
                                norm=norm, act=act, bias=True, dropout_proba=dropout_proba)
        self.skip_conv = ConvBlock3D(out_channels + skip_channels, out_channels,
                                kernel_size=3, stride=1, padding=1,
                                norm=norm, act=act, bias=True, dropout_proba=dropout_proba)
    def forward(self, x, skip):
        pixel_shuffle = self.pixel_shuffle(x)
        if not self.use_pixelshuffle_only:
            upsample = self.upsample(x)
            out = torch.cat([pixel_shuffle, upsample], dim=1)
        else:
            out = pixel_shuffle
        if self.include_conv_transpose:
            conv_transpose = self.conv_transpose(x)
            out = torch.cat([out, conv_transpose], dim=1)
        out = self.concat_conv(out)
        out = torch.cat([out, skip], dim=1)
        out = self.skip_conv(out)
        return out
class ConvTransposeDecoder3D_V2(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, dropout_proba=0.0, kernel_size=2):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                 stride=kernel_size,
                                                **conv_transpose_kwarg_dict)
        self.conv = ConvBlock3D(skip_channels + out_channels, out_channels,
                                kernel_size=3, stride=1, padding=1,
                                norm=norm, act=act, bias=True, dropout_proba=dropout_proba)
    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class HighwayOutput2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_highway=False, act="tanh", init_bias=-3.0):
        super().__init__()
        self.use_highway = use_highway
        conv_out_channels = out_channels if self.use_highway else in_channels // 2
        self.conv_5x5 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=conv_out_channels,
                                  kernel_size=5, padding=2)
        self.conv_3x3 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=conv_out_channels,
                                  kernel_size=3, padding=1)
        if self.use_highway:
            self.highway = HighwayLayer(in_channels=out_channels,
                                        mode="2d", init_bias=init_bias)
        else:
            self.concat_conv = nn.Conv2d(in_channels=conv_out_channels * 2,
                                         out_channels=out_channels,
                                         kernel_size=3, padding=1)
        self.act = get_act(act)

    def forward(self, x):
        conv_5x5 = self.conv_5x5(x)
        conv_3x3 = self.conv_3x3(x)
        if self.use_highway:
            output = self.highway(conv_5x5, conv_3x3)
        else:
            output = torch.cat([conv_5x5, conv_3x3], dim=1)
            output = self.concat_conv(output)
        output = self.act(output)
        return output


class HighwayOutput3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_highway=True, act="tanh", init_bias=-3.0):
        super().__init__()
        self.use_highway = use_highway
        conv_out_channels = out_channels if self.use_highway else in_channels // 2
        self.conv_5x5x5 = nn.Conv3d(in_channels=in_channels,
                                    out_channels=conv_out_channels,
                                    kernel_size=5, padding=2)
        self.conv_3x3x3 = nn.Conv3d(in_channels=in_channels,
                                    out_channels=conv_out_channels,
                                    kernel_size=3, padding=1)
        if self.use_highway:
            self.highway = HighwayLayer(in_channels=out_channels,
                                        mode="3d", init_bias=init_bias)
        else:
            self.concat_conv = nn.Conv3d(in_channels=conv_out_channels * 2,
                                         out_channels=out_channels,
                                         kernel_size=3, padding=1)
        self.act = get_act(act)

    def forward(self, x):
        conv_5x5x5 = self.conv_5x5x5(x)
        conv_3x3x3 = self.conv_3x3x3(x)
        if self.use_highway:
            output = self.highway(conv_5x5x5, conv_3x3x3)
        else:
            output = torch.cat([conv_5x5x5, conv_3x3x3], dim=1)
            output = self.concat_conv(output)
        output = self.act(output)
        return output
