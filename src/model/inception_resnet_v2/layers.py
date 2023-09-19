import math
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

INPLACE = False
DEFAULT_ACT = "gelu"


def get_act(activation):
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


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2,
                           h // block_size, w // block_size)


def space_to_depth_3d(input_tensor, block_size):
    B, C, D, H, W = input_tensor.shape

    # Reshape tensor
    output_tensor = input_tensor.view(B, C,
                                      D // block_size, block_size,
                                      H // block_size, block_size,
                                      W // block_size, block_size)

    # Permute tensor dimensions
    output_tensor = output_tensor.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

    # Merge block_size^3 blocks into the channels
    output_tensor = output_tensor.view(B, C * block_size * block_size * block_size,
                                       D // block_size, H // block_size, W // block_size)
    return output_tensor


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


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 activation=DEFAULT_ACT, bias=False, name=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        if not bias:
            # in keras, scale=False
            self.norm = nn.BatchNorm2d(num_features=out_channels, affine=False)
        else:
            self.norm = nn.Identity()
        self.act = get_act(activation)

    def forward(self, x):
        conv = self.conv(x)
        norm = self.norm(conv)
        act = self.act(norm)
        return act


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 activation=DEFAULT_ACT, bias=False, name=None):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        if not bias:
            # in keras, scale=False
            self.norm = nn.BatchNorm3d(num_features=out_channels, affine=False)
        else:
            self.norm = nn.Identity()

        self.act = get_act(activation)

    def forward(self, x):
        conv = self.conv(x)
        norm = self.norm(conv)
        act = self.act(norm)
        return act


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# Assume Channel First
class ConcatBlock(nn.Module):
    def __init__(self, layer_list, dim=1):
        super().__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.dim = dim

    def forward(self, x):
        tensor_list = [layer(x) for layer in self.layer_list]
        return torch.cat(tensor_list, self.dim)


class SkipUpSample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_2d = ConvBlock2D(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1)
        # self.cbam = CBAM(gate_channels=out_channels,
        #                  reduction_ratio=16)
        self.conv_3d = ConvBlock3D(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=3)

    def forward(self, input_tensor):
        # input_2d.shape: [H C H W]
        # input_2d.shape: [H C 1 H W]
        _, _, H, _ = input_tensor.size()
        input_2d = self.conv_2d(input_tensor)
        # input_2d = self.cbam(input_2d)
        input_3d = input_2d.unsqueeze(2)
        input_3d = input_3d.expand(-1, -1, H, -1, -1)
        input_3d = self.conv_3d(input_3d)
        return input_3d


class HighwayLayer(nn.Module):
    def __init__(self, in_channels, mode="2d", init_bias=-3.0):
        super().__init__()
        self.mode = mode
        if self.mode == "2d":
            self.conv = nn.Conv2d(in_channels, in_channels,
                                  kernel_size=16, stride=16, bias=False)
        elif self.mode == "3d":
            self.conv = nn.Conv3d(in_channels, in_channels,
                                  kernel_size=16, stride=16, bias=False)
        highway_channel = max(in_channels, 32)
        self.conv_avg = nn.AdaptiveAvgPool1d(highway_channel)
        self.transform = nn.Linear(highway_channel, in_channels)
        self.transform.bias.data.fill_(init_bias)

    def forward(self, x, y):
        x_proj = self.conv(x)
        x_proj = x_proj.view(x.size(0), -1)
        x_proj = self.conv_avg(x_proj)

        x_proj = self.transform(x_proj)
        x_proj = torch.sigmoid(x_proj)
        x_proj_shape = x_proj.size()

        if self.mode == "2d":
            x_proj = x_proj.view(*x_proj_shape, 1, 1)
        elif self.mode == "3d":
            x_proj = x_proj.view(*x_proj_shape, 1, 1, 1)
        gated = (x * x_proj) + ((1 - x_proj) * y)
        return gated


class Decoder2D(nn.Module):
    def __init__(self, input_hw, in_channels, out_channels,
                 activation=DEFAULT_ACT, kernel_size=2, use_pixelshuffle=True):
        super().__init__()
        h, w = input_hw
        upsample_shape = (out_channels, 2 * h, 2 * w)
        if use_pixelshuffle:
            conv_before_pixel_shuffle = nn.Conv2d(in_channels=in_channels * (kernel_size ** 2),
                                                  out_channels=out_channels,
                                                  kernel_size=1)
            pixel_shuffle_layer = nn.PixelShuffle(upscale_factor=kernel_size)
            conv_after_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1)
            self.upsample_block = nn.Sequential(
                conv_before_pixel_shuffle,
                pixel_shuffle_layer,
                conv_after_pixel_shuffle
            )
        else:
            upsample_layer = nn.Upsample(scale_factor=kernel_size)
            conv_after_upsample = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)
            self.upsample_block = nn.Sequential(
                upsample_layer,
                conv_after_upsample
            )
        self.norm = nn.LayerNorm(normalized_shape=upsample_shape,
                                 elementwise_affine=False)
        self.act = get_act(activation)

    def forward(self, x):
        out = self.upsample_block(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class Decoder3D(nn.Module):
    def __init__(self, input_zhw, in_channels, out_channels,
                 activation=DEFAULT_ACT, kernel_size=2,
                 use_pixelshuffle=True):
        super().__init__()
        z, h, w = input_zhw
        upsample_shape = (out_channels, 2 * z, 2 * h, 2 * w)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        if use_pixelshuffle:
            conv_before_pixel_shuffle = nn.Conv3d(in_channels=in_channels * (kernel_size ** 3),
                                                  out_channels=out_channels,
                                                  kernel_size=1)
            pixel_shuffle_layer = PixelShuffle3D(upscale_factor=kernel_size)
            conv_after_pixel_shuffle = nn.Conv3d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1)
            self.upsample_block = nn.Sequential(
                conv_before_pixel_shuffle,
                pixel_shuffle_layer,
                conv_after_pixel_shuffle
            )
        else:
            upsample_layer = nn.Upsample(scale_factor=kernel_size)
            conv_after_upsample = nn.Conv3d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)
            self.upsample_block = nn.Sequential(
                upsample_layer,
                conv_after_upsample
            )
        self.norm = nn.LayerNorm(normalized_shape=upsample_shape,
                                 elementwise_affine=False)
        self.act = get_act(activation)

    def forward(self, x):
        out = self.upsample_block(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class Output2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation="tanh"):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1, padding=0)
        self.act = get_act(activation)

    def forward(self, x):
        conv_1x1 = self.conv_1x1(x)
        output = self.act(conv_1x1)
        return output


class Output3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation="tanh"):
        super().__init__()
        self.conv_1x1x1 = nn.Conv3d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1, padding=0)
        self.act = get_act(activation)

    def forward(self, x):
        conv_1x1x1 = self.conv_1x1x1(x)
        output = self.act(conv_1x1x1)
        return output
