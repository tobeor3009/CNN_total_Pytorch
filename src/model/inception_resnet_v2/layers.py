import math

import torch
from torch import nn
from .cbam import CBAM

INPLACE = False


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 activation='relu6', bias=False, name=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        if not bias:
            # in keras, scale=False
            self.norm = nn.BatchNorm2d(num_features=out_channels, affine=False)
        else:
            self.norm = nn.Identity()
        if activation == 'relu6':
            self.act = nn.ReLU6(inplace=INPLACE)
        elif activation is None:
            self.act = nn.Identity()

    def forward(self, x):
        conv = self.conv(x)
        norm = self.norm(conv)
        act = self.act(norm)
        return act


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 activation='relu6', bias=False, name=None):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        if not bias:
            # in keras, scale=False
            self.norm = nn.BatchNorm3d(num_features=out_channels, affine=False)
        else:
            self.norm = nn.Identity()

        if activation == 'relu6':
            self.act = nn.ReLU6(inplace=INPLACE)
        elif activation is None:
            self.act = nn.Identity()

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
        self.transform = nn.Linear(in_channels, in_channels)
        self.transform.bias.data.fill_(init_bias)

    def forward(self, x, y):
        # x.shape: [B C H W]
        # x_proj: [B C]
        if self.mode == "2d":
            x_proj = x.mean([2, 3])
        elif self.mode == "3d":
            x_proj = x.mean([2, 3, 4])

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
    def __init__(self, in_channels, out_channels, kernel_size=2, use_highway=True):
        super().__init__()
        self.use_highway = use_highway
        conv_before_pixel_shuffle = ConvBlock2D(in_channels=in_channels,
                                                out_channels=out_channels *
                                                (kernel_size ** 2),
                                                kernel_size=1)
        pixel_shuffle_layer = nn.PixelShuffle(upscale_factor=kernel_size)
        conv_after_pixel_shuffle = ConvBlock2D(in_channels=out_channels, out_channels=out_channels,
                                               kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            conv_before_pixel_shuffle,
            pixel_shuffle_layer,
            conv_after_pixel_shuffle
        )
        conv_before_upsample = ConvBlock2D(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=1)
        upsample_layer = nn.Upsample(scale_factor=kernel_size)
        conv_after_upsample = ConvBlock2D(in_channels=out_channels, out_channels=out_channels,
                                          kernel_size=1)
        self.upsample = nn.Sequential(
            conv_before_upsample,
            upsample_layer,
            conv_after_upsample
        )
        if self.use_highway:
            self.highway = HighwayLayer(in_channels=out_channels,
                                        mode="2d")
        self.norm = nn.BatchNorm2d(num_features=out_channels, affine=False)
        self.act = nn.ReLU6(inplace=INPLACE)

    def forward(self, x):
        pixel_shuffle = self.pixel_shuffle(x)
        upsample = self.upsample(x)
        if self.use_highway:
            out = self.highway(pixel_shuffle, upsample)
        else:
            out = pixel_shuffle + upsample
        out = self.norm(out)
        out = self.act(out)
        return out


class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_highway=True):
        super().__init__()
        self.use_highway = use_highway
        conv_before_transpose = ConvBlock3D(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)
        conv_transpose_layer = nn.ConvTranspose3d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=4, stride=2, padding=1)
        conv_after_transpose = ConvBlock3D(in_channels=out_channels, out_channels=out_channels,
                                           kernel_size=1)
        self.conv_transpose = nn.Sequential(
            conv_before_transpose,
            conv_transpose_layer,
            conv_after_transpose
        )
        conv_before_upsample = ConvBlock3D(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=1)
        upsample_layer = nn.Upsample(scale_factor=2)
        conv_after_upsample = ConvBlock3D(in_channels=out_channels, out_channels=out_channels,
                                          kernel_size=1)
        self.upsample = nn.Sequential(
            conv_before_upsample,
            upsample_layer,
            conv_after_upsample
        )
        if self.use_highway:
            self.highway = HighwayLayer(in_channels=out_channels,
                                        mode="3d")
        self.norm = nn.BatchNorm3d(num_features=out_channels, affine=False)
        self.act = nn.ReLU6(inplace=INPLACE)

    def forward(self, x):
        conv_transpose = self.conv_transpose(x)
        upsample = self.upsample(x)
        if self.use_highway:
            out = self.highway(conv_transpose, upsample)
        else:
            out = conv_transpose + upsample
        out = self.norm(out)
        out = self.act(out)
        return out


class HighwayOutput2D(nn.Module):
    def __init__(self, in_channels, out_channels, act="tanh", init_bias=-3.0):
        super().__init__()
        self.conv_1x1 = ConvBlock2D(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1)
        self.conv_3x3 = ConvBlock2D(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3)
        self.highway = HighwayLayer(in_channels=out_channels,
                                    mode="2d", init_bias=init_bias)
        if act == "tanh":
            self.act = torch.tanh
        elif act == "sigmoid":
            self.act = torch.sigmoid

    def forward(self, x):
        conv_1x1 = self.conv_1x1(x)
        conv_3x3 = self.conv_3x3(x)
        highway_output = self.highway(conv_1x1, conv_3x3)
        highway_output = self.act(highway_output)
        return highway_output


class HighwayOutput3D(nn.Module):
    def __init__(self, in_channels, out_channels, act="tanh", init_bias=-3.0):
        super().__init__()
        self.conv_1x1 = ConvBlock3D(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1)
        self.conv_3x3 = ConvBlock3D(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3)
        self.highway = HighwayLayer(in_channels=out_channels,
                                    mode="3d", init_bias=init_bias)
        if act == "tanh":
            self.act = torch.tanh
        elif act == "sigmoid":
            self.act = torch.sigmoid

    def forward(self, x):
        conv_1x1 = self.conv_1x1(x)
        conv_3x3 = self.conv_3x3(x)
        highway_output = self.highway(conv_1x1, conv_3x3)
        highway_output = self.act(highway_output)
        return highway_output
