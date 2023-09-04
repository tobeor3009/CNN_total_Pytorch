import math
from functools import partial
import torch
from torch import nn
from .cbam import CBAM

INPLACE = False
DEFAULT_ACT = "mish"


def get_act(activation):
    if isinstance(activation, nn.Module):
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
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=16, stride=16, bias=False)
        highway_channel = max(in_channels, 32)
        self.conv_avg = nn.AdaptiveAvgPool1d(highway_channel)
        self.transform = nn.Linear(highway_channel, in_channels)
        self.transform.bias.data.fill_(init_bias)

    def forward(self, x, y):
        if self.mode == "2d":
            x_proj = self.conv(x)
            x_proj = x_proj.view(x.size(0), -1)
            x_proj = self.conv_avg(x_proj)
        elif self.mode == "3d":
            # You may need to adjust the Conv2d layer to Conv3d for 3D inputs
            raise NotImplementedError("3D mode is not implemented yet")

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
    def __init__(self, in_channels, out_channels,
                 activation=DEFAULT_ACT, kernel_size=2, use_highway=True):
        super().__init__()
        self.use_highway = use_highway
        pixel_shuffle_layer = nn.PixelShuffle(upscale_factor=kernel_size)
        conv_after_pixel_shuffle = nn.Conv2d(in_channels=in_channels // 4,
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


class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=DEFAULT_ACT, use_highway=True):
        super().__init__()
        self.use_highway = use_highway
        conv_before_transpose = nn.Conv3d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=1)
        conv_transpose_layer = nn.ConvTranspose3d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=4, stride=2, padding=1)
        conv_after_transpose = nn.Conv3d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1)
        self.conv_transpose = nn.Sequential(
            conv_before_transpose,
            conv_transpose_layer,
            conv_after_transpose
        )
        conv_before_upsample = nn.Conv3d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1)
        upsample_layer = nn.Upsample(scale_factor=2)
        conv_after_upsample = nn.Conv3d(in_channels=in_channels,
                                        out_channels=out_channels,
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
        self.act = get_act(activation)

    def forward(self, x):
        conv_transpose = self.conv_transpose(x)
        upsample = self.upsample(x)
        if self.use_highway:
            out = self.highway(conv_transpose, upsample)
        else:
            out = (conv_transpose + upsample) / math.sqrt(2)
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
