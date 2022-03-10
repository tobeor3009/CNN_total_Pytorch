import math

import torch
from torch import nn
# from .base_model_2d import ConvBlock2D
from einops import rearrange

INPLACE = False


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


class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()

        self.conv_before_pixel_shuffle = ConvBlock2D(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=kernel_size)
        self.conv_before_pixel_shuffle = ConvBlock2D(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size)


# class ConvBlock2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding='same',
#                  activation='relu6', bias=False, name=None):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                               kernel_size=kernel_size, stride=stride, padding=padding,
#                               bias=bias)
#         if not bias:
#             # in keras, scale=False
#             self.norm = nn.BatchNorm2d(num_features=out_channels, affine=False)
#         else:
#             self.norm = nn.Identity()
#         if activation == 'relu6':
#             self.act = nn.ReLU6(inplace=INPLACE)
#         elif activation is None:
#             self.act = nn.Identity()

#     def forward(self, x):
#         conv = self.conv(x)
#         norm = self.norm(conv)
#         act = self.act(norm)
#         return act
