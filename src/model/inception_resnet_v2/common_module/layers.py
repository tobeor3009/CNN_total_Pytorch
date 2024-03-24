import math
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .cbam import CBAM as CBAM2D
from .cbam_3d import CBAM3D
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
    elif norm == "group":
        norm_layer = nn.GroupNorm(num_groups=8, num_channels=shape[0])
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

    def forward(self, *args, **kwargs):
        tensor_list = [layer(*args, **kwargs) for layer in self.layer_list]
        return torch.cat(tensor_list, self.dim)


class SkipUpSample3D(nn.Module):
    def __init__(self, in_channels, out_channels, cbam=False):
        super().__init__()
        self.cbam = cbam
        self.conv_2d = ConvBlock2D(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1)
        if cbam:
            self.cbam_2d = CBAM2D(gate_channels=out_channels,
                                  reduction_ratio=16)
            self.cbam_3d = CBAM3D(gate_channels=out_channels,
                                  reduction_ratio=16)

        self.conv_3d = ConvBlock3D(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=3)

    def forward(self, input_tensor):
        # input_2d.shape: [H C H W]
        # input_2d.shape: [H C 1 H W]
        _, _, H, _ = input_tensor.size()
        input_2d = self.conv_2d(input_tensor)
        if self.cbam:
            input_2d = self.cbam_2d(input_2d)
        input_3d = input_2d.unsqueeze(2)
        input_3d = input_3d.expand(-1, -1, H, -1, -1)
        input_3d = self.conv_3d(input_3d)
        if self.cbam:
            input_3d = self.cbam_3d(input_3d)
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
                 act=DEFAULT_ACT, norm="layer", kernel_size=2,
                 use_pixelshuffle=True):
        super().__init__()
        h, w = input_hw
        upsample_shape = (out_channels,
                          kernel_size * h,
                          kernel_size * w)
        if use_pixelshuffle:
            conv_before_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                                  out_channels=in_channels *
                                                  (kernel_size ** 2),
                                                  kernel_size=1)
            pixel_shuffle_layer = nn.PixelShuffle(upscale_factor=kernel_size)
            conv_after_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1, padding=1)
            self.upsample_block = nn.Sequential(
                conv_before_pixel_shuffle,
                pixel_shuffle_layer,
                conv_after_pixel_shuffle
            )
        else:
            upsample_layer = nn.Upsample(scale_factor=kernel_size)
            conv_after_upsample = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1, padding=1)
            self.upsample_block = nn.Sequential(
                upsample_layer,
                conv_after_upsample
            )
        self.norm_layer = get_norm(norm, upsample_shape)
        self.act_layer = get_act(act)

    def forward(self, x):
        out = self.upsample_block(x)
        out = self.norm_layer(out)
        out = self.act_layer(out)
        return out


class Decoder3D(nn.Module):
    def __init__(self, input_zhw, in_channels, out_channels,
                 act=DEFAULT_ACT, kernel_size=2,
                 use_pixelshuffle=True):
        super().__init__()
        z, h, w = input_zhw
        upsample_shape = (out_channels,
                          kernel_size * z,
                          kernel_size * h,
                          kernel_size * w)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        if use_pixelshuffle:
            conv_before_pixel_shuffle = nn.Conv3d(in_channels=in_channels,
                                                  out_channels=in_channels *
                                                  (kernel_size ** 3),
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
        self.act = get_act(act)

    def forward(self, x):
        out = self.upsample_block(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class Output2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 act="tanh"):
        super().__init__()
        self.conv_5x5 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=5, padding=2)
        self.conv_3x3 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3, padding=1)
        self.act = get_act(act)

    def forward(self, x):
        conv_5x5 = self.conv_5x5(x)
        conv_3x3 = self.conv_3x3(x)
        output = (conv_5x5 + conv_3x3) / math.sqrt(2)
        output = self.act(output)
        return output


class Output3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 act="tanh"):
        super().__init__()
        self.conv_1x1x1 = nn.Conv3d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1, padding=0)
        self.act = get_act(act)

    def forward(self, x):
        conv_1x1x1 = self.conv_1x1x1(x)
        output = self.act(conv_1x1x1)
        return output


class AttentionPool(nn.Module):
    def __init__(self, feature_num: tuple, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(feature_num + 1,
                                                             embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(
            2, 0, 1)  # BC(HW) -> NBC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (N+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (N+1)NC
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
