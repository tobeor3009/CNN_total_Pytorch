import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv3D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'lse']):
        super(ChannelGate3D, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(
                    4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool.view(avg_pool.size(0), -1))
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(
                    4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool.view(max_pool.size(0), -1))
            elif pool_type == 'lse':
                lse_pool = logsumexp_3d(x)
                channel_att_raw = self.mlp(lse_pool.view(lse_pool.size(0), -1))

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


def logsumexp_3d(tensor):
    # Flatten the tensor while keeping the channel dimension separate
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)

    # Compute the maximum value in each channel
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)

    # Compute the log-sum-exp
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()

    return outputs


class ChannelPool3D(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, avg_pool), dim=1)


class SpatialGate3D(nn.Module):
    def __init__(self):
        super(SpatialGate3D, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3D(2, 1, kernel_size, stride=1, padding=(
            kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM3D, self).__init__()
        self.ChannelGate = ChannelGate3D(gate_channels, reduction_ratio,
                                         pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate3D()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
