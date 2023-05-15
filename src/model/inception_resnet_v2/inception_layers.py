from torch import nn
from .cbam import CBAM
from .layers import ConvBlock2D, ConvBlock3D, LambdaLayer, ConcatBlock

INPLACE = False


class Inception_Resnet_Block2D(nn.Module):
    def __init__(self, in_channels, scale, block_type, block_size=16,
                 activation='relu6', include_cbam=True,
                 include_context=False, context_head_nums=8):
        super().__init__()
        self.include_cbam = include_cbam
        if block_type == 'block35':
            branch_0 = ConvBlock2D(in_channels, block_size * 2, 1)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 2, 1),
                ConvBlock2D(block_size * 2, block_size * 2, 3)
            )
            branch_2 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 2, 1),
                ConvBlock2D(block_size * 2, block_size * 3, 3),
                ConvBlock2D(block_size * 3, block_size * 4, 3)
            )
            mixed_channel = block_size * 8
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = ConvBlock2D(in_channels, block_size * 12, 1)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 8, 1),
                ConvBlock2D(block_size * 8, block_size * 10, [1, 7]),
                ConvBlock2D(block_size * 10, block_size * 12, [7, 1])
            )
            mixed_channel = block_size * 24
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = ConvBlock2D(in_channels, block_size * 12, 1)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 12, 1),
                ConvBlock2D(block_size * 12, block_size * 14, [1, 3]),
                ConvBlock2D(block_size * 14, block_size * 16, [3, 1])
            )
            mixed_channel = block_size * 28
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))
        self.mixed = ConcatBlock(branches)
        # TBD: Name?
        self.up = ConvBlock2D(mixed_channel, in_channels, 1,
                              activation=None, bias=True)
        if self.include_cbam:
            self.cbam = CBAM(gate_channels=in_channels,
                             reduction_ratio=16)
        # TBD: implement of include_context
        self.residual_add = LambdaLayer(
            lambda inputs: inputs[0] + inputs[1] * scale)
        if activation == 'relu6':
            self.act = nn.ReLU6(inplace=INPLACE)
        elif activation is None:
            self.act = nn.Identity()

    def forward(self, x):
        mixed = self.mixed(x)
        up = self.up(mixed)
        if self.include_cbam:
            up = self.cbam(up)
        residual_add = self.residual_add([x, up])
        act = self.act(residual_add)
        return act


class Inception_Resnet_Block3D(nn.Module):
    def __init__(self, in_channels, scale, block_type, block_size=16,
                 activation='relu6', include_context=False, context_head_nums=8):
        super().__init__()
        if block_type == 'block35':
            branch_0 = ConvBlock3D(in_channels, block_size * 2, 1)
            branch_1 = nn.Sequential(
                ConvBlock3D(in_channels, block_size * 2, 1),
                ConvBlock3D(block_size * 2, block_size * 2, 3)
            )
            branch_2 = nn.Sequential(
                ConvBlock3D(in_channels, block_size * 2, 1),
                ConvBlock3D(block_size * 2, block_size * 3, 3),
                ConvBlock3D(block_size * 3, block_size * 4, 3)
            )
            mixed_channel = block_size * 8
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = ConvBlock3D(in_channels, block_size * 12, 1)
            branch_1 = nn.Sequential(
                ConvBlock3D(in_channels, block_size * 8, 1),
                ConvBlock3D(block_size * 8, block_size * 10, [1, 1, 7]),
                ConvBlock3D(block_size * 10, block_size * 11, [1, 7, 1]),
                ConvBlock3D(block_size * 11, block_size * 12, [7, 1, 1])
            )
            mixed_channel = block_size * 24
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = ConvBlock3D(in_channels, block_size * 12, 1)
            branch_1 = nn.Sequential(
                ConvBlock3D(in_channels, block_size * 12, 1),
                ConvBlock3D(block_size * 12, block_size * 13, [1, 1, 3]),
                ConvBlock3D(block_size * 13, block_size * 14, [1, 3, 1]),
                ConvBlock3D(block_size * 14, block_size * 16, [3, 1, 1])
            )
            mixed_channel = block_size * 28
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))
        self.mixed = ConcatBlock(branches)
        # TBD: Name?
        self.up = ConvBlock3D(mixed_channel, in_channels, 1,
                              activation=None, bias=True)
        # TBD: implement of include_context
        self.residual_add = LambdaLayer(
            lambda inputs: inputs[0] + inputs[1] * scale)

        if activation == 'relu6':
            self.act = nn.ReLU6(inplace=INPLACE)
        elif activation is None:
            self.act = nn.Identity()

    def forward(self, x):
        mixed = self.mixed(x)
        up = self.up(mixed)
        residual_add = self.residual_add([x, up])
        act = self.act(residual_add)

        return act