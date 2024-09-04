from torch import nn
from .cbam import CBAM
from .layers import ConvBlock2D, ConvBlock3D, LambdaLayer, ConcatBlock
from .layers import DEFAULT_ACT, get_act

INPLACE = False


class Inception_Resnet_Block2D(nn.Module):
    def __init__(self, in_channels, scale, block_type, block_size=16,
                 include_cbam=True, norm="batch", act=DEFAULT_ACT, dropout_proba=0.0):
        super().__init__()
        self.include_cbam = include_cbam
        conv_block_common_arg_dict = {
            "norm": norm,
            "act": act,
            "dropout_proba": dropout_proba
        }
        if block_type == 'block35':
            branch_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                   **conv_block_common_arg_dict)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 2, 1,
                            **conv_block_common_arg_dict),
                ConvBlock2D(block_size * 2, block_size * 2, 3,
                            **conv_block_common_arg_dict)
            )
            branch_2 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 2, 1,
                            **conv_block_common_arg_dict),
                ConvBlock2D(block_size * 2, block_size * 3, 3,
                            **conv_block_common_arg_dict),
                ConvBlock2D(block_size * 3, block_size * 4, 3,
                            **conv_block_common_arg_dict)
            )
            mixed_channel = block_size * 8
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                   **conv_block_common_arg_dict)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 8, 1,
                            **conv_block_common_arg_dict),
                ConvBlock2D(block_size * 8, block_size * 10, [1, 7],
                            **conv_block_common_arg_dict),
                ConvBlock2D(block_size * 10, block_size * 12, [7, 1],
                            **conv_block_common_arg_dict)
            )
            mixed_channel = block_size * 24
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                   **conv_block_common_arg_dict)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 12, 1,
                            **conv_block_common_arg_dict),
                ConvBlock2D(block_size * 12, block_size * 14, [1, 3],
                            **conv_block_common_arg_dict),
                ConvBlock2D(block_size * 14, block_size * 16, [3, 1],
                            **conv_block_common_arg_dict)
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
                              bias=True, norm=norm, act=None, dropout_proba=0.0)
        if self.include_cbam:
            self.cbam = CBAM(gate_channels=in_channels,
                             reduction_ratio=16)
        # TBD: implement of include_context
        self.residual_add = LambdaLayer(
            lambda inputs: inputs[0] + inputs[1] * scale)
        self.act = get_act(act)

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
                 norm="batch", act=DEFAULT_ACT, dropout_proba=0.0):
        super().__init__()
        self.scale = scale
        conv_block_common_arg_dict = {
            "norm": norm,
            "act": act,
            "dropout_proba": dropout_proba
        }
        if block_type == 'block35':
            branch_0 = ConvBlock3D(in_channels, block_size * 2, 1,
                                   **conv_block_common_arg_dict)
            branch_1 = nn.Sequential(
                ConvBlock3D(in_channels, block_size * 2, 1,
                            **conv_block_common_arg_dict),
                ConvBlock3D(block_size * 2, block_size * 2, 3,
                            **conv_block_common_arg_dict)
            )
            branch_2 = nn.Sequential(
                ConvBlock3D(in_channels, block_size * 2, 1,
                            **conv_block_common_arg_dict),
                ConvBlock3D(block_size * 2, block_size * 3, 3,
                            **conv_block_common_arg_dict),
                ConvBlock3D(block_size * 3, block_size * 4, 3,
                            **conv_block_common_arg_dict)
            )
            mixed_channel = block_size * 8
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = ConvBlock3D(in_channels, block_size * 12, 1)
            branch_1 = nn.Sequential(
                ConvBlock3D(in_channels, block_size * 8, 1,
                            **conv_block_common_arg_dict),
                ConvBlock3D(block_size * 8, block_size * 10, [1, 1, 7],
                            **conv_block_common_arg_dict),
                ConvBlock3D(block_size * 10, block_size * 11, [1, 7, 1],
                            **conv_block_common_arg_dict),
                ConvBlock3D(block_size * 11, block_size * 12, [7, 1, 1],
                            **conv_block_common_arg_dict)
            )
            mixed_channel = block_size * 24
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = ConvBlock3D(in_channels, block_size * 12, 1)
            branch_1 = nn.Sequential(
                ConvBlock3D(in_channels, block_size * 12, 1),
                ConvBlock3D(block_size * 12, block_size * 13, [1, 1, 3],
                            **conv_block_common_arg_dict),
                ConvBlock3D(block_size * 13, block_size * 14, [1, 3, 1],
                            **conv_block_common_arg_dict),
                ConvBlock3D(block_size * 14, block_size * 16, [3, 1, 1],
                            **conv_block_common_arg_dict)
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
                              bias=True, norm=norm, act=None, dropout_proba=0.0)
        self.act = get_act(act)

    def forward(self, x):
        mixed = self.mixed(x)
        up = self.up(mixed)
        residual_add = x + up * self.scale
        act = self.act(residual_add)

        return act
