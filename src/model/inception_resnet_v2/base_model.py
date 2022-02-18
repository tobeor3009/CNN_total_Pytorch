import torch
from torch import nn

from .layers import LambdaLayer

INPLACE = False


# Assume Channel First
class ConcatBlock(nn.Module):
    def __init__(self, layer_list, dim=1):
        super().__init__()
        self.layer_list = layer_list
        self.dim = dim

    def forward(self, x):
        tensor_list = [layer(x) for layer in self.layer_list]
        return torch.cat(tensor_list, self.dim)


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


# TBD: layer Naming Issue
class Inception_Resnet_Block(nn.Module):
    def __init__(self, in_channels, scale, block_type,
                 activation='relu6', include_context=False, context_head_nums=8):
        super().__init__()
        if block_type == 'bloack35':
            branch_0 = ConvBlock2D(in_channels, 32, 1)
            branch_1 = nn.ModuleDict({
                'branch_1_1': ConvBlock2D(in_channels, 32, 1),
                'branch_1_2': ConvBlock2D(32, 32, 3)
            })
            branch_2 = nn.ModuleDict({
                'branch_2_1': ConvBlock2D(in_channels, 32, 1),
                'branch_2_2': ConvBlock2D(32, 48, 3),
                'branch_2_3': ConvBlock2D(48, 64, 3)
            })
            mixed_channel = 128
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'bloack17':
            branch_0 = ConvBlock2D(in_channels, 192, 1)
            branch_1 = nn.ModuleDict({
                'branch_1_1': ConvBlock2D(in_channels, 128, 1),
                'branch_1_2': ConvBlock2D(128, 160, [1, 7]),
                'branch_1_3': ConvBlock2D(160, 192, [7, 1])
            })
            mixed_channel = 384
            branches = [branch_0, branch_1]
        elif block_type == 'bloack8':
            branch_0 = ConvBlock2D(in_channels, 192, 1)
            branch_1 = nn.ModuleDict({
                'branch_1_1': ConvBlock2D(in_channels, 192, 1),
                'branch_1_2': ConvBlock2D(192, 224, [1, 3]),
                'branch_1_3': ConvBlock2D(224, 256, [3, 1])
            })
            mixed_channel = 448
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))
        self.mixed = ConcatBlock(branches)
        # TBD: Name?
        self.up = ConvBlock2D(mixed_channel, in_channels, 1,
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


class InceptionResNetV2(nn.Module):
    def __init__(self, n_input_channels, padding='valid', pooling=None, include_context=False):
        super().__init__()
        if pooling == 'valid':
            pool_3x3_padding = 0
        elif pooling == 'same':
            pool_3x3_padding = 1
        # Stem block
        self.stem = nn.ModuleDict({
            'stem_layer_1': ConvBlock2D(n_input_channels, 32, 3, stride=2, padding=padding),
            'stem_layer_2': ConvBlock2D(32, 32, 3, padding=padding),
            'stem_layer_3': ConvBlock2D(32, 64, 3),
            'stem_layer_4': nn.MaxPool2d(3, stride=2, padding=padding),
            'stem_layer_5': ConvBlock2D(64, 80, 1, padding=padding),
            'stem_layer_6': ConvBlock2D(80, 192, 1, padding=padding),
            'stem_layer_7': nn.MaxPool2d(3, stride=2, padding=pool_3x3_padding)
        })
        # Mixed 5b (Inception-A block): 35 x 35 x 320
        mixed_5b_branch_0 = ConvBlock2D(192, 96, 1)
        mixed_5b_branch_1 = nn.ModuleDict({
            'mixed_5b_branch_1_layer_1': ConvBlock2D(192, 48, 1),
            'mixed_5b_branch_1_layer_2': ConvBlock2D(48, 64, 5)
        })
        mixed_5b_branch_2 = nn.ModuleDict({
            'mixed_5b_branch_2_layer_1': ConvBlock2D(192, 64, 1),
            'mixed_5b_branch_2_layer_2': ConvBlock2D(64, 96, 3),
            'mixed_5b_branch_2_layer_3': ConvBlock2D(96, 96, 3)
        })
        mixed_5b_branch_pool = nn.ModuleDict({
            'mixed_5b_branch_pool_layer_1': nn.AvgPool2d(3, stride=1, padding=pool_3x3_padding),
            'mixed_5b_branch_pool_layer_2': ConvBlock2D(192, 64, 1)
        })
        mixed_5b_branches = [mixed_5b_branch_0, mixed_5b_branch_1,
                             mixed_5b_branch_2, mixed_5b_branch_pool]
        self.mixed_5b = ConcatBlock(mixed_5b_branches)
        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        self.block_35 = nn.ModuleDict({
            f'block35_{block_idx}': Inception_Resnet_Block(in_channels=320, scale=0.17,
                                                           block_type=35)
            for block_idx in range(1, 11)
        })
