from torch import nn

from .layers import LambdaLayer, ConcatBlock

INPLACE = False


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


# TBD
# add Skip Connection output
# implement of include_context with transformer
class InceptionResNetV2(nn.Module):
    def __init__(self, n_input_channels, block_size=16,
                 padding='valid', z_channel_preserve=True,
                 include_context=False):
        super().__init__()
        if padding == 'valid':
            pool_3x3_padding = 0
        elif padding == 'same':
            pool_3x3_padding = 1

        if z_channel_preserve:
            z_stride = [1, 2, 2]
        else:
            z_stride = 2
        # Stem block
        self.stem = nn.ModuleDict({
            'stem_layer_1': ConvBlock3D(n_input_channels, block_size * 2, 3, stride=z_stride, padding=padding),
            'stem_layer_2': ConvBlock3D(block_size * 2, block_size * 2, 3, padding=padding),
            'stem_layer_3': ConvBlock3D(block_size * 2, block_size * 4, 3),
            'stem_layer_4': nn.MaxPool3d(3, stride=2, padding=pool_3x3_padding),
            'stem_layer_5': ConvBlock3D(block_size * 4, block_size * 5, 1, padding=padding),
            'stem_layer_6': ConvBlock3D(block_size * 5, block_size * 12, 3, padding=padding),
            'stem_layer_7': nn.MaxPool3d(3, stride=z_stride, padding=pool_3x3_padding)
        })
        # Mixed 5b (Inception-A block): 35 x 35 x 320
        mixed_5b_branch_0 = ConvBlock3D(block_size * 12, block_size * 6, 1)
        mixed_5b_branch_1 = nn.Sequential(
            ConvBlock3D(block_size * 12, block_size * 3, 1),
            ConvBlock3D(block_size * 3, block_size * 4, 5)
        )
        mixed_5b_branch_2 = nn.Sequential(
            ConvBlock3D(block_size * 12, block_size * 4, 1),
            ConvBlock3D(block_size * 4, block_size * 6, 3),
            ConvBlock3D(block_size * 6, block_size * 6, 3)
        )
        mixed_5b_branch_pool = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1),
            ConvBlock3D(block_size * 12, block_size * 4, 1)
        )
        mixed_5b_branches = [mixed_5b_branch_0, mixed_5b_branch_1,
                             mixed_5b_branch_2, mixed_5b_branch_pool]
        self.mixed_5b = ConcatBlock(mixed_5b_branches)
        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        self.block_35 = nn.Sequential(*[
            Inception_Resnet_Block3D(in_channels=block_size * 20, scale=0.17,
                                     block_type="block35",
                                     block_size=block_size,
                                     include_context=include_context)
            for _ in range(1, 11)
        ])
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        mixed_6a_branch_0 = ConvBlock3D(block_size * 20, block_size * 24, 3,
                                        stride=2, padding=padding)
        mixed_6a_branch_1 = nn.Sequential(
            ConvBlock3D(block_size * 20, block_size * 16, 1),
            ConvBlock3D(block_size * 16, block_size * 16, 3),
            ConvBlock3D(block_size * 16, block_size * 24, 3,
                        stride=2, padding=padding)
        )
        mixed_6a_branch_pool = nn.MaxPool3d(3, stride=2,
                                            padding=pool_3x3_padding)
        mixed_6a_branches = [mixed_6a_branch_0, mixed_6a_branch_1,
                             mixed_6a_branch_pool]
        self.mixed_6a = ConcatBlock(mixed_6a_branches)
        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        self.block_17 = nn.Sequential(*[
            Inception_Resnet_Block3D(in_channels=block_size * 68, scale=0.1,
                                     block_type="block17",
                                     block_size=block_size,
                                     include_context=(include_context and block_idx == 20))
            for block_idx in range(1, 21)
        ])
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        mixed_7a_branch_0 = nn.Sequential(
            ConvBlock3D(block_size * 68, block_size * 16, 1),
            ConvBlock3D(block_size * 16, block_size * 24, 3,
                        stride=z_stride, padding=padding)
        )
        mixed_7a_branch_1 = nn.Sequential(
            ConvBlock3D(block_size * 68, block_size * 16, 1),
            ConvBlock3D(block_size * 16, block_size * 18, 3,
                        stride=z_stride, padding=padding)
        )
        mixed_7a_branch_2 = nn.Sequential(
            ConvBlock3D(block_size * 68, block_size * 16, 1),
            ConvBlock3D(block_size * 16, block_size * 18, 3),
            ConvBlock3D(block_size * 18, block_size * 20, 3,
                        stride=z_stride, padding=padding)
        )
        mixed_7a_branch_pool = nn.MaxPool3d(3, stride=z_stride,
                                            padding=pool_3x3_padding)
        mixed_7a_branches = [mixed_7a_branch_0, mixed_7a_branch_1,
                             mixed_7a_branch_2, mixed_7a_branch_pool]
        self.mixed_7a = ConcatBlock(mixed_7a_branches)
        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        self.block_8 = nn.Sequential(*[
            Inception_Resnet_Block3D(in_channels=block_size * 130, scale=0.2,
                                     block_type="block8",
                                     block_size=block_size,
                                     include_context=(include_context and block_idx == 10))
            for block_idx in range(1, 11)
        ])
        # Final convolution block: 8 x 8 x 1536
        self.final_conv = ConvBlock3D(block_size * 130, block_size * 96, 1)

    def forward(self, input_tensor):
        stem = input_tensor
        for layer_name, layer in self.stem.items():
            stem = layer(stem)

        mixed_5b = self.mixed_5b(stem)
        block_35 = self.block_35(mixed_5b)
        mixed_6a = self.mixed_6a(block_35)
        block_17 = self.block_17(mixed_6a)
        mixed_7a = self.mixed_7a(block_17)
        block_8 = self.block_8(mixed_7a)
        output = self.final_conv(block_8)
        return output
