from torch import nn

from .layers import LambdaLayer, ConcatBlock

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


# TBD: layer Naming Issue
class Inception_Resnet_Block(nn.Module):
    def __init__(self, in_channels, scale, block_type,
                 activation='relu6', include_context=False, context_head_nums=8):
        super().__init__()
        if block_type == 'block35':
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
        elif block_type == 'block17':
            branch_0 = ConvBlock2D(in_channels, 192, 1)
            branch_1 = nn.ModuleDict({
                'branch_1_1': ConvBlock2D(in_channels, 128, 1),
                'branch_1_2': ConvBlock2D(128, 160, [1, 7]),
                'branch_1_3': ConvBlock2D(160, 192, [7, 1])
            })
            mixed_channel = 384
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
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


# TBD
# add Skip Connection output
# implement of include_context with transformer
class InceptionResNetV2(nn.Module):
    def __init__(self, n_input_channels,
                 padding='valid', include_context=False):
        super().__init__()
        if padding == 'valid':
            pool_3x3_padding = 0
        elif padding == 'same':
            pool_3x3_padding = 1
        # Stem block
        self.stem = nn.ModuleDict({
            'stem_layer_1': ConvBlock2D(n_input_channels, 32, 3, stride=2, padding=padding),
            'stem_layer_2': ConvBlock2D(32, 32, 3, padding=padding),
            'stem_layer_3': ConvBlock2D(32, 64, 3),
            'stem_layer_4': nn.MaxPool2d(3, stride=2, padding=padding),
            'stem_layer_5': ConvBlock2D(64, 80, 1, padding=padding),
            'stem_layer_6': ConvBlock2D(80, 192, 3, padding=padding),
            'stem_layer_7': nn.MaxPool2d(3, stride=2, padding=pool_3x3_padding)
        })
        # Mixed 5b (Inception-A block): 35 x 35 x 320
        mixed_5b_branch_0 = ConvBlock2D(192, 96, 1)
        mixed_5b_branch_1 = nn.Sequential(
            ConvBlock2D(192, 48, 1),
            ConvBlock2D(48, 64, 5)
        )
        mixed_5b_branch_2 = nn.Sequential(
            ConvBlock2D(192, 64, 1),
            ConvBlock2D(64, 96, 3),
            ConvBlock2D(96, 96, 3)
        )
        mixed_5b_branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock2D(192, 64, 1)
        )
        mixed_5b_branches = [mixed_5b_branch_0, mixed_5b_branch_1,
                             mixed_5b_branch_2, mixed_5b_branch_pool]
        self.mixed_5b = ConcatBlock(mixed_5b_branches)
        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        self.block_35 = nn.Sequential(*[
            Inception_Resnet_Block(in_channels=320, scale=0.17,
                                   block_type="block35",
                                   include_context=include_context)
            for _ in range(1, 11)
        ])
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        mixed_6a_branch_0 = ConvBlock2D(320, 384, 3, stride=2, padding=padding)
        mixed_6a_branch_1 = nn.Sequential(
            ConvBlock2D(320, 256, 1),
            ConvBlock2D(256, 256, 3),
            ConvBlock2D(256, 384, 3, stride=2, padding=padding)
        )
        mixed_6a_branch_pool = nn.MaxPool2d(3, stride=2,
                                            padding=pool_3x3_padding)
        mixed_6a_branches = [mixed_6a_branch_0, mixed_6a_branch_1,
                             mixed_6a_branch_pool]
        self.mixed_6a = ConcatBlock(mixed_6a_branches)
        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        self.block_17 = nn.Sequential(*[
            Inception_Resnet_Block(in_channels=1088, scale=0.1,
                                   block_type="block17",
                                   include_context=(include_context and block_idx == 20))
            for block_idx in range(1, 21)
        ])
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        mixed_7a_branch_0 = nn.Sequential(
            ConvBlock2D(1088, 256, 1),
            ConvBlock2D(256, 384, 3, stride=2, padding=padding)
        )
        mixed_7a_branch_1 = nn.Sequential(
            ConvBlock2D(1088, 256, 1),
            ConvBlock2D(256, 288, 3, stride=2, padding=padding)
        )
        mixed_7a_branch_2 = nn.Sequential(
            ConvBlock2D(1088, 256, 1),
            ConvBlock2D(256, 288, 3),
            ConvBlock2D(288, 320, 3, stride=2, padding=padding)
        )
        mixed_7a_branch_pool = nn.MaxPool2d(3, stride=2,
                                            padding=pool_3x3_padding)
        mixed_7a_branches = [mixed_7a_branch_0, mixed_7a_branch_1,
                             mixed_7a_branch_2, mixed_7a_branch_pool]
        self.mixed_7a = ConcatBlock(mixed_7a_branches)
        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        self.block_8 = nn.Sequential(*[
            Inception_Resnet_Block(in_channels=2080, scale=0.2,
                                   block_type="block8",
                                   include_context=(include_context and block_idx == 10))
            for block_idx in range(1, 11)
        ])
        # Final convolution block: 8 x 8 x 1536
        self.final_conv = ConvBlock2D(2080, 1536, 1)

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
