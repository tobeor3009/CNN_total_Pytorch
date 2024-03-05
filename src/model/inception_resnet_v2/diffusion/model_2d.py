from torch import nn
import numpy as np
from .diffusion_layer import ConvBlock2D, SinusoidalPosEmb
from ..common_module.layers import LambdaLayer, ConcatBlock, DEFAULT_ACT
from ..common_module.inception_layers import Inception_Resnet_Block2D, INPLACE

def get_skip_connect_channel_list(block_size, mini=False):
    if mini:
        return np.array([block_size * 2, block_size * 4, block_size * 12])
    else:
        return np.array([block_size * 2, block_size * 4, block_size * 12,
                        block_size * 68, block_size * 130])


class InceptionResNetV2_UNet(nn.Module):
    def __init__(self, n_input_channels, block_size=16,
                 padding='valid', norm="batch", act=DEFAULT_ACT, last_act=DEFAULT_ACT,
                 last_channel_ratio=1, include_cbam=False):
        super().__init__()
        if padding == 'valid':
            padding_3x3 = 0
        elif padding == 'same':
            padding_3x3 = 1

        time_emb_channels = block_size * 8

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(block_size * 2),
            nn.Linear(block_size * 2, time_emb_channels),
            nn.GELU(),
            nn.Linear(time_emb_channels, time_emb_channels)
        )


        # Stem block
        self.stem = nn.ModuleDict({
            'stem_layer_1': ConvBlock2D(n_input_channels, block_size * 2, 3,
                                        stride=2, padding=padding_3x3, norm=norm, act=act, 
                                        time_emb_channels=time_emb_channels),
            'stem_layer_2': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                        padding=padding_3x3, norm=norm, act=act,
                                        time_emb_channels=time_emb_channels),
            'stem_layer_3': ConvBlock2D(block_size * 2, block_size * 4, 3, norm=norm, act=act,
                                        time_emb_channels=time_emb_channels),
            'stem_layer_4': nn.MaxPool2d(3, stride=2, padding=padding_3x3),
            'stem_layer_5': ConvBlock2D(block_size * 4, block_size * 5, 1, norm=norm, act=act,
                                        time_emb_channels=time_emb_channels),
            'stem_layer_6': ConvBlock2D(block_size * 5, block_size * 12, 3,
                                        padding=padding_3x3, norm=norm, act=act,
                                        time_emb_channels=time_emb_channels),
            'stem_layer_7': nn.MaxPool2d(3, stride=2, padding=padding_3x3)
        })
        # Mixed 5b (Inception-A block): 35 x 35 x 320
        mixed_5b_branch_0 = ConvBlock2D(block_size * 12, block_size * 6, 1,
                                        norm=norm, act=act)
        mixed_5b_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 12, block_size *
                        3, 1, norm=norm, act=act),
            ConvBlock2D(block_size * 3, block_size * 4, 5, norm=norm, act=act)
        )
        mixed_5b_branch_2 = nn.Sequential(
            ConvBlock2D(block_size * 12, block_size *
                        4, 1, norm=norm, act=act),
            ConvBlock2D(block_size * 4, block_size * 6, 3, norm=norm, act=act),
            ConvBlock2D(block_size * 6, block_size * 6, 3, norm=norm, act=act)
        )
        mixed_5b_branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock2D(block_size * 12, block_size * 4, 1, norm=norm, act=act)
        )
        mixed_5b_branches = [mixed_5b_branch_0, mixed_5b_branch_1,
                             mixed_5b_branch_2, mixed_5b_branch_pool]
        self.mixed_5b = ConcatBlock(mixed_5b_branches)
        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        self.block_35 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 20, scale=0.17,
                                     block_type="block35", norm=norm, act=act,
                                     include_cbam=include_cbam)
            for _ in range(1, 11)
        ])
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        mixed_6a_branch_0 = ConvBlock2D(block_size * 20, block_size * 24, 3,
                                        stride=2, padding=padding_3x3, norm=norm, act=act)
        mixed_6a_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 20, block_size * 16, 1,
                        norm=norm, act=act),
            ConvBlock2D(block_size * 16, block_size * 16, 3,
                        norm=norm, act=act),
            ConvBlock2D(block_size * 16, block_size * 24, 3,
                        stride=2, padding=padding_3x3, norm=norm, act=act)
        )
        mixed_6a_branch_pool = nn.MaxPool2d(3, stride=2,
                                            padding=padding_3x3)
        mixed_6a_branches = [mixed_6a_branch_0, mixed_6a_branch_1,
                             mixed_6a_branch_pool]
        self.mixed_6a = ConcatBlock(mixed_6a_branches)
        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        self.block_17 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 68, scale=0.1,
                                     block_type="block17", norm=norm, act=act,
                                     include_cbam=include_cbam)
            for block_idx in range(1, 21)
        ])
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        mixed_7a_branch_0 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1,
                        norm=norm, act=act),
            ConvBlock2D(block_size * 16, block_size *
                        24, 3, stride=2, padding=padding_3x3,
                        norm=norm, act=act)
        )
        mixed_7a_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1,
                        norm=norm, act=act),
            ConvBlock2D(block_size * 16, block_size *
                        18, 3, stride=2, padding=padding_3x3,
                        norm=norm, act=act)
        )
        mixed_7a_branch_2 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1,
                        norm=norm, act=act),
            ConvBlock2D(block_size * 16, block_size * 18, 3,
                        norm=norm, act=act),
            ConvBlock2D(block_size * 18, block_size * 20, 3, stride=2,
                        padding=padding_3x3, norm=norm, act=act)
        )
        mixed_7a_branch_pool = nn.MaxPool2d(3, stride=2,
                                            padding=padding_3x3)
        mixed_7a_branches = [mixed_7a_branch_0, mixed_7a_branch_1,
                             mixed_7a_branch_2, mixed_7a_branch_pool]
        self.mixed_7a = ConcatBlock(mixed_7a_branches)
        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        self.block_8 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 130, scale=0.2,
                                     block_type="block8", norm=norm, act=act,
                                     include_cbam=include_cbam)
            for block_idx in range(1, 11)
        ])
        # Final convolution block: 8 x 8 x 1536
        self.final_conv = ConvBlock2D(block_size * 130, block_size * 96 * last_channel_ratio, 1,
                                      norm=norm, act=last_act)

    def forward(self, input_tensor):
        # skip connection name list
        # ["stem_layer_1", "stem_layer_4", "stem_layer_7", "mixed_6a", "mixed_7a"]
        skip_connect_list = []
        stem = input_tensor
        for idx, (layer_name, layer) in enumerate(self.stem.items()):
            stem = layer(stem)
            if idx in [0, 3, 6]:
                skip_connect_list.append(stem)
        mixed_5b = self.mixed_5b(stem)
        block_35 = self.block_35(mixed_5b)
        # skip connect target
        mixed_6a = self.mixed_6a(block_35)
        block_17 = self.block_17(mixed_6a)
        # skip connect target
        mixed_7a = self.mixed_7a(block_17)
        block_8 = self.block_8(mixed_7a)
        output = self.final_conv(block_8)
        return output
