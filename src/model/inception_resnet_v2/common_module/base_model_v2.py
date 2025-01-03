from torch import nn
import numpy as np
from functools import partial
from .layers import LambdaLayer, ConcatBlock, ConvBlock2D, ConvBlock3D, DEFAULT_ACT
from .inception_layers import Inception_Resnet_Block2D, Inception_Resnet_Block3D
from src.model.train_util.common import process_with_checkpoint
INPLACE = False

def get_skip_connect_channel_list(block_size, mini=False):
    if mini:
        return np.array([block_size * 2, block_size * 4, block_size * 12])
    else:
        return np.array([block_size * 2, block_size * 2, block_size * 4, block_size * 12,
                        block_size * 68, block_size * 130])


class InceptionResNetV2_2D(nn.Module):
    def __init__(self, n_input_channels, block_size=16,
                 padding='valid', norm="batch", act=DEFAULT_ACT, last_act=DEFAULT_ACT, dropout_proba=0.0,
                 last_channel_ratio=1, include_cbam=False,
                 include_skip_connection_tensor=False, use_checkpoint=False):
        super().__init__()
        self.include_skip_connection_tensor = include_skip_connection_tensor
        self.process_with_checkpoint = partial(process_with_checkpoint, use_checkpoint=use_checkpoint)
        if padding == 'valid':
            padding_3x3 = 0
        elif padding == 'same':
            padding_3x3 = 1
        
        conv_block_common_arg_dict = {
            "norm": norm,
            "act": act,
            "dropout_proba": dropout_proba
        }
        # Stem block
        self.stem = nn.ModuleDict({
            'stem_layer_0_0': ConvBlock2D(n_input_channels, block_size * 2, 3,
                                        padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_0_1': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                        padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_1_0': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                         padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_1_1': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                        stride=2, padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_2': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                        padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_3': ConvBlock2D(block_size * 2, block_size * 4, 3, **conv_block_common_arg_dict),
            'stem_layer_4': nn.MaxPool2d(3, stride=2, padding=padding_3x3),
            'stem_layer_5': ConvBlock2D(block_size * 4, block_size * 5, 1, **conv_block_common_arg_dict),
            'stem_layer_6': ConvBlock2D(block_size * 5, block_size * 12, 3,
                                        padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_7': nn.MaxPool2d(3, stride=2, padding=padding_3x3)
        })
        # Mixed 5b (Inception-A block): 35 x 35 x 320
        mixed_5b_branch_0 = ConvBlock2D(block_size * 12, block_size * 6, 1,
                                        **conv_block_common_arg_dict)
        mixed_5b_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 12, block_size *3, 1, **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 3, block_size * 4, 5, **conv_block_common_arg_dict)
        )
        mixed_5b_branch_2 = nn.Sequential(
            ConvBlock2D(block_size * 12, block_size *4, 1, **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 4, block_size * 6, 3, **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 6, block_size * 6, 3, **conv_block_common_arg_dict)
        )
        mixed_5b_branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock2D(block_size * 12, block_size * 4, 1, **conv_block_common_arg_dict)
        )
        mixed_5b_branches = [mixed_5b_branch_0, mixed_5b_branch_1,
                             mixed_5b_branch_2, mixed_5b_branch_pool]
        self.mixed_5b = ConcatBlock(mixed_5b_branches)
        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        self.block_35_list = nn.ModuleList([
            Inception_Resnet_Block2D(in_channels=block_size * 20, scale=0.17,
                                     block_type="block35", include_cbam=include_cbam,
                                     **conv_block_common_arg_dict)
            for _ in range(1, 11)
        ])
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        mixed_6a_branch_0 = ConvBlock2D(block_size * 20, block_size * 24, 3,
                                        stride=2, padding=padding_3x3, **conv_block_common_arg_dict)
        mixed_6a_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 20, block_size * 16, 1,
                        **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 16, block_size * 16, 3,
                        **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 16, block_size * 24, 3,
                        stride=2, padding=padding_3x3, **conv_block_common_arg_dict)
        )
        mixed_6a_branch_pool = nn.MaxPool2d(3, stride=2,
                                            padding=padding_3x3)
        mixed_6a_branches = [mixed_6a_branch_0, mixed_6a_branch_1,
                             mixed_6a_branch_pool]
        self.mixed_6a = ConcatBlock(mixed_6a_branches)
        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        self.block_17_list = nn.ModuleList([
            Inception_Resnet_Block2D(in_channels=block_size * 68, scale=0.1,
                                     block_type="block17", include_cbam=include_cbam,
                                     **conv_block_common_arg_dict)
            for block_idx in range(1, 21)
        ])
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        mixed_7a_branch_0 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1,
                        **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 16, block_size *
                        24, 3, stride=2, padding=padding_3x3,
                        **conv_block_common_arg_dict)
        )
        mixed_7a_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1,
                        **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 16, block_size *
                        18, 3, stride=2, padding=padding_3x3,
                        **conv_block_common_arg_dict)
        )
        mixed_7a_branch_2 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1,
                        **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 16, block_size * 18, 3,
                        **conv_block_common_arg_dict),
            ConvBlock2D(block_size * 18, block_size * 20, 3, stride=2,
                        padding=padding_3x3, **conv_block_common_arg_dict)
        )
        mixed_7a_branch_pool = nn.MaxPool2d(3, stride=2,
                                            padding=padding_3x3)
        mixed_7a_branches = [mixed_7a_branch_0, mixed_7a_branch_1,
                             mixed_7a_branch_2, mixed_7a_branch_pool]
        self.mixed_7a = ConcatBlock(mixed_7a_branches)
        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        self.block_8_list = nn.ModuleList([
            Inception_Resnet_Block2D(in_channels=block_size * 130, scale=0.2,
                                     block_type="block8", include_cbam=include_cbam,
                                     **conv_block_common_arg_dict)
            for block_idx in range(1, 11)
        ])
        # Final convolution block: 8 x 8 x 1536
        self.final_conv = ConvBlock2D(block_size * 130, block_size * 96 * last_channel_ratio, 1,
                                      norm=norm, act=last_act)

    def process_stem(self, x):
        stem = x
        output_list = []
        for index, (layer_name, layer) in enumerate(self.stem.items()):
            stem = layer(stem)
            # layer_name in ["stem_layer_0", "stem_layer_1", "stem_layer_4", "stem_layer_7"]
            if self.include_skip_connection_tensor and (index in [1, 3, 6, 9]):
                output_list.append(stem)
        return stem, output_list

    def forward(self, input_tensor):
        # stem, skip_connect_index = self.process_with_checkpoint(self.process_stem, input_tensor)
        stem, output_list = self.process_with_checkpoint(self.process_stem, input_tensor)
        block_35_tensor = self.process_with_checkpoint(self.mixed_5b, stem)
        for block_35 in self.block_35_list:
            block_35_tensor = self.process_with_checkpoint(block_35, block_35_tensor)
        mixed_6a_tensor = self.process_with_checkpoint(self.mixed_6a, block_35_tensor)
        output_list.append(mixed_6a_tensor)
        block_17_tensor = mixed_6a_tensor
        for block_17 in self.block_17_list:
            block_17_tensor = self.process_with_checkpoint(block_17, block_17_tensor)
        mixed_7a_tensor = self.process_with_checkpoint(self.mixed_7a, block_17_tensor)
        block_8_tensor = mixed_7a_tensor
        for block_8 in self.block_8_list:
            block_8_tensor = self.process_with_checkpoint(block_8, block_8_tensor)
        output = self.process_with_checkpoint(self.final_conv, block_8_tensor)
        output_list.append(output)
        return output_list

class InceptionResNetV2_3D(nn.Module):
    def __init__(self, n_input_channels, block_size=16,
                 padding='valid', norm="batch", act=DEFAULT_ACT, dropout_proba=0.0,
                 last_act=DEFAULT_ACT, z_channel_preserve=True,
                 include_skip_connection_tensor=False, use_checkpoint=False):
        super().__init__()
        self.include_skip_connection_tensor = include_skip_connection_tensor
        self.process_with_checkpoint = partial(process_with_checkpoint, use_checkpoint=use_checkpoint)

        conv_block_common_arg_dict = {
            "norm": norm,
            "act": act,
            "dropout_proba": dropout_proba
        }
        if padding == 'valid':
            padding_3x3 = 0
        elif padding == 'same':
            padding_3x3 = 1

        if z_channel_preserve:
            z_stride = [1, 2, 2]
        else:
            z_stride = 2
        # Stem block
        self.stem = nn.ModuleDict({
            'stem_layer_0_0': ConvBlock3D(n_input_channels, block_size * 2, 3,
                                        padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_0_1': ConvBlock3D(block_size * 2, block_size * 2, 3,
                                        padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_1_0': ConvBlock3D(block_size * 2, block_size * 2, 3,
                                        padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_1_1': ConvBlock3D(block_size * 2, block_size * 2, 3,
                                        stride=z_stride, padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_2': ConvBlock3D(block_size * 2, block_size * 2, 3,
                                        padding=padding_3x3, norm=norm, act=act),
            'stem_layer_3': ConvBlock3D(block_size * 2, block_size * 4, 3, **conv_block_common_arg_dict),
            'stem_layer_4': nn.MaxPool3d(3, stride=z_stride, padding=padding_3x3),
            'stem_layer_5': ConvBlock3D(block_size * 4, block_size * 5, 1, **conv_block_common_arg_dict),
            'stem_layer_6': ConvBlock3D(block_size * 5, block_size * 12, 3,
                                        padding=padding_3x3, **conv_block_common_arg_dict),
            'stem_layer_7': nn.MaxPool3d(3, stride=z_stride, padding=padding_3x3)
        })
        # Mixed 5b (Inception-A block): 35 x 35 x 320
        mixed_5b_branch_0 = ConvBlock3D(block_size * 12, block_size * 6, 1,
                                        **conv_block_common_arg_dict)
        mixed_5b_branch_1 = nn.Sequential(
            ConvBlock3D(block_size * 12, block_size * 3, 1,
                        **conv_block_common_arg_dict),
            ConvBlock3D(block_size * 3, block_size * 4, 5,
                        **conv_block_common_arg_dict)
        )
        mixed_5b_branch_2 = nn.Sequential(
            ConvBlock3D(block_size * 12, block_size * 4, 1,
                        **conv_block_common_arg_dict),
            ConvBlock3D(block_size * 4, block_size * 6, 3,
                        **conv_block_common_arg_dict),
            ConvBlock3D(block_size * 6, block_size * 6, 3,
                        **conv_block_common_arg_dict)
        )
        mixed_5b_branch_pool = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1),
            ConvBlock3D(block_size * 12, block_size * 4, 1,
                        **conv_block_common_arg_dict)
        )
        mixed_5b_branches = [mixed_5b_branch_0, mixed_5b_branch_1,
                             mixed_5b_branch_2, mixed_5b_branch_pool]
        self.mixed_5b = ConcatBlock(mixed_5b_branches)
        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        self.block_35_list = nn.ModuleList([
            Inception_Resnet_Block3D(in_channels=block_size * 20, scale=0.17,
                                     block_type="block35", block_size=block_size,
                                     **conv_block_common_arg_dict)
            for _ in range(1, 11)
        ])
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        mixed_6a_branch_0 = ConvBlock3D(block_size * 20, block_size * 24, 3,
                                        stride=z_stride, padding=padding_3x3,
                                        **conv_block_common_arg_dict)
        mixed_6a_branch_1 = nn.Sequential(
            ConvBlock3D(block_size * 20, block_size * 16, 1,
                        **conv_block_common_arg_dict),
            ConvBlock3D(block_size * 16, block_size * 16, 3,
                        **conv_block_common_arg_dict),
            ConvBlock3D(block_size * 16, block_size * 24, 3,
                        stride=z_stride, padding=padding_3x3,
                        **conv_block_common_arg_dict)
        )
        mixed_6a_branch_pool = nn.MaxPool3d(3, stride=z_stride,
                                            padding=padding_3x3)
        mixed_6a_branches = [mixed_6a_branch_0, mixed_6a_branch_1,
                             mixed_6a_branch_pool]
        self.mixed_6a = ConcatBlock(mixed_6a_branches)
        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        self.block_17_list = nn.ModuleList([
            Inception_Resnet_Block3D(in_channels=block_size * 68, scale=0.1,
                                     block_type="block17", block_size=block_size,
                                     **conv_block_common_arg_dict)
            for block_idx in range(1, 21)
        ])
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        mixed_7a_branch_0 = nn.Sequential(
            ConvBlock3D(block_size * 68, block_size * 16, 1,
                        norm=norm, act=act),
            ConvBlock3D(block_size * 16, block_size * 24, 3,
                        stride=z_stride, padding=padding_3x3,
                        norm=norm, act=act)
        )
        mixed_7a_branch_1 = nn.Sequential(
            ConvBlock3D(block_size * 68, block_size * 16, 1,
                        **conv_block_common_arg_dict),
            ConvBlock3D(block_size * 16, block_size * 18, 3,
                        stride=z_stride, padding=padding_3x3,
                        **conv_block_common_arg_dict)
        )
        mixed_7a_branch_2 = nn.Sequential(
            ConvBlock3D(block_size * 68, block_size * 16, 1,
                        **conv_block_common_arg_dict),
            ConvBlock3D(block_size * 16, block_size * 18, 3,
                        **conv_block_common_arg_dict),
            ConvBlock3D(block_size * 18, block_size * 20, 3,
                        stride=z_stride, padding=padding_3x3,
                        **conv_block_common_arg_dict)
        )
        mixed_7a_branch_pool = nn.MaxPool3d(3, stride=z_stride,
                                            padding=padding_3x3)
        mixed_7a_branches = [mixed_7a_branch_0, mixed_7a_branch_1,
                             mixed_7a_branch_2, mixed_7a_branch_pool]
        self.mixed_7a = ConcatBlock(mixed_7a_branches)
        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        self.block_8_list = nn.ModuleList([
            Inception_Resnet_Block3D(in_channels=block_size * 130, scale=0.2,
                                     block_type="block8", block_size=block_size,
                                     **conv_block_common_arg_dict)
            for block_idx in range(1, 11)
        ])
        # Final convolution block: 8 x 8 x 1536
        self.final_conv = ConvBlock3D(block_size * 130, block_size * 96, 1,
                                      norm=norm, act=last_act)

    def process_stem(self, x):
        stem = x
        output_list = []
        for index, (layer_name, layer) in enumerate(self.stem.items()):
            stem = layer(stem)
            # layer_name in ["stem_layer_0", "stem_layer_1", "stem_layer_4", "stem_layer_7"]
            if self.include_skip_connection_tensor and (index in [1, 3, 6, 9]):
                output_list.append(stem)
        return stem, output_list

    def forward(self, input_tensor):
        # stem, skip_connect_index = self.process_with_checkpoint(self.process_stem, input_tensor)
        stem, output_list = self.process_with_checkpoint(self.process_stem, input_tensor)
        block_35_tensor = self.process_with_checkpoint(self.mixed_5b, stem)
        for block_35 in self.block_35_list:
            block_35_tensor = self.process_with_checkpoint(block_35, block_35_tensor)
        mixed_6a_tensor = self.process_with_checkpoint(self.mixed_6a, block_35_tensor)
        output_list.append(mixed_6a_tensor)
        block_17_tensor = mixed_6a_tensor
        for block_17 in self.block_17_list:
            block_17_tensor = self.process_with_checkpoint(block_17, block_17_tensor)
        mixed_7a_tensor = self.process_with_checkpoint(self.mixed_7a, block_17_tensor)
        block_8_tensor = mixed_7a_tensor
        for block_8 in self.block_8_list:
            block_8_tensor = self.process_with_checkpoint(block_8, block_8_tensor)
        output = self.process_with_checkpoint(self.final_conv, block_8_tensor)
        output_list.append(output)
        return output_list