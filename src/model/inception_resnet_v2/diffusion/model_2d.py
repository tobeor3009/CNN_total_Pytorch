from torch import nn
import numpy as np
import torch
from .diffusion_layer import default
from .diffusion_layer import ConvBlock2D, Inception_Resnet_Block2D, SinusoidalPosEmb
from .diffusion_layer import MaxPool2d, AvgPool2d

from ..common_module.layers import LambdaLayer, ConcatBlock


def get_skip_connect_channel_list(block_size, mini=False):
    if mini:
        return np.array([block_size * 2, block_size * 4, block_size * 12])
    else:
        return np.array([block_size * 2, block_size * 4, block_size * 12,
                        block_size * 68, block_size * 130])


class InceptionResNetV2_UNet(nn.Module):
    def __init__(self, in_channel, cond_channel, img_size, block_size=16,
                 norm="group", act="silu", last_act=None, num_class_embeds=None,
                 last_channel_ratio=1, include_cbam=False, self_condition=False,
                 use_checkpoint=False, attn_info_list=[None, False, False, False, True],
                 block_depth_info="mini"):
        super().__init__()
        
        if isinstance(block_depth_info, str):
            if block_depth_info == "tiny":
                block_depth_list = [1, 2, 1]
            elif block_depth_info == "mini":
                block_depth_list = [2, 4, 2]
            elif block_depth_info == "middle":
                block_depth_list = [5, 10, 5]
            elif block_depth_info == "large":
                block_depth_list = [10, 20, 10]
        elif isinstance(block_depth_info, int):
            block_depth_list = np.array([1, 2, 1]) * block_depth_info
        else:
            block_depth_list = block_depth_info
        assert isinstance(num_class_embeds, int) and num_class_embeds > 0, "you need to assign positive int to num_class_embeds"
        
        # for compability with Medsegdiff 
        self.image_size = img_size
        self.input_img_channels = cond_channel
        self.mask_channels = in_channel
        self.self_condition = self_condition
        ##################################
        emb_dim = block_size * 16
        emb_type_list = ["seq", "2d"]
        cond_emb_type_list = ["seq"]
        ##################################
        self.padding_3x3 = 1
        self.in_channel = in_channel
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.norm = norm
        self.act = act
        self.use_checkpoint = use_checkpoint

        self.attn_info_list = attn_info_list
        ##################################
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim // 2),
            nn.Linear(emb_dim // 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.class_mlp = nn.Embedding(num_class_embeds, emb_dim)
        # Stem block
        self.cond_stem = self.get_encode_stem(cond_emb_type_list)
        self.stem = self.get_encode_stem(emb_type_list)
        # Mixed 5b (Inception-A block):
        self.cond_mixed_5b = self.get_encode_mixed_5b(cond_emb_type_list)
        self.mixed_5b = self.get_encode_mixed_5b(emb_type_list)
        # 10x block35 (Inception-ResNet-A block):
        self.cond_block_35 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 20, scale=0.17,
                                     block_type="block35", norm=norm, act=act, include_cbam=include_cbam,
                                     emb_dim=emb_dim, attn_info=self.get_attn_info(cond_emb_type_list, attn_info_list[3]),
                                     use_checkpoint=use_checkpoint)
            for _ in range(block_depth_list[0])
        ])
        self.block_35 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 20, scale=0.17,
                                     block_type="block35", norm=norm, act=act, include_cbam=include_cbam,
                                     emb_dim=emb_dim, attn_info=self.get_attn_info(emb_type_list, attn_info_list[3]),
                                     use_checkpoint=use_checkpoint)
            for _ in range(block_depth_list[0])
        ])
        # Mixed 6a (Reduction-A block)
        self.cond_mixed_6a = self.get_encode_mixed_6a(cond_emb_type_list)
        self.mixed_6a = self.get_encode_mixed_6a(emb_type_list)
        # 20x block17 (Inception-ResNet-B block)
        self.cond_block_17 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 68, scale=0.1,
                                     block_type="block17", norm=norm, act=act, include_cbam=include_cbam,
                                     emb_dim=emb_dim, attn_info=self.get_attn_info(cond_emb_type_list, attn_info_list[3]),
                                     use_checkpoint=use_checkpoint)
            for _ in range(block_depth_list[1])
        ])
        self.block_17 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 68, scale=0.1,
                                     block_type="block17", norm=norm, act=act, include_cbam=include_cbam,
                                     emb_dim=emb_dim, attn_info=self.get_attn_info(emb_type_list, attn_info_list[3]),
                                     use_checkpoint=use_checkpoint)
            for _ in range(block_depth_list[1])
        ])
        # Mixed 7a (Reduction-B block)
        self.cond_mixed_7a = self.get_encode_mixed_7a(cond_emb_type_list)
        self.mixed_7a = self.get_encode_mixed_7a(emb_type_list)
        # 10x block8 (Inception-ResNet-C block)
        self.cond_block_8 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 130, scale=0.2,
                                     block_type="block8", norm=norm, act=act, include_cbam=include_cbam,
                                     emb_dim=emb_dim, attn_info=self.get_attn_info(cond_emb_type_list, attn_info_list[4]),
                                     use_checkpoint=use_checkpoint)
            for _ in range(block_depth_list[2])
        ])
        self.block_8 = nn.Sequential(*[
            Inception_Resnet_Block2D(in_channels=block_size * 130, scale=0.2,
                                     block_type="block8", norm=norm, act=act, include_cbam=include_cbam,
                                     emb_dim=emb_dim, attn_info=self.get_attn_info(cond_emb_type_list, attn_info_list[4]),
                                     use_checkpoint=use_checkpoint)
            for _ in range(block_depth_list[2])
        ])

        self.cond_final_conv = ConvBlock2D(block_size * 130, block_size * 96 * last_channel_ratio, 1,
                                           norm=norm, act=last_act,
                                           emb_dim=emb_dim, attn_info=self.get_attn_info(cond_emb_type_list, attn_info_list[4]),
                                           use_checkpoint=use_checkpoint)
        # Final convolution block
        self.final_conv = ConvBlock2D(block_size * 130, block_size * 96 * last_channel_ratio, 1,
                                      norm=norm, act=last_act,
                                      emb_dim=emb_dim, attn_info=self.get_attn_info(cond_emb_type_list, attn_info_list[4]),
                                      use_checkpoint=use_checkpoint)

    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None):
        time_emb = self.time_mlp(time)
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        class_emb = self.class_mlp(class_labels)
        if class_emb.ndim == 3:
            class_emb = class_emb.mean(1)
        class_emb = class_emb.to(dtype=x.dtype)        
        
        encode_feature, skip_connect_list = self.encode_forward(x, cond, time_emb, class_emb)
        
        return encode_feature, skip_connect_list
    
    def encode_forward(self, x, cond, time_emb, class_emb):
        # skip connection name list
        # ["stem_layer_1", "stem_layer_4", "stem_layer_7", "mixed_6a", "mixed_7a"]
        skip_connect_list = []
        cond_stem = cond
        stem = x

        for idx, ((_, layer), (_, cond_layer)) in enumerate(zip(self.stem.items(), self.cond_stem.items())):
            cond_stem = cond_layer(cond_stem, time_emb, class_emb)
            stem = layer(stem, time_emb, class_emb, cond_stem)
            if idx in [0, 3, 6]:
                skip_connect_list.append([stem, cond_stem])
        # mixed_5b
        cond_mixed_5b = self.cond_mixed_5b(cond_stem, time_emb, class_emb)
        mixed_5b = self.mixed_5b(stem, time_emb, class_emb, cond_stem)
        
        # block_35
        cond_block_35 = self.cond_block_35(cond_mixed_5b, time_emb, class_emb)
        block_35 = self.block_35(mixed_5b, time_emb, class_emb, cond_block_35)
        # mixed_6a: skip connect target
        cond_mixed_6a = self.cond_mixed_6a(cond_block_35, time_emb, class_emb)
        mixed_6a = self.mixed_6a(block_35, time_emb, class_emb, cond_mixed_6a)
        skip_connect_list.append([mixed_6a, cond_mixed_6a])
        # block_17
        cond_block_17 = self.cond_block_17(cond_mixed_6a, time_emb, class_emb)
        block_17 = self.block_17(mixed_6a, time_emb, class_emb, cond_block_17)
        # mixed_7a: skip connect target
        cond_mixed_7a = self.mixed_7a(cond_block_17, time_emb, class_emb)
        mixed_7a = self.mixed_7a(block_17, time_emb, class_emb, cond_mixed_7a)
        skip_connect_list.append([mixed_7a, cond_mixed_7a])
        # block_8
        cond_block_8 = self.cond_block_8(cond_mixed_7a, time_emb, class_emb)
        block_8 = self.block_8(mixed_7a, time_emb, class_emb, cond_block_8)
        
        # final_output
        cond_output = self.cond_final_conv(cond_block_8, time_emb, class_emb) 
        output = self.final_conv(block_8, time_emb, class_emb, cond_output)

        return output, skip_connect_list

    def get_attn_info(self, emb_type_list, attn_info):
        num_heads = 8
        if attn_info is None:
            return None
        elif attn_info is False:
            return {"emb_type_list":emb_type_list, "num_heads": num_heads, "full_attn": False}
        elif attn_info is True:
            return {"emb_type_list":emb_type_list, "num_heads": num_heads, "full_attn": True}
        
    def get_encode_stem(self, emb_type_list):
        padding_3x3 = self.padding_3x3
        in_channel = self.in_channel
        block_size = self.block_size
        emb_dim = self.emb_dim
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list
        return nn.ModuleDict({
            'stem_layer_1': ConvBlock2D(in_channel, block_size * 2, 3,
                                        stride=2, padding=padding_3x3, norm="instance", act=act, 
                                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[0]),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_2': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                        padding=padding_3x3, norm=norm, act=act,
                                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[1]),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_3': ConvBlock2D(block_size * 2, block_size * 4, 3, norm=norm, act=act,
                                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[1]),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_4': MaxPool2d(3, stride=2, padding=padding_3x3),
            'stem_layer_5': ConvBlock2D(block_size * 4, block_size * 5, 1, norm=norm, act=act,
                                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[2]),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_6': ConvBlock2D(block_size * 5, block_size * 12, 3,
                                        padding=padding_3x3, norm=norm, act=act,
                                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[2]),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_7': MaxPool2d(3, stride=2, padding=padding_3x3)
        })
    
    def get_encode_mixed_5b(self, emb_type_list):
        block_size = self.block_size
        emb_dim = self.emb_dim
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list

        mixed_5b_branch_0 = ConvBlock2D(block_size * 12, block_size * 6, 1,
                                        norm=norm, act=act,
                                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                                        use_checkpoint=use_checkpoint)
        mixed_5b_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 12, block_size *
                        3, 1, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 3, block_size * 4, 5, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint)
        )
        mixed_5b_branch_2 = nn.Sequential(
            ConvBlock2D(block_size * 12, block_size *4, 1, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 4, block_size * 6, 3, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 6, block_size * 6, 3, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint)
        )
        mixed_5b_branch_pool = nn.Sequential(
            AvgPool2d(3, stride=1, padding=1),
            ConvBlock2D(block_size * 12, block_size * 4, 1, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint)
        )
        mixed_5b_branches = [mixed_5b_branch_0, mixed_5b_branch_1,
                             mixed_5b_branch_2, mixed_5b_branch_pool]
        mixed_5b = ConcatBlock(mixed_5b_branches)
        return mixed_5b
    
    def get_encode_mixed_6a(self, emb_type_list):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        emb_dim = self.emb_dim
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list

        mixed_6a_branch_0 = ConvBlock2D(block_size * 20, block_size * 24, 3,
                                        stride=2, padding=padding_3x3, norm=norm, act=act,
                                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                                        use_checkpoint=use_checkpoint)
        mixed_6a_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 20, block_size * 16, 1, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 16, block_size * 16, 3, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 16, block_size * 24, 3,
                        stride=2, padding=padding_3x3, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[3]),
                        use_checkpoint=use_checkpoint)
        )
        mixed_6a_branch_pool = MaxPool2d(3, stride=2,
                                            padding=padding_3x3)
        mixed_6a_branches = [mixed_6a_branch_0, mixed_6a_branch_1,
                             mixed_6a_branch_pool]
        mixed_6a = ConcatBlock(mixed_6a_branches)
        return mixed_6a
    
    def get_encode_mixed_7a(self, emb_type_list):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        emb_dim = self.emb_dim
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list

        mixed_7a_branch_0 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[4]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 16, block_size * 24, 3, stride=2, padding=padding_3x3,
                        norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[4]),
                        use_checkpoint=use_checkpoint)
        )
        mixed_7a_branch_1 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[4]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 16, block_size * 18, 3, stride=2, padding=padding_3x3,
                        norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[4]),
                        use_checkpoint=use_checkpoint)
        )
        mixed_7a_branch_2 = nn.Sequential(
            ConvBlock2D(block_size * 68, block_size * 16, 1, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[4]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 16, block_size * 18, 3, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[4]),
                        use_checkpoint=use_checkpoint),
            ConvBlock2D(block_size * 18, block_size * 20, 3, stride=2,
                        padding=padding_3x3, norm=norm, act=act,
                        emb_dim=emb_dim, attn_info=get_attn_info(emb_type_list, attn_info_list[4]),
                        use_checkpoint=use_checkpoint)
        )
        mixed_7a_branch_pool = MaxPool2d(3, stride=2,
                                         padding=padding_3x3)
        mixed_7a_branches = [mixed_7a_branch_0, mixed_7a_branch_1,
                             mixed_7a_branch_2, mixed_7a_branch_pool]
        mixed_7a = ConcatBlock(mixed_7a_branches)
        return mixed_7a