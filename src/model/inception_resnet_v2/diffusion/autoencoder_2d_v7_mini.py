from torch import nn
import numpy as np
import torch
from functools import partial
from .diffusion_layer import default
from .diffusion_layer import WrapGroupNorm
from .diffusion_layer import ConvBlock2D, MultiDecoder2D, Output2D
from .diffusion_layer import SinusoidalPosEmb, MaxPool2d, AvgPool2d

from ..common_module.layers import get_act, AttentionPool

BLOCK_NUM = 4

def get_skip_connect_channel_list(block_size, mini=False):
    if mini:
        return np.array([block_size * 68, block_size * 12, block_size * 4, block_size * 2])
    else:
        return np.array([block_size * 130, block_size * 68, block_size * 12,
                         block_size * 4, block_size * 2])


class InceptionResNetV2_UNet(nn.Module):
    def __init__(self, in_channel, cond_channel, out_channel, img_size, block_size=16,
                 emb_channel=1024, decode_init_channel=None, act="silu", last_act=None, num_class_embeds=None,
                 last_channel_ratio=1, self_condition=False, use_checkpoint=False,
                 num_head_list=[4, 8, 16, 16],
                 block_depth_info="mini"):
        super().__init__()
        
        if decode_init_channel is None:
            decode_init_channel = block_size * 48

        if isinstance(block_depth_info, str):
            if block_depth_info == "tiny":
                block_depth_list = [1, 2]
            elif block_depth_info == "mini":
                block_depth_list = [2, 4]
            elif block_depth_info == "middle":
                block_depth_list = [5, 10]
            elif block_depth_info == "large":
                block_depth_list = [10, 20]
        elif isinstance(block_depth_info, int):
            block_depth_list = np.array([1, 2]) * block_depth_info
        else:
            block_depth_list = block_depth_info

        # for compability with Medsegdiff
        self.image_size = img_size
        self.input_img_channels = cond_channel
        self.mask_channels = in_channel
        self.self_condition = self_condition
        if self.self_condition:
            in_channel *= 2

        ##################################
        emb_dim = block_size * 16
        self.padding_3x3 = 1
        self.in_channel = in_channel
        self.block_size = block_size
        self.norm = WrapGroupNorm
        self.act = act
        self.use_checkpoint = use_checkpoint
        self.num_head_list = num_head_list

        self.block_scale_list = [0.17, 0.1]
        self.act_layer = get_act(act)
        ##################################
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim // 2),
            nn.Linear(emb_dim // 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        emb_dim_list = [emb_dim, emb_channel]
        emb_type_list = ["seq", "seq"]
        if num_class_embeds:
            self.class_emb_layer = nn.Embedding(num_class_embeds, emb_dim * 2)
            self.class_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim * 2, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim)
            )
            emb_dim_list += [emb_dim]
            emb_type_list += ["seq"]
        else:
            self.class_emb_layer = None
        self.emb_dim_list = emb_dim_list
        
        self.latent_model = InceptionResNetV2_Encoder(in_channel=cond_channel, img_size=img_size, block_size=block_size, 
                                                      emb_channel=emb_channel, norm=norm, act=act, last_channel_ratio=1,
                                                    use_checkpoint=use_checkpoint, num_head_list=num_head_list,
                                                    block_depth_info=block_depth_info)
        # Stem block
        self.stem = self.get_encode_stem(emb_dim_list, emb_type_list)
        # Mixed 5b (Inception-A block):
        self.mixed_5b = self.get_encode_mixed_5b(emb_dim_list, emb_type_list)
        # 10x block35 (Inception-ResNet-A block):
        self.block_35_mixed, self.block_35_up = self.get_inception_block(emb_dim_list, emb_type_list,
                                                                         block_depth_list[0],
                                                                         self.get_inception_block_35)
        # Mixed 6a (Reduction-A block)
        self.mixed_6a = self.get_encode_mixed_6a(emb_dim_list, emb_type_list)
        # 20x block17 (Inception-ResNet-B block)
        self.block_17_mixed, self.block_17_up = self.get_inception_block(emb_dim_list, emb_type_list,
                                                                         block_depth_list[1],
                                                                         self.get_inception_block_17)
        # Final convolution block
        self.final_conv = ConvBlock2D(block_size * 68, block_size * 48 * last_channel_ratio, 3,
                                      norm=norm, act=act,
                                      emb_dim_list=emb_dim_list, attn_info=self.get_attn_info(emb_type_list, attn_info_list[3]),
                                      use_checkpoint=use_checkpoint)
        
        skip_conv_list = []
        decoder_layer_list = []
        skip_channel_list = get_skip_connect_channel_list(block_size, mini=True)
        for decode_idx, skip_channel in enumerate(skip_channel_list):
            skip_channel = skip_channel_list[decode_idx]
            skip_conv_out_channel = decode_init_channel // (2 ** decode_idx)
            decode_in_channel = skip_conv_out_channel
            decode_out_channel = decode_in_channel // 2

            skip_conv_in_channel = skip_channel
            if decode_idx == 0:
                skip_conv_in_channel += block_size * 48
            else:
                skip_conv_in_channel += decode_init_channel // (2 ** decode_idx)

            attn_info = self.get_attn_info(emb_type_list, attn_info_list[BLOCK_NUM - decode_idx - 1])
            decode_emb_dim_list = emb_dim_list + [decode_out_channel]
            
            skip_conv = nn.Conv2d(skip_conv_in_channel, skip_conv_out_channel, 1, bias=False)

            decoder_layer = MultiDecoder2D(decode_in_channel, decode_out_channel,
                                                norm=norm, act=act, kernel_size=2,
                                                emb_dim_list=decode_emb_dim_list, attn_info=attn_info,
                                                use_checkpoint=use_checkpoint)
            skip_conv_list.append(skip_conv)
            decoder_layer_list.append(decoder_layer)
        self.skip_conv_list = nn.ModuleList(skip_conv_list)
        self.decoder_layer_list = nn.ModuleList(decoder_layer_list)
        self.decode_final_conv = Output2D(decode_out_channel, out_channel, act=last_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        time_emb = self.time_mlp(time)
        latent = self.latent_model(cond)
        if self.class_emb_layer is not None:
            class_emb = self.class_emb_layer(class_labels).to(dtype=x.dtype)
            class_emb = self.class_mlp(class_emb)
        encode_feature, skip_connect_list = self.encode_forward(x, time_emb, latent, class_emb)
        decode_feature = self.decode_forward(encode_feature, skip_connect_list,
                                             time_emb, latent, class_emb)
        return decode_feature
    
    def encode_forward(self, x, *args):
        # skip connection name list
        # ["stem_layer_1", "stem_layer_4", "stem_layer_7", "mixed_6a", "mixed_7a"]
        skip_connect_list = []
        stem = x
        for idx, (_, layer) in enumerate(self.stem.items()):
            stem = layer(stem, *args)
            if idx in [0, 3, 6]:
                skip_connect_list.append(stem)
        # mixed_5b
        mixed_5b = self.process_encode_block(self.mixed_5b, stem, *args)
        # block_35
        block_35 = self.process_inception_block(self.block_35_mixed, self.block_35_up,
                                                mixed_5b, self.block_scale_list[0], *args)
        # mixed_6a: skip connect target
        mixed_6a = self.process_encode_block(self.mixed_6a, block_35, *args)
        skip_connect_list.append(mixed_6a)
        # block_17
        block_17 = self.process_inception_block(self.block_17_mixed, self.block_17_up,
                                                mixed_6a, self.block_scale_list[1], *args)
        # final_output
        output = self.final_conv(block_17, *args)

        return output, skip_connect_list[::-1]
    
    def process_encode_block(self, block, x, *args):
        output = []
        for (_, layer_list) in block.items():
            output_part = x
            for layer in layer_list:
                output_part = layer(output_part, *args)
            output.append(output_part)
        output = torch.cat(output, dim=1)
        return output

    def process_inception_block(self, block_mixed, block_up,
                                x, scale, *args):
        for mixed, up_layer in zip(block_mixed, block_up):
            x_temp = self.process_encode_block(mixed, x, *args)
            x_temp = up_layer(x_temp, *args)

            x = x + x_temp * scale
            x = self.act_layer(x)
        return x
        
    def decode_forward(self, encode_feature, skip_connect_list, *args):
        decode_feature = encode_feature
        for decode_idx, (skip_conv, layer) in enumerate(zip(self.skip_conv_list, self.decoder_layer_list)):
            skip = skip_connect_list[decode_idx]
            decode_feature = torch.cat([decode_feature, skip], dim=1)
            decode_feature = skip_conv(decode_feature)
            decode_feature = layer(decode_feature, *args)
        decode_feature = self.decode_final_conv(decode_feature)
        return decode_feature

    def get_attn_info(self, emb_type_list, attn_info):
        num_heads = 4
        if attn_info is None:
            return None
        elif attn_info is False:
            return {"emb_type_list":emb_type_list, "num_heads": num_heads, "full_attn": False}
        elif attn_info is True:
            return {"emb_type_list":emb_type_list, "num_heads": num_heads, "full_attn": True}
        
    def get_encode_stem(self, emb_dim_list, emb_type_list):
        padding_3x3 = self.padding_3x3
        in_channel = self.in_channel
        block_size = self.block_size
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        get_attn_info = self.get_attn_info

        attn_info_list = self.attn_info_list
        
        return nn.ModuleDict({
            'stem_layer_1': ConvBlock2D(in_channel, block_size * 2, 3,
                                        stride=2, padding=padding_3x3, norm="instance", act=act,
                                        emb_dim_list=emb_dim_list, attn_info=get_attn_info(emb_type_list, attn_info_list[0]),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_2': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                        padding=padding_3x3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=get_attn_info(emb_type_list, None),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_3': ConvBlock2D(block_size * 2, block_size * 4, 3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=get_attn_info(emb_type_list, attn_info_list[1]),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_4': MaxPool2d(3, stride=2, padding=padding_3x3),
            'stem_layer_5': ConvBlock2D(block_size * 4, block_size * 5, 1, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=get_attn_info(emb_type_list, None),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_6': ConvBlock2D(block_size * 5, block_size * 12, 3,
                                        padding=padding_3x3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=get_attn_info(emb_type_list, attn_info_list[2]),
                                        use_checkpoint=use_checkpoint),
            'stem_layer_7': MaxPool2d(3, stride=2, padding=padding_3x3)
        })
    
    def get_encode_mixed_5b(self, emb_dim_list, emb_type_list):
        block_size = self.block_size
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[3])
        non_attn_info = self.get_attn_info(emb_type_list, None)
        
        mixed_5b_branch_0_0 = ConvBlock2D(block_size * 12, block_size * 6, 1,
                                        norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_5b_branch_1_0 = ConvBlock2D(block_size * 12, block_size * 3, 1, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_5b_branch_1_1 = ConvBlock2D(block_size * 3, block_size * 4, 5, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_5b_branch_2_0 = ConvBlock2D(block_size * 12, block_size * 4, 1, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_5b_branch_2_1 = ConvBlock2D(block_size * 4, block_size * 6, 3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_5b_branch_2_2 = ConvBlock2D(block_size * 6, block_size * 6, 3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_5b_branch_pool_0 = AvgPool2d(3, stride=1, padding=1)
        mixed_5b_branch_pool_1 = ConvBlock2D(block_size * 12, block_size * 4, 1, norm=norm, act=act,
                                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                                            use_checkpoint=use_checkpoint)
        mixed_5b = nn.ModuleDict({
            "mixed_5b_branch_0": nn.ModuleList([mixed_5b_branch_0_0]),
            "mixed_5b_branch_1": nn.ModuleList([mixed_5b_branch_1_0, 
                                                mixed_5b_branch_1_1]),
            "mixed_5b_branch_2_0": nn.ModuleList([mixed_5b_branch_2_0,
                                                  mixed_5b_branch_2_1,
                                                  mixed_5b_branch_2_2]),
            "mixed_5b_branch_pool": nn.ModuleList([mixed_5b_branch_pool_0,
                                                   mixed_5b_branch_pool_1]),
        })
        return mixed_5b
    
    def get_encode_mixed_6a(self, emb_dim_list, emb_type_list):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[3])
        non_attn_info = self.get_attn_info(emb_type_list, None)

        mixed_6a_branch_0_0 = ConvBlock2D(block_size * 20, block_size * 24, 3,
                                        stride=2, padding=padding_3x3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_6a_branch_1_1 = ConvBlock2D(block_size * 20, block_size * 16, 1, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_6a_branch_1_2 = ConvBlock2D(block_size * 16, block_size * 16, 3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_6a_branch_1_3 = ConvBlock2D(block_size * 16, block_size * 24, 3,
                                        stride=2, padding=padding_3x3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_6a_branch_pool_0 = MaxPool2d(3, stride=2, padding=padding_3x3)
        mixed_6a = nn.ModuleDict({
            "mixed_6a_branch_0": nn.ModuleList([mixed_6a_branch_0_0]),
            "mixed_6a_branch_1": nn.ModuleList([mixed_6a_branch_1_1,
                                                mixed_6a_branch_1_2,
                                                mixed_6a_branch_1_3]),
            "mixed_6a_branch_pool": nn.ModuleList([mixed_6a_branch_pool_0])
        })
        return mixed_6a
    
    def get_encode_mixed_7a(self, emb_dim_list, emb_type_list):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[4])
        non_attn_info = self.get_attn_info(emb_type_list, None)

        mixed_7a_branch_0_1 = ConvBlock2D(block_size * 68, block_size * 16, 1, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_7a_branch_0_2 = ConvBlock2D(block_size * 16, block_size * 24, 3, stride=2, padding=padding_3x3,
                                        norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_7a_branch_1_1 = ConvBlock2D(block_size * 68, block_size * 16, 1, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_7a_branch_1_2 = ConvBlock2D(block_size * 16, block_size * 18, 3, stride=2, padding=padding_3x3,
                                        norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_7a_branch_2_1 = ConvBlock2D(block_size * 68, block_size * 16, 1, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_7a_branch_2_2 = ConvBlock2D(block_size * 16, block_size * 18, 3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=non_attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_7a_branch_2_3 = ConvBlock2D(block_size * 18, block_size * 20, 3, stride=2,
                                        padding=padding_3x3, norm=norm, act=act,
                                        emb_dim_list=emb_dim_list, attn_info=attn_info,
                                        use_checkpoint=use_checkpoint)
        mixed_7a_branch_pool_0 = MaxPool2d(3, stride=2, padding=padding_3x3)
        
        mixed_7a = nn.ModuleDict({
            "mixed_7a_branch_0": nn.ModuleList([mixed_7a_branch_0_1,
                                                mixed_7a_branch_0_2]),
            "mixed_7a_branch_1": nn.ModuleList([mixed_7a_branch_1_1,
                                                mixed_7a_branch_1_2]),
            "mixed_7a_branch_2_1": nn.ModuleList([mixed_7a_branch_2_1,
                                                  mixed_7a_branch_2_2,
                                                  mixed_7a_branch_2_3]),
            "mixed_7a_branch_pool": nn.ModuleList([mixed_7a_branch_pool_0])
        })
        return mixed_7a
    
    def get_inception_block(self, emb_dim_list, emb_type_list, block_depth, get_block_fn):
        block_list = [get_block_fn(emb_dim_list, emb_type_list)
                      for _ in range(block_depth)]
        mixed_list, up_list = [list(item) for item in zip(*block_list)]
        return nn.ModuleList(mixed_list), nn.ModuleList(up_list)

    def get_inception_block_35(self, emb_dim_list, emb_type_list):
        block_size = self.block_size
        norm = partial(self.norm, num_groups=self.num_head_list[2])
        act = self.act
        use_checkpoint = self.use_checkpoint

        common_kwarg_dict = {
            "norm": norm, "act": act,
            "emb_dim_list": emb_dim_list, "emb_type_list": emb_type_list,
            "use_checkpoint": use_checkpoint
        }

        in_channels = block_size * 20
        mixed_channel = block_size * 8
        
        branch_0_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                **common_kwarg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                **common_kwarg_dict)
        branch_1_1 = ConvBlock2D(block_size * 2, block_size * 2, 3,
                                **common_kwarg_dict)
        branch_2_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                **common_kwarg_dict)
        branch_2_1 = ConvBlock2D(block_size * 2, block_size * 3, 3,
                                **common_kwarg_dict)
        branch_2_2 = ConvBlock2D(block_size * 3, block_size * 4, 3,
                                **common_kwarg_dict)
        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=norm, act=None,
                        emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                        use_checkpoint=use_checkpoint)
        
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                       branch_1_1]),
            "branch_2_1": nn.ModuleList([branch_2_0,
                                         branch_2_1,
                                         branch_2_2])
        })
        return mixed, up

    def get_inception_block_17(self, emb_dim_list, emb_type_list):
        block_size = self.block_size
        norm = partial(self.norm, num_groups=self.num_head_list[2])
        act = self.act
        use_checkpoint = self.use_checkpoint

        common_kwarg_dict = {
            "norm": norm, "act": act,
            "emb_dim_list": emb_dim_list, "emb_type_list": emb_type_list,
            "use_checkpoint": use_checkpoint
        }

        in_channels = block_size * 68
        mixed_channel = block_size * 24

        branch_0_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                **common_kwarg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 8, 1,
                                **common_kwarg_dict)
        branch_1_1 = ConvBlock2D(block_size * 8, block_size * 10, [1, 7],
                                **common_kwarg_dict)
        branch_1_2 = ConvBlock2D(block_size * 10, block_size * 12, [7, 1],
                                **common_kwarg_dict)
        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=norm, act=None,
                        emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                        use_checkpoint=use_checkpoint)
        
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                       branch_1_1,
                                       branch_1_2]),
        })
        return mixed, up
    
    def get_inception_block_8(self, emb_dim_list, emb_type_list):
        block_size = self.block_size
        norm = partial(self.norm, num_groups=self.num_head_list[3])
        act = self.act
        use_checkpoint = self.use_checkpoint

        common_kwarg_dict = {
            "norm": norm, "act": act,
            "emb_dim_list": emb_dim_list, "emb_type_list": emb_type_list,
            "use_checkpoint": use_checkpoint
        }

        in_channels = block_size * 130
        mixed_channel = block_size * 28

        branch_0_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                **common_kwarg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                **common_kwarg_dict)
        branch_1_1 = ConvBlock2D(block_size * 12, block_size * 14, [1, 3],
                                **common_kwarg_dict)
        branch_1_2 = ConvBlock2D(block_size * 14, block_size * 16, [3, 1],
                                **common_kwarg_dict)
        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=norm, act=None,
                        emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                        use_checkpoint=use_checkpoint)
        
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                       branch_1_1,
                                       branch_1_2]),
        })
        return mixed, up
    
class InceptionResNetV2_Encoder(InceptionResNetV2_UNet):
    def __init__(self, in_channel, img_size, block_size=16, emb_channel=1024,
                 norm="group", act="silu", last_channel_ratio=1,
                 use_checkpoint=False, num_head_list=[4, 8, 8, 16],
                 block_depth_info="mini"):
        super(InceptionResNetV2_UNet, self).__init__()

        if isinstance(block_depth_info, str):
            if block_depth_info == "tiny":
                block_depth_list = [1, 2]
            elif block_depth_info == "mini":
                block_depth_list = [2, 4]
            elif block_depth_info == "middle":
                block_depth_list = [5, 10]
            elif block_depth_info == "large":
                block_depth_list = [10, 20]
        elif isinstance(block_depth_info, int):
            block_depth_list = np.array([1, 2]) * block_depth_info
        else:
            block_depth_list = block_depth_info
        
        ##################################
        self.padding_3x3 = 1
        self.in_channel = in_channel
        self.block_size = block_size
        self.norm = norm
        self.act = act
        self.use_checkpoint = use_checkpoint
        self.attn_info_list = attn_info_list
        
        self.block_scale_list = [0.17, 0.1]
        self.act_layer = get_act(act)
        ##################################
        # Stem block
        self.stem = self.get_encode_stem(None, None)
        # Mixed 5b (Inception-A block):
        self.mixed_5b = self.get_encode_mixed_5b(None, None)
        # 10x block35 (Inception-ResNet-A block):
        self.block_35_mixed, self.block_35_up = self.get_inception_block(None, None,
                                                                         block_depth_list[0],
                                                                         self.get_inception_block_35)
        # Mixed 6a (Reduction-A block)
        self.mixed_6a = self.get_encode_mixed_6a(None, None)
        # 20x block17 (Inception-ResNet-B block)
        self.block_17_mixed, self.block_17_up = self.get_inception_block(None, None,
                                                                         block_depth_list[1],
                                                                         self.get_inception_block_17)

        feature_map_size = (img_size // 16, img_size // 16)
        # Final convolution block
        self.final_conv = ConvBlock2D(block_size * 68, block_size * 48 * last_channel_ratio, 3,
                                      norm=norm, act=act,
                                      emb_dim_list=None, attn_info=None,
                                      use_checkpoint=use_checkpoint)
        self.pool_layer = AttentionPool(feature_num=np.prod(feature_map_size),
                                        embed_dim=block_size * 48 * last_channel_ratio,
                                        num_heads=8, output_dim=emb_channel * 2)
        self.out = nn.Sequential(nn.SiLU(), nn.Dropout(0.05), nn.Linear(emb_channel * 2, emb_channel),
                                 nn.SiLU(), nn.Dropout(0.05), nn.Linear(emb_channel, emb_channel))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):

        encode_feature = self.encode_forward(x)

        return encode_feature
    
    def encode_forward(self, x):
        # skip connection name list
        # ["stem_layer_1", "stem_layer_4", "stem_layer_7", "mixed_6a", "mixed_7a"]
        stem = x
        for layer in self.stem.values():
            stem = layer(stem)
        # mixed_5b
        mixed_5b = self.process_encode_block(self.mixed_5b, stem)
        # block_35
        block_35 = self.process_inception_block(self.block_35_mixed, self.block_35_up,
                                                mixed_5b, self.block_scale_list[0])
        # mixed_6a: skip connect target
        mixed_6a = self.process_encode_block(self.mixed_6a, block_35)
        # block_17
        block_17 = self.process_inception_block(self.block_17_mixed, self.block_17_up,
                                                               mixed_6a, self.block_scale_list[1])
        # final_output
        output = self.final_conv(block_17)
        output = self.pool_layer(output)
        output = self.out(output)
        return output