from torch import nn
import numpy as np
import torch
from .diffusion_layer import default
from .diffusion_layer import ConvBlock2D, MultiDecoder2D_V2, Output2D
from .diffusion_layer import SinusoidalPosEmb, MaxPool2d, AvgPool2d, MultiInputSequential
from ..common_module.layers import get_act

def get_skip_connect_channel_list(block_size, mini=False):
    if mini:
        return np.array([block_size * 2, block_size * 4, block_size * 12])
    else:
        return np.array([block_size * 2, block_size * 2, block_size * 4, block_size * 12,
                        block_size * 68])


class InceptionResNetV2_UNet(nn.Module):
    def __init__(self, in_channel, cond_channel, out_channel, img_size, block_size=16, decode_init_channel=None,
                 norm="group", act="silu", last_act=None, num_class_embeds=None,
                 last_channel_ratio=1, self_condition=False,
                 use_checkpoint=False, attn_info_list=[None, False, False, False, True],
                 block_depth_info="mini"):
        super().__init__()
        
        if decode_init_channel is None:
            decode_init_channel = block_size * 96

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
        if num_class_embeds is not None:
            assert isinstance(num_class_embeds, int) and num_class_embeds > 0, "you need to assign positive int to num_class_embeds"
        self.num_class_embeds = num_class_embeds

        # for compability with Medsegdiff
        self.image_size = img_size
        self.input_img_channels = cond_channel
        self.mask_channels = in_channel
        self.self_condition = self_condition
        if self.self_condition:
            in_channel *= 2
        ##################################
        emb_dim = block_size * 16
        emb_dim_list = [emb_dim, emb_dim]
        cond_emb_type_list = ["seq"]
        emb_type_list = ["seq", "2d"]

        final_conv_emb_dim_list = emb_dim_list + [block_size * 96 * last_channel_ratio]
        ##################################
        self.padding_3x3 = 1
        self.in_channel = in_channel
        self.block_size = block_size
        self.emb_dim_list = emb_dim_list
        self.norm = norm
        self.act = act
        self.use_checkpoint = use_checkpoint
        self.attn_info_list = attn_info_list
        
        self.block_scale_list = [0.17, 0.1, 0.2]
        self.act_layer = get_act(act)
        ##################################
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim // 2),
            nn.Linear(emb_dim // 2, emb_dim),
            get_act(act),
            nn.Linear(emb_dim, emb_dim)
        )
        if self.num_class_embeds is not None:
            self.class_mlp = nn.Embedding(num_class_embeds, emb_dim)
            cond_emb_type_list.append("seq")
            emb_type_list.append("seq")
        else:
            self.class_mlp = None
        # Stem block
        self.cond_stem = self.get_encode_stem(emb_dim_list, cond_emb_type_list, False)
        self.stem = self.get_encode_stem(emb_dim_list, emb_type_list, True)
        # Mixed 5b (Inception-A block):
        self.cond_mixed_5b = self.get_encode_mixed_5b(emb_dim_list, cond_emb_type_list, False)
        self.mixed_5b = self.get_encode_mixed_5b(emb_dim_list, emb_type_list, True)
        # 10x block35 (Inception-ResNet-A block):
        self.cond_block_35_mixed, self.cond_block_35_up = self.get_inception_block(emb_dim_list, emb_type_list, False,
                                                                                   block_depth_list[0],
                                                                                   self.get_inception_block_35)
        self.block_35_mixed, self.block_35_up = self.get_inception_block(emb_dim_list, emb_type_list, True,
                                                                         block_depth_list[0],
                                                                         self.get_inception_block_35)
        # Mixed 6a (Reduction-A block)
        self.cond_mixed_6a = self.get_encode_mixed_6a(emb_dim_list, cond_emb_type_list, False)
        self.mixed_6a = self.get_encode_mixed_6a(emb_dim_list, emb_type_list, True)
        # 20x block17 (Inception-ResNet-B block)
        self.cond_block_17_mixed, self.cond_block_17_up = self.get_inception_block(emb_dim_list, emb_type_list, False,
                                                                                   block_depth_list[1],
                                                                                   self.get_inception_block_17)
        self.block_17_mixed, self.block_17_up = self.get_inception_block(emb_dim_list, emb_type_list, True,
                                                                         block_depth_list[1],
                                                                         self.get_inception_block_17)
        # Mixed 7a (Reduction-B block)
        self.cond_mixed_7a = self.get_encode_mixed_7a(emb_dim_list, cond_emb_type_list, False)
        self.mixed_7a = self.get_encode_mixed_7a(emb_dim_list, emb_type_list, True)
        # 10x block8 (Inception-ResNet-C block)
        self.cond_block_8_mixed, self.cond_block_8_up = self.get_inception_block(emb_dim_list, emb_type_list, False,
                                                                                   block_depth_list[2],
                                                                                   self.get_inception_block_8)
        self.block_8_mixed, self.block_8_up = self.get_inception_block(emb_dim_list, emb_type_list, True,
                                                                         block_depth_list[2],
                                                                         self.get_inception_block_8)

        # Final convolution block
        self.cond_final_conv = ConvBlock2D(block_size * 130, block_size * 96 * last_channel_ratio, 3,
                                           norm=norm, act=act,
                                           emb_dim_list=emb_dim_list, attn_info=self.get_attn_info(cond_emb_type_list, attn_info_list[4]),
                                           use_checkpoint=use_checkpoint)
        self.final_conv = ConvBlock2D(block_size * 130, block_size * 96 * last_channel_ratio, 3,
                                      norm=norm, act=act,
                                      emb_dim_list=final_conv_emb_dim_list, attn_info=self.get_attn_info(cond_emb_type_list, attn_info_list[4]),
                                      use_checkpoint=use_checkpoint)
        
        cond_decoder_layer_list = []
        decoder_layer_list = []
        skip_channel_list = get_skip_connect_channel_list(block_size)
        for decode_idx in range(0, 5):
            skip_channel = skip_channel_list[4 - decode_idx]
            skip_conv_out_channel = decode_init_channel // (2 ** decode_idx)
            decode_in_channel = skip_conv_out_channel
            decode_out_channel = decode_in_channel // 2

            skip_conv_in_channel = skip_channel
            if decode_idx == 0:
                skip_conv_in_channel += block_size * 96
            else:
                skip_conv_in_channel += decode_init_channel // (2 ** decode_idx)


            cond_attn_info = self.get_attn_info(cond_emb_type_list, attn_info_list[4 - decode_idx])
            attn_info = self.get_attn_info(emb_type_list, attn_info_list[4 - decode_idx])
            decode_emb_dim_list = emb_dim_list + [decode_out_channel]

            conv_decoder_layer = MultiDecoder2D_V2(decode_in_channel, skip_channel, decode_out_channel,
                                                    norm=norm, act=act, kernel_size=2,
                                                    emb_dim_list=emb_dim_list, attn_info=cond_attn_info,
                                                    use_checkpoint=use_checkpoint)
            decoder_layer = MultiDecoder2D_V2(decode_in_channel, skip_channel, decode_out_channel,
                                                norm=norm, act=act, kernel_size=2,
                                                emb_dim_list=decode_emb_dim_list, attn_info=attn_info,
                                                use_checkpoint=use_checkpoint)
            cond_decoder_layer_list.append(conv_decoder_layer)
            decoder_layer_list.append(decoder_layer)
        self.cond_decoder_layer_list = nn.ModuleList(cond_decoder_layer_list)
        self.decoder_layer_list = nn.ModuleList(decoder_layer_list)

        self.decode_final_conv = Output2D(decode_out_channel, out_channel, act=last_act)

    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None):
        time_emb = self.time_mlp(time)
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        if self.num_class_embeds is not None:
            class_emb = self.class_mlp(class_labels)
            if class_emb.ndim == 3:
                class_emb = class_emb.mean(1)
            class_emb = class_emb.to(dtype=x.dtype)
        else:
            class_emb = None
        cond_encode_feature, encode_feature, skip_connect_list = self.encode_forward(x, cond, time_emb, class_emb)

        decode_feature = self.decode_forward(cond_encode_feature, encode_feature, skip_connect_list,
                                             time_emb, class_emb)
        return decode_feature
    
    def encode_forward(self, x, cond, time_emb, class_emb):
        # skip connection name list
        # ["stem_layer_1", "stem_layer_4", "stem_layer_7", "mixed_6a", "mixed_7a"]
        skip_connect_list = []
        cond_stem = cond
        stem = x
        for idx, ((_, cond_layer), (_, layer)) in enumerate(zip(self.cond_stem.items(), self.stem.items())):
            cond_stem = cond_layer(cond_stem, time_emb, class_emb)
            stem = layer(stem, time_emb, class_emb, cond_stem)
            if idx in [1, 3, 6, 9]:
                skip_connect_list.append([cond_stem, stem])
        # mixed_5b
        cond_mixed_5b, mixed_5b = self.process_encode_block(self.cond_mixed_5b, self.mixed_5b, 
                                                            cond_stem, stem, time_emb, class_emb)
        # block_35
        cond_block_35, block_35 = self.process_inception_block(self.cond_block_35_mixed, self.cond_block_35_up,
                                                               self.block_35_mixed, self.block_35_up,
                                                               cond_mixed_5b, mixed_5b, time_emb, class_emb,
                                                               self.block_scale_list[0])
        # mixed_6a: skip connect target
        cond_mixed_6a, mixed_6a = self.process_encode_block(self.cond_mixed_6a, self.mixed_6a,
                                                            cond_block_35, block_35, time_emb, class_emb)
        skip_connect_list.append([cond_mixed_6a, mixed_6a])
        # block_17
        cond_block_17, block_17 = self.process_inception_block(self.cond_block_17_mixed, self.cond_block_17_up,
                                                               self.block_17_mixed, self.block_17_up,
                                                               cond_mixed_6a, mixed_6a, time_emb, class_emb,
                                                               self.block_scale_list[1])
        # mixed_7a: skip connect target
        cond_mixed_7a, mixed_7a = self.process_encode_block(self.cond_mixed_7a, self.mixed_7a,
                                                            cond_block_17, block_17, time_emb, class_emb)
        # skip_connect_list.append([cond_mixed_7a, mixed_7a])
        # block_8
        cond_block_8, block_8 = self.process_inception_block(self.cond_block_8_mixed, self.cond_block_8_up,
                                                               self.block_8_mixed, self.block_8_up,
                                                               cond_mixed_7a, mixed_7a, time_emb, class_emb,
                                                               self.block_scale_list[2])
        # final_output
        cond_output = self.cond_final_conv(cond_block_8, time_emb, class_emb)
        output = self.final_conv(block_8, time_emb, class_emb, cond_output)

        return cond_output, output, skip_connect_list[::-1]
    
    def process_encode_block(self, cond_block, block, cond_x, x, time_emb, class_emb):
        cond_output = []
        output = []
        for (_, cond_layer_list), (_, layer_list) in zip(cond_block.items(), block.items()):
            cond_output_part = cond_x
            output_part = x
            for cond_layer, layer in zip(cond_layer_list, layer_list):
                cond_output_part = cond_layer(cond_output_part, time_emb, class_emb)
                output_part = layer(output_part, time_emb, class_emb, cond_output_part)
            cond_output.append(cond_output_part)
            output.append(output_part)
        cond_output = torch.cat(cond_output, dim=1)
        output = torch.cat(output, dim=1)
        return cond_output, output

    def process_inception_block(self, cond_block_mixed, cond_block_up, block_mixed, block_up,
                                cond_x, x, time_emb, class_emb, scale):
        for cond_mixed, cond_up_layer, mixed, up_layer in zip(cond_block_mixed, cond_block_up,
                                                              block_mixed, block_up):
            
            cond_x_temp, x_temp = self.process_encode_block(cond_mixed, mixed,
                                                            cond_x, x, time_emb, class_emb)
            cond_x_temp = cond_up_layer(cond_x_temp, time_emb, class_emb)
            x_temp = up_layer(x_temp, time_emb, class_emb, cond_x_temp)

            cond_x = cond_x + cond_x_temp * scale
            cond_x = self.act_layer(cond_x)
            x = x + x_temp * scale
            x = self.act_layer(x)
        return cond_x, x
        
    def decode_forward(self, cond_encode_feature, encode_feature, skip_connect_list, time_emb, class_emb):
        cond_decode_feature = cond_encode_feature
        decode_feature = encode_feature
        for item in enumerate(zip(self.cond_decoder_layer_list, self.decoder_layer_list)):
            decode_idx, (cond_layer, layer) = item
            cond_skip, skip = skip_connect_list[decode_idx]
            cond_decode_feature = cond_layer(cond_decode_feature, cond_skip, time_emb, class_emb)
            decode_feature = layer(decode_feature, skip, time_emb, class_emb, cond_decode_feature)
        decode_feature = self.decode_final_conv(decode_feature)
        return decode_feature

    def get_attn_info(self, emb_type_list, attn_info):
        num_heads = 8
        if attn_info is None:
            return None
        elif attn_info is False:
            return {"emb_type_list":emb_type_list, "num_heads": num_heads, "full_attn": False}
        elif attn_info is True:
            return {"emb_type_list":emb_type_list, "num_heads": num_heads, "full_attn": True}
        
    def get_encode_stem(self, emb_dim_list, emb_type_list, has_cond):
        if has_cond:
            in_channel = self.in_channel
        else:
            in_channel = self.input_img_channels
        block_size = self.block_size
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list

        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "use_checkpoint": self.use_checkpoint,
        }
        if has_cond:
            emb_dim_list_set = [emb_dim_list + [block_size * 2],
                                emb_dim_list + [block_size * 2],
                                emb_dim_list + [block_size * 4],
                                emb_dim_list + [block_size * 5],
                                emb_dim_list + [block_size * 12]]
        else: 
            emb_dim_list_set = [emb_dim_list for _ in range(5)]
        return nn.ModuleDict({
            'stem_layer_0_0': ConvBlock2D(in_channel, block_size * 2, 3, stride=1,
                                        emb_dim_list=emb_dim_list_set[0],
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0]),
                                        **common_kwarg_dict),
            'stem_layer_0_1': ConvBlock2D(block_size * 2, block_size * 2, 3, stride=1,
                                        emb_dim_list=emb_dim_list_set[0],
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0]),
                                        **common_kwarg_dict),
            'stem_layer_1_0': ConvBlock2D(block_size * 2, block_size * 2, 3, stride=1,
                                        emb_dim_list=emb_dim_list_set[0],
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0]),
                                        **common_kwarg_dict),
            'stem_layer_1_1': ConvBlock2D(block_size * 2, block_size * 2, 3, stride=2, padding=self.padding_3x3,
                                        emb_dim_list=emb_dim_list_set[0],
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0]),
                                        **common_kwarg_dict),
            'stem_layer_2': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                        emb_dim_list=emb_dim_list_set[1],
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[1]),
                                        **common_kwarg_dict),
            'stem_layer_3': ConvBlock2D(block_size * 2, block_size * 4, 3,
                                        emb_dim_list=emb_dim_list_set[2],
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[1]),
                                        **common_kwarg_dict),
            'stem_layer_4': MaxPool2d(3, stride=2, padding=self.padding_3x3),
            'stem_layer_5': ConvBlock2D(block_size * 4, block_size * 5, 1,
                                        emb_dim_list=emb_dim_list_set[3],
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[2]),
                                        **common_kwarg_dict),
            'stem_layer_6': ConvBlock2D(block_size * 5, block_size * 12, 3,
                                        emb_dim_list=emb_dim_list_set[4],
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[2]),
                                        **common_kwarg_dict),
            'stem_layer_7': MaxPool2d(3, stride=2, padding=self.padding_3x3)
        })
    
    def get_encode_mixed_5b(self, emb_dim_list, emb_type_list, has_cond):
        block_size = self.block_size
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[3])
        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "attn_info": attn_info,
            "use_checkpoint": self.use_checkpoint,
        }
        if has_cond:
            emb_dim_list_set = [emb_dim_list + [block_size * 6],
                                emb_dim_list + [block_size * 3],
                                emb_dim_list + [block_size * 4],
                                emb_dim_list + [block_size * 4],
                                emb_dim_list + [block_size * 6],
                                emb_dim_list + [block_size * 6],
                                emb_dim_list + [block_size * 4]]
        else: 
            emb_dim_list_set = [emb_dim_list for _ in range(7)]
        mixed_5b_branch_0_0 = ConvBlock2D(block_size * 12, block_size * 6, 1,
                                        emb_dim_list=emb_dim_list_set[0], **common_kwarg_dict)
        mixed_5b_branch_1_0 = ConvBlock2D(block_size * 12, block_size * 3, 1,
                                        emb_dim_list=emb_dim_list_set[1], **common_kwarg_dict)
        mixed_5b_branch_1_1 = ConvBlock2D(block_size * 3, block_size * 4, 5,
                                        emb_dim_list=emb_dim_list_set[2], **common_kwarg_dict)
        mixed_5b_branch_2_0 = ConvBlock2D(block_size * 12, block_size * 4, 1,
                                        emb_dim_list=emb_dim_list_set[3], **common_kwarg_dict)
        mixed_5b_branch_2_1 = ConvBlock2D(block_size * 4, block_size * 6, 3,
                                        emb_dim_list=emb_dim_list_set[4], **common_kwarg_dict)
        mixed_5b_branch_2_2 = ConvBlock2D(block_size * 6, block_size * 6, 3,
                                        emb_dim_list=emb_dim_list_set[5], **common_kwarg_dict)
        mixed_5b_branch_pool_0 = AvgPool2d(3, stride=1, padding=1)
        mixed_5b_branch_pool_1 = ConvBlock2D(block_size * 12, block_size * 4, 1,
                                            emb_dim_list=emb_dim_list_set[6], **common_kwarg_dict)
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
    
    def get_encode_mixed_6a(self, emb_dim_list, emb_type_list, has_cond):
        block_size = self.block_size
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[3])
        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "attn_info": attn_info,
            "use_checkpoint": self.use_checkpoint,
        }
        if has_cond:
            emb_dim_list_set = [emb_dim_list + [block_size * 24],
                                emb_dim_list + [block_size * 16],
                                emb_dim_list + [block_size * 16],
                                emb_dim_list + [block_size * 24]]
        else:
            emb_dim_list_set = [emb_dim_list for _ in range(4)]

        mixed_6a_branch_0_0 = ConvBlock2D(block_size * 20, block_size * 24, 3, stride=2, padding=self.padding_3x3,
                                        emb_dim_list=emb_dim_list_set[0], **common_kwarg_dict)
        mixed_6a_branch_1_1 = ConvBlock2D(block_size * 20, block_size * 16, 1,
                                        emb_dim_list=emb_dim_list_set[1], **common_kwarg_dict)
        mixed_6a_branch_1_2 = ConvBlock2D(block_size * 16, block_size * 16, 3,
                                        emb_dim_list=emb_dim_list_set[2], **common_kwarg_dict)
        mixed_6a_branch_1_3 = ConvBlock2D(block_size * 16, block_size * 24, 3, stride=2, padding=self.padding_3x3,
                                        emb_dim_list=emb_dim_list_set[3], **common_kwarg_dict)
        mixed_6a_branch_pool_0 = MaxPool2d(3, stride=2, padding=self.padding_3x3)
        mixed_6a = nn.ModuleDict({
            "mixed_6a_branch_0": nn.ModuleList([mixed_6a_branch_0_0]),
            "mixed_6a_branch_1": nn.ModuleList([mixed_6a_branch_1_1,
                                                mixed_6a_branch_1_2,
                                                mixed_6a_branch_1_3]),
            "mixed_6a_branch_pool": nn.ModuleList([mixed_6a_branch_pool_0])
        })
        return mixed_6a
    
    def get_encode_mixed_7a(self, emb_dim_list, emb_type_list, has_cond):
        block_size = self.block_size
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[4])
        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "attn_info": attn_info,
            "use_checkpoint": self.use_checkpoint,
        }
        if has_cond:
            emb_dim_list_set = [emb_dim_list + [block_size * 16],
                                emb_dim_list + [block_size * 24],
                                emb_dim_list + [block_size * 16],
                                emb_dim_list + [block_size * 18],
                                emb_dim_list + [block_size * 16],
                                emb_dim_list + [block_size * 18],
                                emb_dim_list + [block_size * 20]]
        else: 
            emb_dim_list_set = [emb_dim_list for _ in range(7)]

        mixed_7a_branch_0_1 = ConvBlock2D(block_size * 68, block_size * 16, 1,
                                        emb_dim_list=emb_dim_list_set[0], **common_kwarg_dict)
        mixed_7a_branch_0_2 = ConvBlock2D(block_size * 16, block_size * 24, 3, stride=2, padding=self.padding_3x3,
                                        emb_dim_list=emb_dim_list_set[1], **common_kwarg_dict)
        mixed_7a_branch_1_1 = ConvBlock2D(block_size * 68, block_size * 16, 1,
                                        emb_dim_list=emb_dim_list_set[2], **common_kwarg_dict)
        mixed_7a_branch_1_2 = ConvBlock2D(block_size * 16, block_size * 18, 3, stride=2, padding=self.padding_3x3,
                                        emb_dim_list=emb_dim_list_set[3], **common_kwarg_dict)
        mixed_7a_branch_2_1 = ConvBlock2D(block_size * 68, block_size * 16, 1,
                                        emb_dim_list=emb_dim_list_set[4], **common_kwarg_dict)
        mixed_7a_branch_2_2 = ConvBlock2D(block_size * 16, block_size * 18, 3,
                                        emb_dim_list=emb_dim_list_set[5], **common_kwarg_dict)
        mixed_7a_branch_2_3 = ConvBlock2D(block_size * 18, block_size * 20, 3, stride=2, padding=self.padding_3x3,
                                        emb_dim_list=emb_dim_list_set[6], **common_kwarg_dict)
        mixed_7a_branch_pool_0 = MaxPool2d(3, stride=2, padding=self.padding_3x3)
        
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
    
    def get_inception_block(self, emb_dim_list, emb_type_list, has_cond, block_depth, get_block_fn):
        block_list = [get_block_fn(emb_dim_list, emb_type_list, has_cond)
                      for _ in range(block_depth)]
        mixed_list, up_list = [list(item) for item in zip(*block_list)]
        return nn.ModuleList(mixed_list), nn.ModuleList(up_list)

    def get_inception_block_35(self, emb_dim_list, emb_type_list, has_cond):
        block_size = self.block_size
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[3])
        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "attn_info": attn_info,
            "use_checkpoint": self.use_checkpoint,
        }
        in_channels = block_size * 20
        mixed_channel = block_size * 8
        if has_cond:
            emb_dim_list_set = [emb_dim_list + [block_size * 2],
                                emb_dim_list + [block_size * 2],
                                emb_dim_list + [block_size * 2],
                                emb_dim_list + [block_size * 2],
                                emb_dim_list + [block_size * 3],
                                emb_dim_list + [block_size * 4],
                                emb_dim_list + [in_channels]]
        else: 
            emb_dim_list_set = [emb_dim_list for _ in range(7)]

        branch_0_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                emb_dim_list=emb_dim_list_set[0], **common_kwarg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                emb_dim_list=emb_dim_list_set[1], **common_kwarg_dict)
        branch_1_1 = ConvBlock2D(block_size * 2, block_size * 2, 3,
                                emb_dim_list=emb_dim_list_set[2], **common_kwarg_dict)
        branch_2_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                emb_dim_list=emb_dim_list_set[3], **common_kwarg_dict)
        branch_2_1 = ConvBlock2D(block_size * 2, block_size * 3, 3,
                                emb_dim_list=emb_dim_list_set[4], **common_kwarg_dict)
        branch_2_2 = ConvBlock2D(block_size * 3, block_size * 4, 3,
                                emb_dim_list=emb_dim_list_set[5], **common_kwarg_dict)
        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=emb_dim_list_set[6], attn_info=attn_info,
                        use_checkpoint=self.use_checkpoint)
        
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                       branch_1_1]),
            "branch_2_1": nn.ModuleList([branch_2_0,
                                         branch_2_1,
                                         branch_2_2])
        })
        return mixed, up

    def get_inception_block_17(self, emb_dim_list, emb_type_list, has_cond):
        block_size = self.block_size
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[3])
        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "attn_info": attn_info,
            "use_checkpoint": self.use_checkpoint,
        }
        in_channels = block_size * 68
        mixed_channel = block_size * 24
        if has_cond:
            emb_dim_list_set = [emb_dim_list + [block_size * 12],
                                emb_dim_list + [block_size * 8],
                                emb_dim_list + [block_size * 10],
                                emb_dim_list + [block_size * 12],
                                emb_dim_list + [in_channels]]
        else: 
            emb_dim_list_set = [emb_dim_list for _ in range(5)]

        branch_0_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                emb_dim_list=emb_dim_list_set[0], **common_kwarg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 8, 1,
                                emb_dim_list=emb_dim_list_set[1], **common_kwarg_dict)
        branch_1_1 = ConvBlock2D(block_size * 8, block_size * 10, [1, 7],
                                emb_dim_list=emb_dim_list_set[2], **common_kwarg_dict)
        branch_1_2 = ConvBlock2D(block_size * 10, block_size * 12, [7, 1],
                                emb_dim_list=emb_dim_list_set[3], **common_kwarg_dict)
        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=emb_dim_list_set[4], attn_info=attn_info,
                        use_checkpoint=self.use_checkpoint)
        
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                       branch_1_1,
                                       branch_1_2]),
        })
        return mixed, up

    def get_inception_block_8(self, emb_dim_list, emb_type_list, has_cond):
        block_size = self.block_size
        attn_info = self.get_attn_info(emb_type_list, self.attn_info_list[4])
        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "attn_info": attn_info,
            "use_checkpoint": self.use_checkpoint,
        }
        in_channels = block_size * 130
        mixed_channel = block_size * 28
        if has_cond:
            emb_dim_list_set = [emb_dim_list + [block_size * 12],
                                emb_dim_list + [block_size * 12],
                                emb_dim_list + [block_size * 14],
                                emb_dim_list + [block_size * 16],
                                emb_dim_list + [in_channels]]
        else: 
            emb_dim_list_set = [emb_dim_list for _ in range(5)]

        branch_0_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                emb_dim_list=emb_dim_list_set[0], **common_kwarg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                emb_dim_list=emb_dim_list_set[1], **common_kwarg_dict)
        branch_1_1 = ConvBlock2D(block_size * 12, block_size * 14, [1, 3],
                                emb_dim_list=emb_dim_list_set[2], **common_kwarg_dict)
        branch_1_2 = ConvBlock2D(block_size * 14, block_size * 16, [3, 1],
                                emb_dim_list=emb_dim_list_set[3], **common_kwarg_dict)
        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=emb_dim_list_set[4], attn_info=attn_info,
                        use_checkpoint=self.use_checkpoint)
        
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                       branch_1_1,
                                       branch_1_2]),
        })
        return mixed, up
