from torch import nn
import numpy as np
import torch
from .diffusion_layer import default
from .diffusion_layer import ConvBlock2D, MultiDecoder2D_V2, Output2D, RMSNorm
from .diffusion_layer import LinearAttention, Attention, MaxPool2d, AvgPool2d, MultiInputSequential
from .diffusion_layer import default, prob_mask_like, LearnedSinusoidalPosEmb, SinusoidalPosEmb

from ..common_module.layers import get_act, AttentionPool
from einops import rearrange, repeat        

def get_skip_connect_channel_list(block_size, mini=False):
    if mini:
        return np.array([block_size * 2, block_size * 4, block_size * 12])
    else:
        return np.array([block_size * 2, block_size * 2, block_size * 4, block_size * 12,
                        block_size * 68])


class InceptionResNetV2_UNet(nn.Module):
    def __init__(self, in_channel, cond_channel, out_channel, img_size, block_size=16,
                 emb_channel=1024, decode_init_channel=None,
                 norm=RMSNorm, act="silu", last_act=None, num_class_embeds=None, drop_prob=0.0, cond_drop_prob=0.5,
                 last_channel_ratio=1, self_condition=False, use_checkpoint=[True, False, False, False, False],
                 attn_info_list=[False, False, False, False, True], attn_dim_head=32, num_head_list=[2, 4, 8, 8, 16],
                 block_depth_info="mini"):
        super().__init__()
        

        self.use_inception_block_attn = True
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

        # for compability with Medsegdiff
        self.image_size = img_size
        self.input_img_channels = cond_channel
        self.mask_channels = in_channel
        self.self_condition = self_condition
        if self.self_condition:
            in_channel *= 2
        ##################################
        emb_dim = block_size * 16

        ##################################
        self.padding_3x3 = 1
        self.in_channel = in_channel
        self.block_size = block_size
        self.norm = RMSNorm
        self.act = act
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in attn_info_list]
        if isinstance(attn_dim_head, int):
            attn_dim_head = [attn_dim_head for _ in attn_info_list]
        self.use_checkpoint = use_checkpoint
        self.attn_info_list = attn_info_list
        self.attn_dim_head = attn_dim_head
        self.num_head_list = num_head_list
        self.drop_prob = drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.block_scale_list = [0.17, 0.1, 0.2]
        self.act_layer = get_act(act)
        ##################################

        assert len(attn_info_list) == 5, "check len(attn_info_list) == 5"
        assert len(use_checkpoint) == 5, "check len(use_checkpoint) == 5"
        assert len(num_head_list) == 5, "check len(num_head_list) == 5"

        time_emb_dim = emb_dim * 4

        if False:
            sinu_pos_emb = LearnedSinusoidalPosEmb(time_emb_dim)
            fourier_dim = time_emb_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(time_emb_dim)
            fourier_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            get_act(act),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            class_emb_dim = emb_dim * 4
            self.class_emb_layer = nn.Embedding(num_class_embeds, class_emb_dim)
            if self.cond_drop_prob > 0:
                self.null_class_emb = nn.Parameter(torch.randn(class_emb_dim))
            else:
                self.null_class_emb = None
            
            self.class_mlp = nn.Sequential(
                nn.Linear(class_emb_dim, class_emb_dim),
                get_act(act),
                nn.Linear(class_emb_dim, class_emb_dim)
            )
            emb_dim_list = [time_emb_dim, emb_channel, time_emb_dim]
            emb_type_list = ["seq", "seq", "seq"]
        else:
            self.class_emb_layer = None
            self.null_class_emb = None
            self.class_mlp = None
            emb_dim_list = [time_emb_dim, emb_channel]
            emb_type_list = ["seq", "seq"]
        self.emb_dim_list = emb_dim_list
        self.emb_type_list = emb_type_list
        self.cond_drop_prob = cond_drop_prob
        
        feature_channel = block_size * 96 * last_channel_ratio

        self.latent_model = InceptionResNetV2_Encoder(in_channel=cond_channel, img_size=img_size, block_size=block_size,
                                                      emb_channel=emb_channel, norm=norm, act=act, last_channel_ratio=1,
                                                    use_checkpoint=use_checkpoint, attn_info_list=attn_info_list,
                                                    attn_dim_head=attn_dim_head, num_head_list=num_head_list,
                                                    block_depth_info=block_depth_info)
        # Stem block
        self.stem = self.get_encode_stem(emb_dim_list, emb_type_list)
        # Mixed 5b (Inception-A block):
        self.mixed_5b, self.mixed_5b_attn = self.get_encode_mixed_5b(emb_dim_list, emb_type_list)
        # 10x block35 (Inception-ResNet-A block):
        self.block_35_mixed, self.block_35_attn, self.block_35_up = self.get_inception_block(emb_dim_list, emb_type_list,
                                                                         block_depth_list[0], self.get_inception_block_35,
                                                                         block_size * 20, 3)
        # Mixed 6a (Reduction-A block)
        self.mixed_6a, self.mixed_6a_attn = self.get_encode_mixed_6a(emb_dim_list, emb_type_list)
        # 20x block17 (Inception-ResNet-B block)
        self.block_17_mixed, self.block_17_attn, self.block_17_up = self.get_inception_block(emb_dim_list, emb_type_list,
                                                                         block_depth_list[1], self.get_inception_block_17,
                                                                         block_size * 68, 3)
        # Mixed 7a (Reduction-B block)
        self.mixed_7a, self.mixed_7a_attn = self.get_encode_mixed_7a(emb_dim_list, emb_type_list)
        # 10x block8 (Inception-ResNet-C block)
        self.block_8_mixed, self.block_8_attn, self.block_8_up = self.get_inception_block(emb_dim_list, emb_type_list,
                                                                         block_depth_list[2], self.get_inception_block_8,
                                                                         block_size * 130, 4)
        layer_idx = 4
        self.encode_final_block1 = ConvBlock2D(block_size * 130, feature_channel, 3,
                                      norm=norm, act=act, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                                      attn_info=None,
                                      use_checkpoint=use_checkpoint[layer_idx])
        self.encode_final_attn = self.get_attn_layer(feature_channel, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                                     True, use_checkpoint=self.use_checkpoint[layer_idx])
        self.encode_final_block2 = ConvBlock2D(feature_channel, feature_channel, 3,
                                                norm=norm, act=act, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                                                attn_info=None,
                                                use_checkpoint=use_checkpoint[layer_idx])
        
        decoder_layer_list = []
        skip_channel_list = get_skip_connect_channel_list(block_size)
        for decode_idx, skip_channel in enumerate(skip_channel_list):
            skip_channel = skip_channel_list[4 - decode_idx]
            decode_out_channel = decode_init_channel // (2 ** (decode_idx + 1))

            if decode_idx == 0:
                decode_in_channel = block_size * 96
            else:
                decode_in_channel = decode_init_channel // (2 ** decode_idx)

            attn_info = self.get_attn_info(attn_info_list[4 - decode_idx], num_head_list[4 - decode_idx])
            decode_emb_dim_list = emb_dim_list + [decode_out_channel]
            
            decoder_layer = MultiDecoder2D_V2(decode_in_channel, skip_channel, decode_out_channel,
                                                norm=norm, act=act, kernel_size=2, drop_prob=drop_prob,
                                                emb_dim_list=decode_emb_dim_list, attn_info=attn_info,
                                                use_checkpoint=use_checkpoint[4 - decode_idx])
            decoder_layer_list.append(decoder_layer)
        self.decoder_layer_list = nn.ModuleList(decoder_layer_list)
        self.decode_final_conv = Output2D(decode_out_channel, out_channel, act=last_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None,
                cond_drop_prob=None):
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        batch, device = x.size(0), x.device
        latent = self.latent_model(cond)
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        time_emb = self.time_mlp(time)
        
        if self.num_class_embeds is not None:
            class_emb = self.class_emb_layer(class_labels).to(dtype=x.dtype)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                null_classes_emb = repeat(self.null_class_emb, 'd -> b d', b=batch)

                class_emb = torch.where(
                    rearrange(keep_mask, 'b -> b 1'),
                    class_emb, null_classes_emb
                )

            class_emb = self.class_mlp(class_emb)
            emb_list = [time_emb, latent, class_emb]
        else:
            emb_list = [time_emb, latent]
        
        encode_feature, skip_connect_list = self.encode_forward(x, *emb_list)

        decode_feature = self.decode_forward(encode_feature, skip_connect_list, *emb_list)
        return decode_feature
    
    def encode_forward(self, x, *args):
        # skip connection name list
        # ["stem_layer_1", "stem_layer_4", "stem_layer_7", "mixed_6a", "mixed_7a"]
        skip_connect_list = []
        stem = x
        for idx, (_, layer) in enumerate(self.stem.items()):
            stem = layer(stem, *args)
            if idx in [1, 3, 6, 9]:
                skip_connect_list.append(stem)
        # mixed_5b
        mixed_5b = self.process_encode_block(self.mixed_5b, self.mixed_5b_attn, stem, *args)
        # block_35
        block_35 = self.process_inception_block(self.block_35_mixed, self.block_35_attn, self.block_35_up,
                                                mixed_5b, self.block_scale_list[0], *args)
        # mixed_6a: skip connect target
        mixed_6a = self.process_encode_block(self.mixed_6a, self.mixed_6a_attn, block_35, *args)
        skip_connect_list.append(mixed_6a)
        # block_17
        block_17 = self.process_inception_block(self.block_17_mixed, self.block_17_attn, self.block_17_up,
                                                mixed_6a, self.block_scale_list[1], *args)
        # mixed_7a: skip connect target
        mixed_7a = self.process_encode_block(self.mixed_7a, self.mixed_7a_attn, block_17, *args)
        # skip_connect_list.append(mixed_7a)
        # block_8
        block_8 = self.process_inception_block(self.block_8_mixed, self.block_8_attn, self.block_8_up,
                                               mixed_7a, self.block_scale_list[2], *args)
        # final_output
        output = self.encode_final_block1(block_8, *args)
        output = self.encode_final_attn(output)
        output = self.encode_final_block2(output, *args)

        return output, skip_connect_list[::-1]
    
    def process_encode_block(self, block, attn, x, *args):
        output = []
        for (_, layer_list) in block.items():
            output_part = x
            for layer in layer_list:
                output_part = layer(output_part, *args)
            output.append(output_part)
        output = torch.cat(output, dim=1)
        if attn is not None:
            output = attn(output)
        return output

    def process_inception_block(self, block_mixed, block_attn, block_up,
                                x, scale, *args):
        for mixed, up_layer in zip(block_mixed, block_up):
            x_temp = self.process_encode_block(mixed, None, x, *args)
            x_temp = up_layer(x_temp, *args)
            x = x + x_temp * scale
            x = self.act_layer(x)
        x = block_attn(x)
        return x
        
    def decode_forward(self, encode_feature, skip_connect_list, *args):
        decode_feature = encode_feature
        for decode_idx, layer in enumerate(self.decoder_layer_list):
            skip = skip_connect_list[decode_idx]
            decode_feature = layer(decode_feature, skip, *args)
        decode_feature = self.decode_final_conv(decode_feature)
        return decode_feature

    def get_attn_info(self, attn_info, num_heads=None, dim_head=32):
        if attn_info is None:
            return None
        elif attn_info is False:
            return {"num_heads": num_heads, "dim_head": dim_head, "full_attn": False}
        elif attn_info is True:
            return {"num_heads": num_heads, "dim_head": dim_head, "full_attn": True}
    
    def get_attn_layer(self, dim, num_heads, dim_head, use_full_attn, use_checkpoint):
        if use_full_attn:
            attn_class = Attention
        else:
            attn_class = LinearAttention
        return attn_class(dim=dim, num_heads=num_heads, dim_head=dim_head,
                          norm=self.norm, use_checkpoint=use_checkpoint)

    def get_encode_stem(self, emb_dim_list, emb_type_list):
        in_channel = self.in_channel
        block_size = self.block_size
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list

        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "emb_dim_list": emb_dim_list,
            "use_checkpoint": self.use_checkpoint,
        }
        return nn.ModuleDict({
            'stem_layer_0_0': ConvBlock2D(in_channel, block_size * 2, 3, stride=1,
                                        attn_info=None,
                                        **common_kwarg_dict),
            'stem_layer_0_1': ConvBlock2D(block_size * 2, block_size * 2, 3, stride=1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        **common_kwarg_dict),
            'stem_layer_1_0': ConvBlock2D(block_size * 2, block_size * 2, 3, stride=1,
                                        attn_info=None,
                                        **common_kwarg_dict),
            'stem_layer_1_1': ConvBlock2D(block_size * 2, block_size * 2, 3, stride=2, padding=self.padding_3x3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        **common_kwarg_dict),
            'stem_layer_2': ConvBlock2D(block_size * 2, block_size * 2, 3,
                                        attn_info=None,
                                        **common_kwarg_dict),
            'stem_layer_3': ConvBlock2D(block_size * 2, block_size * 4, 3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[1], self.attn_dim_head[1]),
                                        **common_kwarg_dict),
            'stem_layer_4': MaxPool2d(3, stride=2, padding=self.padding_3x3),
            'stem_layer_5': ConvBlock2D(block_size * 4, block_size * 5, 1,
                                        attn_info=None,
                                        **common_kwarg_dict),
            'stem_layer_6': ConvBlock2D(block_size * 5, block_size * 12, 3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[2], self.attn_dim_head[2]),
                                        **common_kwarg_dict),
            'stem_layer_7': MaxPool2d(3, stride=2, padding=self.padding_3x3)
        })
    
    def get_encode_mixed_5b(self, emb_dim_list, emb_type_list):
        block_size = self.block_size
        layer_idx = 3

        common_arg_dict = {
            "norm": self.norm,
            "act": self.act,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "use_checkpoint": self.use_checkpoint[layer_idx],
            "dropout_proba": self.drop_prob
        }

        mixed_5b_branch_0_0 = ConvBlock2D(block_size * 12, block_size * 6, 1, **common_arg_dict)
        mixed_5b_branch_1_0 = ConvBlock2D(block_size * 12, block_size * 3, 1, **common_arg_dict)
        mixed_5b_branch_1_1 = ConvBlock2D(block_size * 3, block_size * 4, 5, **common_arg_dict)
        mixed_5b_branch_2_0 = ConvBlock2D(block_size * 12, block_size * 4, 1, **common_arg_dict)
        mixed_5b_branch_2_1 = ConvBlock2D(block_size * 4, block_size * 6, 3, **common_arg_dict)
        mixed_5b_branch_2_2 = ConvBlock2D(block_size * 6, block_size * 6, 3, **common_arg_dict)
        mixed_5b_branch_pool_0 = AvgPool2d(3, stride=1, padding=1)
        mixed_5b_branch_pool_1 = ConvBlock2D(block_size * 12, block_size * 4, 1, **common_arg_dict)
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
        attn_layer = self.get_attn_layer(block_size * 20, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                         self.attn_info_list[layer_idx], use_checkpoint=self.use_checkpoint[layer_idx])

        return mixed_5b, attn_layer
    
    def get_encode_mixed_6a(self, emb_dim_list, emb_type_list):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        layer_idx = 3

        common_arg_dict = {
            "norm": self.norm,
            "act": self.act,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "use_checkpoint": self.use_checkpoint[layer_idx],
            "dropout_proba": self.drop_prob
        }

        mixed_6a_branch_0_0 = ConvBlock2D(block_size * 20, block_size * 24, 3,
                                          stride=2, padding=padding_3x3, **common_arg_dict)
        mixed_6a_branch_1_1 = ConvBlock2D(block_size * 20, block_size * 16, 1, **common_arg_dict)
        mixed_6a_branch_1_2 = ConvBlock2D(block_size * 16, block_size * 16, 3, **common_arg_dict)
        mixed_6a_branch_1_3 = ConvBlock2D(block_size * 16, block_size * 24, 3,
                                          stride=2, padding=padding_3x3, **common_arg_dict)
        mixed_6a_branch_pool_0 = MaxPool2d(3, stride=2, padding=padding_3x3)
        mixed_6a = nn.ModuleDict({
            "mixed_6a_branch_0": nn.ModuleList([mixed_6a_branch_0_0]),
            "mixed_6a_branch_1": nn.ModuleList([mixed_6a_branch_1_1,
                                                mixed_6a_branch_1_2,
                                                mixed_6a_branch_1_3]),
            "mixed_6a_branch_pool": nn.ModuleList([mixed_6a_branch_pool_0]),
        })
        attn_layer = self.get_attn_layer(block_size * 68, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                         self.attn_info_list[layer_idx], use_checkpoint=self.use_checkpoint[layer_idx])
        return mixed_6a, attn_layer
    
    def get_encode_mixed_7a(self, emb_dim_list, emb_type_list):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        layer_idx = 4
        common_arg_dict = {
            "norm": self.norm,
            "act": self.act,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "use_checkpoint": self.use_checkpoint[layer_idx],
            "dropout_proba": self.drop_prob
        }
        mixed_7a_branch_0_1 = ConvBlock2D(block_size * 68, block_size * 16, 1, **common_arg_dict)
        mixed_7a_branch_0_2 = ConvBlock2D(block_size * 16, block_size * 24, 3,
                                          stride=2, padding=padding_3x3, **common_arg_dict)
        mixed_7a_branch_1_1 = ConvBlock2D(block_size * 68, block_size * 16, 1, **common_arg_dict)
        mixed_7a_branch_1_2 = ConvBlock2D(block_size * 16, block_size * 18, 3,
                                          stride=2, padding=padding_3x3, **common_arg_dict)
        mixed_7a_branch_2_1 = ConvBlock2D(block_size * 68, block_size * 16, 1, **common_arg_dict)
        mixed_7a_branch_2_2 = ConvBlock2D(block_size * 16, block_size * 18, 3, **common_arg_dict)
        mixed_7a_branch_2_3 = ConvBlock2D(block_size * 18, block_size * 20, 3,
                                          stride=2, padding=padding_3x3, **common_arg_dict)
        mixed_7a_branch_pool_0 = MaxPool2d(3, stride=2, padding=padding_3x3)
        mixed_7a = nn.ModuleDict({
            "mixed_7a_branch_0": nn.ModuleList([mixed_7a_branch_0_1,
                                                mixed_7a_branch_0_2]),
            "mixed_7a_branch_1": nn.ModuleList([mixed_7a_branch_1_1,
                                                mixed_7a_branch_1_2]),
            "mixed_7a_branch_2_1": nn.ModuleList([mixed_7a_branch_2_1,
                                                  mixed_7a_branch_2_2,
                                                  mixed_7a_branch_2_3]),
            "mixed_7a_branch_pool": nn.ModuleList([mixed_7a_branch_pool_0]),
        })
        attn_layer = self.get_attn_layer(block_size * 130, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                         self.attn_info_list[layer_idx], use_checkpoint=self.use_checkpoint[layer_idx])
        return mixed_7a, attn_layer
    
    def get_inception_block(self, emb_dim_list, emb_type_list, block_depth, get_block_fn, out_channel, layer_idx):
        block_list = [get_block_fn(emb_dim_list, emb_type_list)
                      for _ in range(block_depth)]
        mixed_list, up_list = [list(item) for item in zip(*block_list)]
        attn_layer = self.get_attn_layer(out_channel, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                         self.attn_info_list[layer_idx], use_checkpoint=self.use_checkpoint[layer_idx])
        return nn.ModuleList(mixed_list), attn_layer, nn.ModuleList(up_list)

    def get_inception_block_35(self, emb_dim_list, emb_type_list):

        block_size = self.block_size
        in_channels = block_size * 20
        mixed_channel = block_size * 8
        layer_idx = 3
        common_arg_dict = {
            "norm": self.norm,
            "act": self.act,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "use_checkpoint": self.use_checkpoint[layer_idx],
            "dropout_proba": self.drop_prob
        }

        branch_0_0 = ConvBlock2D(in_channels, block_size * 2, 1, **common_arg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 2, 1, **common_arg_dict)
        branch_1_1 = ConvBlock2D(block_size * 2, block_size * 2, 3, **common_arg_dict)
        branch_2_0 = ConvBlock2D(in_channels, block_size * 2, 1, **common_arg_dict)
        branch_2_1 = ConvBlock2D(block_size * 2, block_size * 3, 3, **common_arg_dict)
        branch_2_2 = ConvBlock2D(block_size * 3, block_size * 4, 3, **common_arg_dict)

        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                        use_checkpoint=self.use_checkpoint[layer_idx])
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
        in_channels = block_size * 68
        mixed_channel = block_size * 24
        layer_idx = 3
        common_arg_dict = {
            "norm": self.norm,
            "act": self.act,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "use_checkpoint": self.use_checkpoint[layer_idx],
            "dropout_proba": self.drop_prob
        }
        branch_0_0 = ConvBlock2D(in_channels, block_size * 12, 1, **common_arg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 8, 1, **common_arg_dict)
        branch_1_1 = ConvBlock2D(block_size * 8, block_size * 10, [1, 7], **common_arg_dict)
        branch_1_2 = ConvBlock2D(block_size * 10, block_size * 12, [7, 1], **common_arg_dict)
        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                        use_checkpoint=self.use_checkpoint[layer_idx])
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                    branch_1_1,
                                    branch_1_2]),
        })
        return mixed, up

    def get_inception_block_8(self, emb_dim_list, emb_type_list):
        block_size = self.block_size

        in_channels = block_size * 130
        mixed_channel = block_size * 28
        layer_idx = 4
        common_arg_dict = {
            "norm": self.norm,
            "act": self.act,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "use_checkpoint": self.use_checkpoint[layer_idx],
            "dropout_proba": self.drop_prob
        }
        branch_0_0 = ConvBlock2D(in_channels, block_size * 12, 1, **common_arg_dict)
        branch_1_0 = ConvBlock2D(in_channels, block_size * 12, 1, **common_arg_dict)
        branch_1_1 = ConvBlock2D(block_size * 12, block_size * 14, [1, 3], **common_arg_dict)
        branch_1_2 = ConvBlock2D(block_size * 14, block_size * 16, [3, 1], **common_arg_dict)
        up = ConvBlock2D(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                        use_checkpoint=self.use_checkpoint[layer_idx])
        
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                    branch_1_1,
                                    branch_1_2]),
        })
        return mixed, up


class InceptionResNetV2_Encoder(InceptionResNetV2_UNet):
    def __init__(self, in_channel, img_size, block_size=16, emb_channel=1024, drop_prob=0.0,
                 norm="group", act="silu", last_channel_ratio=1,
                 use_checkpoint=False, attn_info_list=[None, False, False, False, True],
                 attn_dim_head=32, num_head_list=[2, 4, 8, 8, 16],
                 block_depth_info="mini"):
        super(InceptionResNetV2_UNet, self).__init__()

        self.use_inception_block_attn = True
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
        
        ##################################
        self.padding_3x3 = 1
        self.in_channel = in_channel
        self.block_size = block_size
        self.norm = norm
        self.act = act
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in attn_info_list]
        if isinstance(attn_dim_head, int):
            attn_dim_head = [attn_dim_head for _ in attn_info_list]
        self.use_checkpoint = use_checkpoint
        self.attn_info_list = attn_info_list
        self.attn_dim_head = attn_dim_head
        self.num_head_list = num_head_list
        self.drop_prob = drop_prob
        self.block_scale_list = [0.17, 0.1, 0.2]
        self.act_layer = get_act(act)
        ##################################
        # Stem block
        self.stem = self.get_encode_stem(None, None)
        # Mixed 5b (Inception-A block):
        self.mixed_5b, self.mixed_5b_attn = self.get_encode_mixed_5b(None, None)
        # 10x block35 (Inception-ResNet-A block):
        self.block_35_mixed, self.block_35_attn, self.block_35_up = self.get_inception_block(None, None,
                                                                         block_depth_list[0], self.get_inception_block_35,
                                                                         block_size * 20, 3)
        # Mixed 6a (Reduction-A block)
        self.mixed_6a, self.mixed_6a_attn = self.get_encode_mixed_6a(None, None)
        # 20x block17 (Inception-ResNet-B block)
        self.block_17_mixed, self.block_17_attn, self.block_17_up = self.get_inception_block(None, None,
                                                                         block_depth_list[1], self.get_inception_block_17,
                                                                         block_size * 68, 3)
        # Mixed 7a (Reduction-B block)
        self.mixed_7a, self.mixed_7a_attn = self.get_encode_mixed_7a(None, None)
        # 10x block8 (Inception-ResNet-C block)
        self.block_8_mixed, self.block_8_attn, self.block_8_up = self.get_inception_block(None, None,
                                                                         block_depth_list[2], self.get_inception_block_8,
                                                                         block_size * 130, 3)

        feature_map_size = (img_size // 32, img_size // 32)
        # Final convolution block
        layer_idx = 4
        feature_channel = block_size * 96 * last_channel_ratio
        self.encode_final_block1 = ConvBlock2D(block_size * 130, feature_channel, 3,
                                      norm=norm, act=act, emb_dim_list=None, emb_type_list=None, attn_info=None,
                                      use_checkpoint=use_checkpoint[layer_idx])
        self.encode_final_attn = self.get_attn_layer(feature_channel, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                                     True, use_checkpoint=self.use_checkpoint[layer_idx])
        self.encode_final_block2 = ConvBlock2D(feature_channel, feature_channel, 3,
                                                norm=norm, act=act, emb_dim_list=None, emb_type_list=None, attn_info=None,
                                                use_checkpoint=use_checkpoint[layer_idx])
        self.pool_layer = AttentionPool(feature_num=np.prod(feature_map_size),
                                        embed_dim=feature_channel,
                                        num_heads=self.num_head_list[layer_idx], output_dim=emb_channel)
        # self.pool_layer = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(start_dim=1)
        # ) 

        self.out = nn.Sequential(
            nn.Linear(emb_channel, emb_channel), nn.Dropout(drop_prob),
            nn.Linear(emb_channel, emb_channel), nn.Dropout(drop_prob),
            nn.Linear(emb_channel, emb_channel)
        )

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
        output, _ = super().encode_forward(x)
        output = self.pool_layer(output)
        return output