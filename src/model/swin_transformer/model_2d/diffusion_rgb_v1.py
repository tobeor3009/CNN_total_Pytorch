from copy import deepcopy
from timm.models.layers import trunc_normal_
import torch
import numpy as np
from functools import partial
from torch import nn
from torch.utils.checkpoint import checkpoint
from .swin_layers import Output2D
from .swin_layers_diffusion_2d import BasicLayerV1, BasicLayerV2, SkipConv1D, AttentionPool1d
from .swin_layers_diffusion_2d import exists, default, extract, SinusoidalPosEmb
from .swin_layers_diffusion_2d import PatchEmbed, PatchMerging, PatchMergingConv, PatchExpanding, PatchExpandingMulti, ConvBlock2D
from .swin_layers_diffusion_2d import LinearAttention, Attention, GroupNormChannelFirst, WrapGroupNorm



def get_norm_layer_partial(num_groups):
    return partial(GroupNormChannelFirst, num_groups=num_groups)

def get_norm_layer_partial_conv(num_groups):
    return partial(WrapGroupNorm, num_groups=num_groups)

class SwinDiffusionUnet(nn.Module):
    def __init__(self, img_size=512, patch_size=4,
                 in_chans=1, cond_chans=3, out_chans=1, out_act=None, num_class_embeds=None,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True, patch_norm=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                self_condition=False, use_residual=False, last_emb_conv_num=3, last_conv_num=3
                ):
        super().__init__()
        patch_size = int(patch_size)
        # for compability with Medsegdiff
        self.image_size = img_size
        self.input_img_channels = cond_chans
        self.mask_channels = in_chans
        
        ##################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.self_condition = self_condition
        self.use_checkpoint = use_checkpoint
        if self.self_condition:
            in_chans = in_chans * 2

        time_emb_dim = embed_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # class embedding

        emb_dim_list = [time_emb_dim, time_emb_dim]
        self.num_class_embeds = num_class_embeds
        self.r_class_emb_layer = nn.Embedding(num_class_embeds, time_emb_dim)
        self.g_class_emb_layer = nn.Embedding(num_class_embeds, time_emb_dim)
        self.b_class_emb_layer = nn.Embedding(num_class_embeds, time_emb_dim)
        self.rgb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim * 3, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.cond_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                            in_chans=in_chans, embed_dim=embed_dim,
                                            norm_layer=get_norm_layer_partial(num_heads[0]) if self.patch_norm else None)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=get_norm_layer_partial(num_heads[0]) if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            cond_pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_cond_pos_embed = nn.Parameter(cond_pos_embed_shape)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_cond_pos_embed, std=.02)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.cond_pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop = nn.Dropout(p=drop_rate)

        i_layer = 0
        common_kwarg_dict = {"dim":embed_dim,
                            "input_resolution":patches_resolution,
                            "depth":depths[i_layer],
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "use_checkpoint":use_checkpoint, "use_residual":use_residual
                            }
        self.cond_init_layer = BasicLayerV2(emb_dim_list=emb_dim_list, **common_kwarg_dict)
        self.init_layer = BasicLayerV2(emb_dim_list=emb_dim_list + [embed_dim], **common_kwarg_dict)

        # build layers
        self.cond_encode_layers_1 = nn.ModuleList()
        self.cond_encode_layers_2 = nn.ModuleList()
        self.cond_encode_attn_layers = nn.ModuleList()
        self.encode_layers_1 = nn.ModuleList()
        self.encode_layers_2 = nn.ModuleList()
        self.encode_attn_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            
            common_kwarg_dict = {"depth":depths[i_layer],
                               "num_heads":num_heads[i_layer],
                               "window_size":window_sizes[i_layer],
                               "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                               "drop":drop_rate, "attn_drop":attn_drop_rate,
                               "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                               "pretrained_window_size":pretrained_window_sizes[i_layer],
                               "use_checkpoint":use_checkpoint, "use_residual":use_residual
                               }

            cond_encode_layer_1 = BasicLayerV2(dim=layer_dim,
                                          input_resolution=feature_resolution,
                                        downsample=PatchMergingConv if i_layer == 0 else PatchMerging,
                                        emb_dim_list=emb_dim_list, **common_kwarg_dict)
            
            cond_encode_layer_2 = BasicLayerV2(dim=layer_dim * 2,
                                          input_resolution=feature_resolution // 2,
                                        downsample=None,
                                        emb_dim_list=emb_dim_list, **common_kwarg_dict)
            
            cond_encode_attn_layer = LinearAttention(dim=layer_dim * 2, num_heads=num_heads[i_layer])

            encode_layer_1 = BasicLayerV2(dim=layer_dim,
                                          input_resolution=feature_resolution,
                                        downsample=PatchMergingConv if i_layer == 0 else PatchMerging,
                                        emb_dim_list=emb_dim_list + [layer_dim * 2],
                                        **common_kwarg_dict)
            
            encode_layer_2 = BasicLayerV2(dim=layer_dim * 2,
                                          input_resolution=feature_resolution // 2,
                                        downsample=None,
                                        emb_dim_list=emb_dim_list + [layer_dim * 2],
                                        **common_kwarg_dict)
            encode_attn_layer = LinearAttention(dim=layer_dim * 2, num_heads=num_heads[i_layer])
            
            self.cond_encode_layers_1.append(cond_encode_layer_1)
            self.cond_encode_layers_2.append(cond_encode_layer_2)
            self.cond_encode_attn_layers.append(cond_encode_attn_layer)

            self.encode_layers_1.append(encode_layer_1)
            self.encode_layers_2.append(encode_layer_2)
            self.encode_attn_layers.append(encode_attn_layer)

        
        depth_level = self.num_layers
        layer_dim = int(embed_dim * 2 ** depth_level)
        feature_hw = (patches_resolution[0] // (2 ** depth_level),
                    patches_resolution[1] // (2 ** depth_level))
        
        common_kwarg_dict = {"dim":layer_dim,
                            "depth":1,
                           "input_resolution":feature_hw,
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "use_checkpoint":use_checkpoint, "use_residual":use_residual
                            }

        self.mid_layer_1 = BasicLayerV2(emb_dim_list=emb_dim_list + [layer_dim],
                                        **common_kwarg_dict)
        self.mid_attn = Attention(dim=layer_dim, num_heads=num_heads[i_layer])
        self.mid_layer_2 = BasicLayerV2(emb_dim_list=emb_dim_list + [layer_dim],
                                        **common_kwarg_dict)

        self.skip_conv_layers_1 = nn.ModuleList()
        self.skip_conv_layers_2 = nn.ModuleList()

        self.decode_layers_1 = nn.ModuleList()
        self.decode_layers_2 = nn.ModuleList()
        self.decode_attn_layers = nn.ModuleList()
        for d_i_layer in range(self.num_layers, 0, -1):

            i_layer = d_i_layer - 1
            layer_dim = int(embed_dim * 2 ** d_i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** d_i_layer),
                                            patches_resolution[1] // (2 ** d_i_layer)))

            skip_conv_layer_1 = SkipConv1D(layer_dim * 2, layer_dim)
            skip_conv_layer_2 = SkipConv1D(layer_dim * 2, layer_dim)

            common_kwarg_dict = {"dim":layer_dim,
                               "input_resolution":feature_resolution,
                                "depth":depths[i_layer],
                               "num_heads":num_heads[i_layer],
                               "window_size":window_sizes[i_layer],
                               "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                               "drop":drop_rate, "attn_drop":attn_drop_rate,
                               "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                               "pretrained_window_size":pretrained_window_sizes[i_layer],
                               "use_checkpoint":use_checkpoint, "use_residual":use_residual
                               }
            
            decode_layer_1 = BasicLayerV1(upsample=None,
                                          emb_dim_list=emb_dim_list + [layer_dim],
                                          **common_kwarg_dict)
            decode_layer_2 = BasicLayerV1(upsample=PatchExpandingMulti if d_i_layer == 0 else PatchExpanding,
                                          emb_dim_list=emb_dim_list + [layer_dim],
                                          **common_kwarg_dict)
            decode_attn_layer = LinearAttention(dim=layer_dim // 2, num_heads=num_heads[i_layer])
            
            self.skip_conv_layers_1.append(skip_conv_layer_1)
            self.skip_conv_layers_2.append(skip_conv_layer_2)
            self.decode_layers_1.append(decode_layer_1)
            self.decode_layers_2.append(decode_layer_2)
            self.decode_attn_layers.append(decode_attn_layer)

        self.seg_final_layer = BasicLayerV1(dim=embed_dim,
                                            input_resolution=patches_resolution,
                                            depth=depths[i_layer],
                                            num_heads=num_heads[i_layer],
                                            window_size=window_sizes[i_layer],
                                            mlp_ratio=self.mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                            drop_path=dpr[sum(depths[:i_layer]):sum(
                                                depths[:i_layer + 1])],
                                            norm_layer=get_norm_layer_partial(num_heads[i_layer]),
                                            upsample=None,
                                            use_checkpoint=use_checkpoint,
                                            pretrained_window_size=pretrained_window_sizes[i_layer],
                                            emb_dim_list=emb_dim_list + [embed_dim], use_residual=use_residual)
        
        self.seg_final_expanding = PatchExpandingMulti(input_resolution=patches_resolution,
                                                        dim=embed_dim,
                                                        return_vector=False,
                                                        dim_scale=patch_size,
                                                        norm_layer=get_norm_layer_partial(num_heads[i_layer])
                                                        )
        emb_type_list = ["seq", "seq", "seq"]
        self.final_emb_conv_list = nn.ModuleList()
        self.final_conv_list = nn.ModuleList()
        for emb_conv_idx in range(last_emb_conv_num):
            in_channel = embed_dim // 2 if emb_conv_idx == 0 else embed_dim
            out_channel = embed_dim // 2 if emb_conv_idx == last_emb_conv_num - 1 else embed_dim
            final_conv = ConvBlock2D(in_channel, out_channel, 3,
                                        stride=1, norm=get_norm_layer_partial_conv(num_heads[i_layer]), bias=False,
                                        emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                                        attn_info=None, use_checkpoint=use_checkpoint)
            self.final_emb_conv_list.append(final_conv)
        for conv_idx in range(last_conv_num):
            in_channel = embed_dim // 2 if conv_idx == 0 else embed_dim
            out_channel = embed_dim // 2 if conv_idx == last_conv_num - 1 else embed_dim
            final_conv = ConvBlock2D(in_channel, out_channel, 3,
                                        stride=1, norm=get_norm_layer_partial_conv(num_heads[i_layer]), bias=False,
                                        emb_dim_list=[], emb_type_list=[],
                                        attn_info=None, use_checkpoint=use_checkpoint)
            self.final_conv_list.append(final_conv)
        self.out_conv = Output2D(embed_dim // 2, out_chans, act=out_act)

        for bly in (self.cond_encode_layers_1 + self.cond_encode_layers_2):
            bly._init_respostnorm()
        for bly in (self.encode_layers_1 + self.encode_layers_2):
            bly._init_respostnorm()
        self.mid_layer_1._init_respostnorm()
        self.mid_layer_2._init_respostnorm()
        for bly in (self.decode_layers_1 + self.decode_layers_2):
            bly._init_respostnorm()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def attn_forward_checkpoint(self, attn_layer, x):
        if self.use_checkpoint:
            if self.use_checkpoint:
                x = checkpoint(attn_layer, x,
                               use_reentrant=False)
            else:
                x = attn_layer(x)
        return x
    
    def final_conv_forward_checkpoint(self, conv_layer, x, *args):
        if self.use_checkpoint:
            if self.use_checkpoint:
                x = checkpoint(conv_layer, x, *args,
                               use_reentrant=False)
            else:
                x = conv_layer(x, *args)
        return x
    
    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        time_emb = self.time_mlp(time)

        r_class, g_class, b_class = class_labels.chunk(3, dim=1)
        r_class, g_class, b_class = r_class.squeeze(1), g_class.squeeze(1), b_class.squeeze(1)
        r_class = self.r_class_emb_layer(r_class).to(dtype=x.dtype)
        g_class = self.g_class_emb_layer(g_class).to(dtype=x.dtype)
        b_class = self.b_class_emb_layer(b_class).to(dtype=x.dtype)
        class_emb = torch.cat([r_class, g_class, b_class], dim=1)
        class_emb = self.rgb_mlp(class_emb)

        emb_list = [time_emb, class_emb]
        cond = self.cond_patch_embed(cond)
        
        x = self.patch_embed(x)
        if self.ape:
            cond = cond + self.absolute_cond_pos_embed
            x = x + self.absolute_pos_embed
        cond = self.cond_pos_drop(cond)
        x = self.pos_drop(x)
        r_x = x.clone()

        cond = self.cond_init_layer(cond, *emb_list)
        x = self.init_layer(x, *emb_list, cond)

        skip_connect_list_1 = []
        skip_connect_list_2 = []
        for cond_encode_layer_1, cond_encode_layer_2, cond_encode_attn, encode_layer_1, encode_layer_2, encode_attn in zip(self.cond_encode_layers_1, self.cond_encode_layers_2, self.cond_encode_attn_layers, 
                                                                                                            self.encode_layers_1, self.encode_layers_2, self.encode_attn_layers):
            cond = cond_encode_layer_1(cond, *emb_list)
            x = encode_layer_1(x, *emb_list, cond)
            skip_connect_list_1.append([cond, x])
            
            cond = cond_encode_layer_2(cond, *emb_list)
            x = encode_layer_2(x, *emb_list, cond)
            skip_connect_list_2.append([cond, x])
            
            cond = self.attn_forward_checkpoint(cond_encode_attn, x)
            x = self.attn_forward_checkpoint(encode_attn, x)

        x = self.mid_layer_1(x, *emb_list, cond)
        x = self.attn_forward_checkpoint(self.mid_attn, x)
        x = self.mid_layer_2(x, *emb_list, cond)

        for skip_conv_layer_1, skip_conv_layer_2, decode_layer_1, decode_layer_2, decode_attn in zip(self.skip_conv_layers_1, self.skip_conv_layers_2,
                                                                                                     self.decode_layers_1, self.decode_layers_2, self.decode_attn_layers):
            
            skip_cond, skip_x = skip_connect_list_1.pop()
            x = skip_conv_layer_1(x, skip_x)
            x = decode_layer_1(x, *emb_list, skip_cond)
            
            skip_cond, skip_x = skip_connect_list_2.pop()
            x = skip_conv_layer_2(x, skip_x)
            x = decode_layer_2(x, *emb_list, skip_cond)

            x = self.attn_forward_checkpoint(decode_attn, x)

        x = self.seg_final_layer(x, *emb_list, r_x)
        x = self.seg_final_expanding(x)
        for final_emb_conv in self.final_emb_conv_list:
            x = self.final_conv_forward_checkpoint(final_emb_conv, x, *emb_list)
        for final_conv in self.final_conv_list:
            x = self.final_conv_forward_checkpoint(final_conv, x)
        x = self.out_conv(x)
        
        return x

    def print_tensor_info(self, tensor):
        print(tensor.min(), tensor.max(), torch.isnan(tensor).any())