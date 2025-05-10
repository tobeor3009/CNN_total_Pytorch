import torch
from timm.models.layers import trunc_normal_
import torch
import numpy as np
from itertools import zip_longest
from functools import partial
from torch import nn
from copy import deepcopy
from .swin_layers import Output3D
from .swin_layers_diffusion_3d import BasicLayerV1 as BasicLayerV1_3D
from .swin_layers_diffusion_3d import BasicLayerV2 as BasicLayerV2_3D
from .swin_layers_diffusion_3d import PatchEmbed as PatchEmbed3D
from .swin_layers_diffusion_3d import PatchMerging as PatchMerging3D
from .swin_layers_diffusion_3d import PatchExpanding as PatchExpanding3D
from .swin_layers_diffusion_3d import PatchExpandingMulti as PatchExpandingMulti3D
from .swin_layers_diffusion_3d import ConvBlock3D
from .swin_layers_diffusion_2d import BasicLayerV2, SkipConv1D, AttentionPool1d
from .swin_layers_diffusion_2d import exists, default, extract, SinusoidalPosEmb
from .swin_layers_diffusion_2d import PatchEmbed, PatchMerging, PatchExpanding
from .swin_layers_diffusion_2d import get_norm_layer_partial, get_norm_layer_partial_conv


class GroupNormChannelFirst(nn.GroupNorm):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return super().forward(x).permute(0, 2, 1)
    
default_norm = partial(GroupNormChannelFirst, num_groups=8)

class SwinXrayCTAutoEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=4, cond_chans=16,
                 in_chans=1, out_chans=1, out_act="sigmoid", emb_chans=1024,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, latent_drop_rate=0.1,
                patch_norm=True, skip_connect=True,
                use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                self_condition=False, use_residual=False, last_conv_num=2
                ):
        super().__init__()
        patch_size = int(patch_size)
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
        self.skip_connect = skip_connect
        self.self_condition = self_condition
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_heads]
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
        self.latent_encoder = SwinDiffusionEncoder(img_size=img_size, patch_size=patch_size, in_chans=cond_chans, emb_chans=emb_chans,
                                                    embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                                    window_sizes=window_sizes, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape,
                                                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                                    patch_norm=patch_norm,
                                                    use_checkpoint=use_checkpoint, pretrained_window_sizes=pretrained_window_sizes)
        emb_dim_list = [time_emb_dim, emb_chans]
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size,
                                        in_chans=in_chans, embed_dim=embed_dim,
                                        norm_layer=get_norm_layer_partial(num_heads[0]) if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.latent_drop = nn.Dropout(p=latent_drop_rate)
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
                            "emb_dim_list":emb_dim_list,
                            "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                            }
        self.init_layer = BasicLayerV2_3D(**common_kwarg_dict)
        # build layers
        self.encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array(patches_resolution) // (2 ** i_layer)
            common_kwarg_dict = {"dim":layer_dim,
                                "input_resolution":feature_resolution,
                                "depth":depths[i_layer],
                                "num_heads":num_heads[i_layer],
                                "window_size":window_sizes[i_layer],
                                "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                                "drop":drop_rate, "attn_drop":attn_drop_rate,
                                "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                                "downsample":PatchMerging3D,
                                "pretrained_window_size":pretrained_window_sizes[i_layer],
                                "emb_dim_list":emb_dim_list,
                                "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                                }
            
            encode_layer = BasicLayerV2_3D(**common_kwarg_dict)
            self.encode_layers.append(encode_layer)
        depth_level = self.num_layers
        layer_dim = int(embed_dim * 2 ** depth_level)
        feature_hw = np.array(patches_resolution) // (2 ** depth_level)
        common_kwarg_dict = {"dim":layer_dim,
                            "depth":depths[i_layer],
                           "input_resolution":feature_hw,
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "emb_dim_list":emb_dim_list,
                            "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                            }
        self.mid_layer = BasicLayerV2_3D(**common_kwarg_dict)
        self.skip_conv_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        for d_i_layer in range(self.num_layers, 0, -1):
            i_layer = d_i_layer - 1
            layer_dim = int(embed_dim * 2 ** d_i_layer)
            feature_resolution = np.array(patches_resolution) // (2 ** d_i_layer)

            skip_conv_layer = SkipConv1D(layer_dim * 2, layer_dim,
                                         norm=get_norm_layer_partial_conv(num_heads[i_layer]))
            common_kwarg_dict = {"dim":layer_dim,
                               "input_resolution":feature_resolution,
                                "depth":depths[i_layer],
                               "num_heads":num_heads[i_layer],
                               "window_size":window_sizes[i_layer],
                               "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                               "drop":drop_rate, "attn_drop":attn_drop_rate,
                               "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                               "upsample":PatchExpanding3D,
                               "pretrained_window_size":pretrained_window_sizes[i_layer],
                               "emb_dim_list":emb_dim_list,
                               "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                               }

            decode_layer = BasicLayerV1_3D(**common_kwarg_dict)
            self.skip_conv_layers.append(skip_conv_layer)
            self.decode_layers.append(decode_layer)

        common_kwarg_dict = {"dim":embed_dim,
                            "input_resolution":patches_resolution,
                            "depth":depths[i_layer],
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                            "upsample":None,
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "emb_dim_list":emb_dim_list,
                            "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                            }
        self.final_layer = BasicLayerV1_3D(**common_kwarg_dict)
        self.final_expanding = PatchExpanding3D(input_resolution=patches_resolution,
                                                dim=embed_dim,
                                                return_vector=False,
                                                dim_scale=patch_size,
                                                norm_layer=get_norm_layer_partial(num_heads[i_layer]))
        self.final_conv_list = nn.ModuleList()
        common_kwarg_dict = {
            "kernel_size":3, "stride":1,
            "norm":get_norm_layer_partial_conv(num_heads[i_layer]), "bias":False,
            "emb_dim_list":[], "emb_type_list":[], "use_checkpoint":use_checkpoint[i_layer]
        }
        for _ in range(last_conv_num):
            in_channel = embed_dim // 2
            out_channel = embed_dim // 2
            final_conv = ConvBlock3D(in_channel, out_channel, **common_kwarg_dict)
            self.final_conv_list.append(final_conv)
        self.out_conv = Output3D(embed_dim // 2, out_chans, act=out_act)
        self.apply(self._init_weights)

        for bly in self.encode_layers:
            bly._init_respostnorm()
        self.mid_layer._init_respostnorm()
        for bly in self.decode_layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, x, time, cond=None, x_self_cond=None, *args, **kwargs):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        time_emb = self.time_mlp(time)
        latent = self.latent_encoder(cond)
        latent = self.latent_drop(latent)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        emb_list = [time_emb, latent]
        skip_connect_list = []
        for encode_layer in self.encode_layers:
            x = encode_layer(x, *emb_list)
            skip_connect_list.append(x)

        x = self.mid_layer(x, *emb_list)
        
        for skip_conv_layer, decode_layer in zip(self.skip_conv_layers, self.decode_layers):
            skip_x = skip_connect_list.pop()
            x = skip_conv_layer(x, skip_x)
            x = decode_layer(x, *emb_list)
        
        x = self.final_layer(x, *emb_list)
        x = self.final_expanding(x)
        for final_conv in self.final_conv_list:
            x = final_conv(x)
        x = self.out_conv(x)
        return x

    def print_tensor_info(self, tensor):
        print(tensor.min(), tensor.max(), torch.isnan(tensor).any())


class SwinDiffusionEncoder(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=1, emb_chans=1024,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, patch_norm=True,
                use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                use_residual=False
                ):
        super().__init__()
        patch_size = int(patch_size)

        ##################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=get_norm_layer_partial(num_heads[0]) if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        self.encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array(patches_resolution) // (2 ** i_layer)
            common_kwarg_dict = {"dim":layer_dim,
                                "input_resolution":feature_resolution,
                                "depth":depths[i_layer],
                                "num_heads":num_heads[i_layer],
                                "window_size":window_sizes[i_layer],
                                "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                                "drop":drop_rate, "attn_drop":attn_drop_rate,
                                "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                                "downsample":PatchMerging,
                                "pretrained_window_size":pretrained_window_sizes[i_layer],
                                "emb_dim_list":[],
                                "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                                }
            
            encode_layer = BasicLayerV2(**common_kwarg_dict)
            self.encode_layers.append(encode_layer)
        depth_level = self.num_layers
        layer_dim = int(embed_dim * 2 ** depth_level)
        feature_hw = np.array(patches_resolution) // (2 ** depth_level)
        common_kwarg_dict = {"dim":layer_dim,
                            "depth":depths[i_layer],
                           "input_resolution":feature_hw,
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":get_norm_layer_partial(num_heads[i_layer]),
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "emb_dim_list":[],
                            "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                            }
        self.mid_layer = BasicLayerV2(**common_kwarg_dict)
        
        self.pool_layer = AttentionPool1d(sequence_length=np.prod(feature_hw), embed_dim=layer_dim, 
                                          num_heads=8, output_dim=emb_chans, channel_first=False)
        for bly in self.encode_layers:
            bly._init_respostnorm()
        self.mid_layer._init_respostnorm()
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}
    
    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for encode_layer in self.encode_layers:
            x = encode_layer(x)

        x = self.mid_layer(x)
        x = self.pool_layer(x)
        return x