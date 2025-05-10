from timm.models.layers import trunc_normal_
import torch
import numpy as np
from functools import partial
from torch import nn
from einops.layers.torch import Rearrange
from ..layers import PixelShuffle3D
from .swin_layers import Output3D, get_act
from .swin_layers_diffusion_2d import Attention, RMSNorm, LambdaLayer
from .swin_layers_diffusion_2d import BasicLayerV1, BasicLayerV2, SkipLinear, AttentionPool1d
from .swin_layers_diffusion_2d import default, prob_mask_like
from .swin_layers_diffusion_2d import PatchEmbed, PatchMerging, PatchExpanding, ConvBlock2D, Interpolate
from .swin_layers_diffusion_3d import BasicLayerV1 as BasicLayerV1_3D
from .swin_layers_diffusion_3d import BasicLayerV2 as BasicLayerV2_3D
from .swin_layers_diffusion_3d import PatchEmbed as PatchEmbed3D
from .swin_layers_diffusion_3d import PatchMerging as PatchMerging3D
from .swin_layers_diffusion_3d import PatchExpanding as PatchExpanding3D
from .swin_layers_diffusion_3d import PatchExpandingLinear as PatchExpandingLinear3D
from .swin_layers_diffusion_3d import PatchExpandingMulti as PatchExpandingMulti3D
from .swin_layers_diffusion_3d import ConvBlock3D
from .swin_layers_diffusion_2d import BasicLayerV2, SkipConv1D, AttentionPool1d
from .swin_layers_diffusion_2d import exists, default, extract, LearnedSinusoidalPosEmb, Attention, RMSNorm
from .swin_layers_diffusion_2d import PatchEmbed, PatchMerging, PatchExpanding
from .swin_layers_diffusion_2d import get_norm_layer_partial, get_norm_layer_partial_conv
from einops import rearrange, repeat

class SwinX2CT(nn.Module):
    def __init__(self, img_size=512, patch_size=4,
                 in_chans=1, seg_out_chans=1, seg_out_act=None,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True, patch_norm=True,
                drop_rate=0., qkv_drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                use_checkpoint=False, pretrained_window_sizes=0,
                use_residual=False):
        super().__init__()
        self.model_norm_layer = nn.LayerNorm
        patch_size = int(patch_size)
        
        ##################################
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_sizes = window_sizes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.qkv_drop_rate = qkv_drop_rate
        self.attn_drop_rate = attn_drop_rate
        if isinstance(pretrained_window_sizes, int):
            pretrained_window_sizes = [pretrained_window_sizes for _ in num_heads]
        self.pretrained_window_sizes = pretrained_window_sizes
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_heads]
        self.use_checkpoint = use_checkpoint
        self.use_residual = use_residual
        self.seg_out_chans = seg_out_chans
        self.seg_out_act = seg_out_act
        ##################################
        self.num_layers = len(depths)
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_dim = int(self.embed_dim * 2 ** self.num_layers)
        self.emb_dim_list = []
        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=self.model_norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patches_resolution_3d = np.array([patches_resolution[0] for _ in range(3)])
        self.feature_hw = self.patches_resolution // (2 ** self.num_layers)
        self.feature_zhw = self.patches_resolution_3d // (2 ** self.num_layers)

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.init_layer = self.get_init_layer()
        self.encode_layers = self.get_encode_layers()
        self.mid_layer_1, self.mid_2d_3d, self.mid_layer_2 = self.get_mid_layer()
        
        self.decode_layers = self.get_decode_layers()
        self.seg_final_layer, self.seg_final_interpolate, self.out_act = self.get_seg_final_layers()
        for bly in self.decode_layers:
            bly._init_respostnorm()
        self.seg_final_layer._init_respostnorm()

        self.apply(self._init_weights)

        for bly in self.encode_layers:
            bly._init_respostnorm()
        self.mid_layer_1._init_respostnorm()
        self.mid_layer_2._init_respostnorm()
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

    def forward(self, x):
        x, skip_connect_list = self.process_encode_layers(x)
        x = self.process_mid_layers(x)
        seg_output = self.process_decode_layers(x, skip_connect_list)
        return seg_output
    
    def process_encode_layers(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.init_layer(x)

        skip_connect_list = []
        for encode_layer in self.encode_layers:
            
            x = encode_layer(x)
            skip_connect_list.append(x)
        return x, skip_connect_list

    def process_mid_layers(self, x):
        x = self.mid_layer_1(x)
        x = self.mid_2d_3d(x)
        x = self.mid_layer_2(x)
        return x

    def process_decode_layers(self, x, skip_connect_list):
        for idx, decode_layer in enumerate(self.decode_layers):
            skip_x = skip_connect_list.pop()
            skip_x = repeat(skip_x, 'b n c -> b (repeat_num n) c',
                            repeat_num=self.feature_zhw[0] * (2 ** idx))
            x = decode_layer(x, skip_x)

        x = self.seg_final_layer(x)
        x = self.seg_final_interpolate(x)
        x = self.out_act(x)
        return x
    
    
    def get_layer_config_dict(self, dim, feature_resolution, i_layer, emb_dim_list):
        depths = self.depths
        common_kwarg_dict = {"dim":dim,
                            "input_resolution":feature_resolution,
                            "depth":self.depths[i_layer],
                            "num_heads":self.num_heads[i_layer],
                            "window_size":self.window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":self.qkv_bias, 
                            "drop":self.drop_rate, "qkv_drop": self.qkv_drop_rate, "attn_drop":self.attn_drop_rate,
                            "drop_path":self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":self.model_norm_layer,
                            "pretrained_window_size":self.pretrained_window_sizes[i_layer],
                            "emb_dim_list":emb_dim_list,
                            "use_checkpoint":self.use_checkpoint[i_layer], "use_residual":self.use_residual
                            }
        return common_kwarg_dict
    
    def get_init_layer(self):
        common_kwarg_dict = self.get_layer_config_dict(self.embed_dim, self.patches_resolution, 0, self.emb_dim_list)
        init_layer = BasicLayerV2(**common_kwarg_dict)
        return init_layer
    
    def get_encode_layers(self):
        encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(self.embed_dim * 2 ** i_layer)
            feature_resolution = self.patches_resolution // (2 ** i_layer)
            common_kwarg_dict = self.get_layer_config_dict(layer_dim, feature_resolution, i_layer, self.emb_dim_list)
            common_kwarg_dict["downsample"] = PatchMerging
            encode_layer = BasicLayerV2(**common_kwarg_dict)
            encode_layers.append(encode_layer)
        return encode_layers
    
    def get_mid_layer(self):
        i_layer = self.num_layers - 1
        feature_dim = self.feature_dim

        common_kwarg_dict_2d = self.get_layer_config_dict(feature_dim, self.feature_hw, i_layer, self.emb_dim_list)
        common_kwarg_dict_3d = self.get_layer_config_dict(feature_dim, self.feature_zhw, i_layer, self.emb_dim_list)
        mid_layer_1 = BasicLayerV2(**common_kwarg_dict_2d)

        mid_2d_3d = nn.Sequential(
            Attention(dim=feature_dim, num_heads=self.num_heads[i_layer],
                                  use_checkpoint=self.use_checkpoint[i_layer]),
            nn.Conv1d(np.prod(self.feature_hw), np.prod(self.feature_zhw), 
                      kernel_size=1, bias=False)
        )
        mid_layer_2 = BasicLayerV2_3D(**common_kwarg_dict_3d)
        return mid_layer_1, mid_2d_3d, mid_layer_2
    
    def get_decode_layers(self):
        decode_layers = nn.ModuleList()
        for d_i_layer in range(self.num_layers, 0, -1):
            i_layer = d_i_layer - 1
            layer_dim = int(self.embed_dim * 2 ** d_i_layer)
            feature_resolution_3d = self.patches_resolution_3d // (2 ** d_i_layer)
            common_kwarg_dict = self.get_layer_config_dict(layer_dim, feature_resolution_3d, i_layer, [layer_dim])
            common_kwarg_dict["upsample"] = PatchExpanding3D
            decode_layer = BasicLayerV1_3D(**common_kwarg_dict)
            decode_layers.append(decode_layer)

        return decode_layers
    
    def get_seg_final_layers(self):
        z, h, w = self.patches_resolution_3d
        i_layer = 0
        common_kwarg_dict = self.get_layer_config_dict(self.embed_dim, self.patches_resolution_3d, i_layer, self.emb_dim_list)
        seg_final_layer = BasicLayerV1_3D(**common_kwarg_dict)
        seg_final_interpolate = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 32, bias=False),
            Rearrange('b (z h w) c -> b c z h w', z=z, h=h, w=w),
            PixelShuffle3D(4)
        )
        if 4 // self.patch_size >= 1:
            seg_final_interpolate.append(Interpolate(scale_factor=1 / (4 // self.patch_size), mode="trilinear"))
        out_conv = Output3D(self.embed_dim // 2, self.seg_out_chans, act=self.seg_out_act)
        return seg_final_layer, seg_final_interpolate, out_conv