import torch
from timm.models.layers import trunc_normal_
import torch
import numpy as np
from itertools import zip_longest
from functools import partial
from torch import nn
from copy import deepcopy
from .swin_layers import Output2D
from .swin_layers_diffusion_2d import BasicLayerV2, CondLayer, SkipConv1D, AttentionPool1d
from .swin_layers_diffusion_2d import exists, default, extract, SinusoidalPosEmb
from .swin_layers_diffusion_2d import PatchEmbed, PatchMerging, PatchExpanding


class GroupNormChannelFirst(nn.GroupNorm):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return super().forward(x).permute(0, 2, 1)
    
default_norm = partial(GroupNormChannelFirst, num_groups=8)

class SwinXrayCT2D_2D(nn.Module):
    def __init__(self, img_size=512, patch_size=4,
                 in_chans=1, out_chans=1, out_act="sigmoid",
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=default_norm, patch_norm=True, skip_connect=True,
                use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]
                ):
        super().__init__()
        patch_size = int(patch_size)
        self.image_size = img_size
        self.mask_channels = in_chans
        
        ##################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.skip_connect = skip_connect

        # class embedding
        class_emb_dim = embed_dim * 4
        self.num_class_embeds = img_size
        self.class_mlp = nn.Embedding(self.num_class_embeds, class_emb_dim)
            

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=norm_layer if self.patch_norm else None)
        
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
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            encode_layer = BasicLayerV2(dim=layer_dim,
                                        input_resolution=feature_resolution,
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        window_size=window_sizes[i_layer],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(
                                            depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer],
                                        time_emb_dim=None,
                                        class_emb_dim=class_emb_dim,
                                        use_residual=True)
            self.encode_layers.append(encode_layer)
        depth_level = self.num_layers - 1
        feature_hw = (patches_resolution[0] // (2 ** depth_level),
                    patches_resolution[1] // (2 ** depth_level))
        
        self.mid_layer = BasicLayerV2(dim=layer_dim,
                                    input_resolution=feature_hw,
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_sizes[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(
                                        depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    upsample=None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    time_emb_dim=None,
                                    class_emb_dim=class_emb_dim)
        self.skip_conv_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))

            decode_layer = BasicLayerV2(dim=layer_dim,
                                    input_resolution=feature_resolution,
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_sizes[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(
                                        depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    upsample=PatchExpanding if (
                                        i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    time_emb_dim=None,
                                    class_emb_dim=class_emb_dim)
            self.decode_layers.append(decode_layer)
            
            if i_layer > 0:
                skip_conv_layer = SkipConv1D(layer_dim * 2, layer_dim)
                self.skip_conv_layers.append(skip_conv_layer)
        self.seg_final_expanding = PatchExpanding(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                    patches_resolution[1] // (2 ** i_layer)),
                                                    dim=layer_dim,
                                                    return_vector=False,
                                                    dim_scale=patch_size,
                                                    norm_layer=norm_layer
                                                    )
        self.seg_final_conv = Output2D(layer_dim // 2, out_chans, act=out_act)
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

    def forward(self, x, z_position=None):

        if z_position is None:
            raise ValueError("z_position is essential.")
        class_emb = self.class_mlp(z_position).to(dtype=x.dtype)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        skip_connect_list = []
        for idx, encode_layer in enumerate(self.encode_layers):
            x = encode_layer(x, time_emb=None, class_emb=class_emb)
            if idx < len(self.encode_layers) - 1:
                skip_connect_list.insert(0, x)
        x = self.mid_layer(x, time_emb=None, class_emb=class_emb)
        
        for idx, (skip_conv_layer, decode_layer) in enumerate(zip_longest(self.skip_conv_layers, self.decode_layers)):
            if idx < len(self.decode_layers) - 1 and self.skip_connect:
                skip_x = skip_connect_list[idx]
                x = skip_conv_layer(x, skip_x)
            x = decode_layer(x, time_emb=None, class_emb=class_emb)
        
        x = self.seg_final_expanding(x)
        x = self.seg_final_conv(x)
        return x

    def print_tensor_info(self, tensor):
        print(tensor.min(), tensor.max(), torch.isnan(tensor).any())