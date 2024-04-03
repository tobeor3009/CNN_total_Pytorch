import torch
from timm.models.layers import trunc_normal_
import math

import torch
import numpy as np
from torch import nn
from copy import deepcopy
from .swin_layers import PatchEmbed, PatchMerging, PatchExpanding, Output2D
from .swin_layers_diffusion import BasicLayerV2, CondLayer, SkipConv1D

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(target_list, t, x_shape):
    batch_size, *_ = t.shape
    out = target_list[t - 1]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class SwinDiffusion(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=1, cond_chans=3, out_chans=1, out_act=None,
                num_class_embeds=None,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, patch_norm=True, skip_connect=True,
                use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                self_condition=False
                ):
        super().__init__()
        cond_residuial = True
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
        self.skip_connect = skip_connect
        self.self_condition = self_condition

        if self.self_condition:
            in_chans = in_chans * 2

        time_emb_dim = embed_dim * 8
        if exists(num_class_embeds):
            class_emb_dim = embed_dim * 8
        else:
            class_emb_dim = None

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_mlp = nn.Embedding(num_class_embeds, time_emb_dim)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=norm_layer if self.patch_norm else None)
        
        self.cond_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                            in_chans=cond_chans, embed_dim=embed_dim,
                                            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.cond_init_layer = BasicLayerV2(dim=embed_dim,
                                            input_resolution=patches_resolution,
                                            depth=depths[0],
                                            num_heads=num_heads[0],
                                            window_size=window_sizes[0],
                                            mlp_ratio=self.mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                            drop_path=dpr[sum(depths[:0]):sum(
                                                depths[:+ 1])],
                                            norm_layer=norm_layer,
                                            downsample=None,
                                            use_checkpoint=use_checkpoint,
                                            pretrained_window_size=pretrained_window_sizes[0],
                                            time_emb_dim=time_emb_dim,
                                            class_emb_dim=class_emb_dim,
                                            use_residual=cond_residuial)
        # build layers
        self.encode_layers_1 = nn.ModuleList()
        self.encode_layers_2 = nn.ModuleList()
        self.encode_layers_3 = nn.ModuleList()
        self.cond_encode_layers_1 = nn.ModuleList()
        self.cond_encode_layers_2 = nn.ModuleList()
        self.cond_encode_layers_3 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            layer_1 = CondLayer(dim=layer_dim,
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
                                downsample=None,
                                use_checkpoint=use_checkpoint,
                                pretrained_window_size=pretrained_window_sizes[i_layer],
                                time_emb_dim=time_emb_dim,
                                class_emb_dim=class_emb_dim)
            layer_2 = deepcopy(layer_1)
            layer_3 = CondLayer(dim=layer_dim,
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
                                time_emb_dim=time_emb_dim,
                                class_emb_dim=class_emb_dim)

            cond_layer_1 = BasicLayerV2(dim=layer_dim,
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
                                        downsample=None,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer],
                                        time_emb_dim=time_emb_dim,
                                        class_emb_dim=class_emb_dim,
                                        use_residual=cond_residuial)
            cond_layer_2 = deepcopy(cond_layer_1)
            cond_layer_3 = BasicLayerV2(dim=layer_dim,
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
                                        time_emb_dim=time_emb_dim,
                                        class_emb_dim=class_emb_dim,
                                        use_residual=cond_residuial)
            self.encode_layers_1.append(layer_1)
            self.encode_layers_2.append(layer_2)
            self.encode_layers_3.append(layer_3)
            self.cond_encode_layers_1.append(cond_layer_1)
            self.cond_encode_layers_2.append(cond_layer_2)
            self.cond_encode_layers_3.append(cond_layer_3)
        depth_level = self.num_layers - 1
        feature_hw = (patches_resolution[0] // (2 ** depth_level),
                    patches_resolution[1] // (2 ** depth_level))
        
        self.cond_mid_layer = BasicLayerV2(dim=layer_dim,
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
                                    time_emb_dim=time_emb_dim,
                                    class_emb_dim=class_emb_dim,
                                    use_residual=cond_residuial)
        self.mid_layer = CondLayer(dim=layer_dim,
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
                                    time_emb_dim=time_emb_dim,
                                    class_emb_dim=class_emb_dim)
        self.skip_conv_layers_1 = nn.ModuleList()
        self.skip_conv_layers_2 = nn.ModuleList()
        self.cond_skip_conv_layers_1 = nn.ModuleList()
        self.cond_skip_conv_layers_2 = nn.ModuleList()
        self.decode_layers_1 = nn.ModuleList()
        self.decode_layers_2 = nn.ModuleList()
        self.decode_layers_3 = nn.ModuleList()
        self.cond_decode_layers_1 = nn.ModuleList()
        self.cond_decode_layers_2 = nn.ModuleList()
        self.cond_decode_layers_3 = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            skip_conv_layer_1 = SkipConv1D(layer_dim * 2, layer_dim)
            skip_conv_layer_2 = deepcopy(skip_conv_layer_1)
            cond_skip_conv_layer_1 = deepcopy(skip_conv_layer_2)
            cond_skip_conv_layer_2 = deepcopy(cond_skip_conv_layer_1)

            layer_1 = CondLayer(dim=layer_dim,
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
                                    upsample=None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    time_emb_dim=time_emb_dim,
                                    class_emb_dim=None)
            layer_2 = deepcopy(layer_1)
            layer_3 = CondLayer(dim=layer_dim,
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
                                    time_emb_dim=time_emb_dim,
                                    class_emb_dim=None)
            cond_layer_1 = BasicLayerV2(dim=layer_dim,
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
                                    upsample=None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    time_emb_dim=time_emb_dim,
                                    class_emb_dim=None,
                                    use_residual=cond_residuial)
            cond_layer_2 = deepcopy(cond_layer_1)
            cond_layer_3 = BasicLayerV2(dim=layer_dim,
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
                                    time_emb_dim=time_emb_dim,
                                    class_emb_dim=None,
                                    use_residual=cond_residuial)
            self.skip_conv_layers_1.append(skip_conv_layer_1)
            self.skip_conv_layers_2.append(skip_conv_layer_2)
            self.cond_skip_conv_layers_1.append(cond_skip_conv_layer_1)
            self.cond_skip_conv_layers_2.append(cond_skip_conv_layer_2)
            self.decode_layers_1.append(layer_1)
            self.decode_layers_2.append(layer_2)
            self.decode_layers_3.append(layer_3)
            self.cond_decode_layers_1.append(cond_layer_1)
            self.cond_decode_layers_2.append(cond_layer_2)
            self.cond_decode_layers_3.append(cond_layer_3)
        self.seg_final_expanding = PatchExpanding(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                    patches_resolution[1] // (2 ** i_layer)),
                                                    dim=layer_dim,
                                                    return_vector=False,
                                                    dim_scale=patch_size,
                                                    norm_layer=norm_layer
                                                    )
        self.seg_final_conv = Output2D(layer_dim // 2, out_chans, act=out_act)
        self.apply(self._init_weights)

        self.cond_init_layer._init_respostnorm()
        for bly in (self.encode_layers_1 +self.cond_encode_layers_1 +
                    self.encode_layers_2 +self.cond_encode_layers_2 +
                    self.encode_layers_3 +self.cond_encode_layers_3):
            bly._init_respostnorm()
        self.mid_layer._init_respostnorm()
        for bly in (self.decode_layers_1 + self.decode_layers_2 + self.decode_layers_3 +
                    self.cond_decode_layers_1 + self.cond_decode_layers_2 + self.cond_decode_layers_3):
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

    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None):

        time_emb = self.time_mlp(time)
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_mlp(class_labels)
            if class_emb.ndim == 3:
                class_emb = class_emb.mean(1)
            class_emb.to(dtype=x.dtype)
        x = self.patch_embed(x)
        cond = self.cond_patch_embed(cond)
        if self.ape:
            x = x + self.absolute_pos_embed
            cond = cond + self.absolute_pos_embed
        x = self.pos_drop(x)
        cond = self.pos_drop(cond)

        cond = self.cond_init_layer(cond, time_emb=time_emb, class_emb=class_emb)
        skip_connect_list_1 = []
        skip_connect_list_2 = []
        for idx, (encode_layer_1, encode_layer_2, encode_layer_3,
                  cond_encode_layer_1, cond_encode_layer_2, cond_encode_layer_3) in enumerate(zip(self.encode_layers_1,
                                                                                                self.encode_layers_2,
                                                                                                self.encode_layers_3,
                                                                                                self.cond_encode_layers_1,
                                                                                                self.cond_encode_layers_2,
                                                                                                self.cond_encode_layers_3)):
            cond = cond_encode_layer_1(cond, time_emb=time_emb, class_emb=class_emb)
            x = encode_layer_1(x, time_emb=time_emb, cond_emb=cond, class_emb=class_emb)
            if idx < len(self.encode_layers_1):
                skip_connect_list_1.insert(0, [x, cond])

            cond = cond_encode_layer_2(cond, time_emb=time_emb, class_emb=class_emb)
            x = encode_layer_2(x, time_emb=time_emb, cond_emb=cond, class_emb=class_emb)
            if idx < len(self.encode_layers_1):
                skip_connect_list_2.insert(0, [x, cond])
            cond = cond_encode_layer_3(cond, time_emb=time_emb, class_emb=class_emb)
            x = encode_layer_3(x, time_emb=time_emb, cond_emb=cond, class_emb=class_emb)

        cond = self.cond_mid_layer(x, time_emb=time_emb, class_emb=class_emb)
        x = self.mid_layer(x, time_emb=time_emb, cond_emb=cond, class_emb=class_emb)

        for idx, (skip_conv_layer_1, skip_conv_layer_2, cond_skip_conv_layer_1, cond_skip_conv_layer_2,
                  layer_1, layer_2, layer_3, cond_layer_1, cond_layer_2, cond_layer_3) in enumerate(zip(self.skip_conv_layers_1,
                                                                                                        self.skip_conv_layers_2,
                                                                                                        self.cond_skip_conv_layers_2,
                                                                                                        self.cond_skip_conv_layers_2,
                                                                                                        self.decode_layers_1,
                                                                                                        self.decode_layers_2,
                                                                                                        self.decode_layers_3,
                                                                                                        self.cond_decode_layers_1,
                                                                                                        self.cond_decode_layers_2,
                                                                                                        self.cond_decode_layers_3)):
            skip_x, skip_cond = skip_connect_list_1[idx]
            x = skip_conv_layer_1(x, skip_x)
            cond = cond_skip_conv_layer_1(x, skip_cond)
            cond = cond_layer_1(x, time_emb=time_emb, class_emb=None)
            x = layer_1(x, time_emb=time_emb, cond_emb=cond, class_emb=None)

            skip_x, skip_cond = skip_connect_list_2[idx]
            x = skip_conv_layer_2(x, skip_x)
            cond = cond_skip_conv_layer_2(x, skip_cond)
            cond = cond_layer_2(x, time_emb=time_emb, class_emb=None)
            x = layer_2(x, time_emb=time_emb, cond_emb=cond, class_emb=None)

            cond = cond_layer_3(x, time_emb=time_emb, class_emb=class_emb)
            x = layer_3(x, time_emb=time_emb, cond_emb=cond, class_emb=None)

        x = self.seg_final_expanding(x)
        x = self.seg_final_conv(x)
        return x

    def print_tensor_info(self, tensor):
        print(tensor.min(), tensor.max(), torch.isnan(tensor).any())