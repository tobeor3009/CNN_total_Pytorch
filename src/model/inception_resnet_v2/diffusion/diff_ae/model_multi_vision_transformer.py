from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.utils.checkpoint import checkpoint
from .nn import timestep_embedding
from .diffusion_layer import default, Return, EmbedSequential, conv_nd
from .diffusion_layer import ConvBlockND, ResNetBlockND, OutputND
from .diffusion_layer import ResNetBlockNDSkip, ConvBlockNDSkip, MultiDecoderND_V2
from .diffusion_layer import get_maxpool_nd, get_avgpool_nd
from .diffusion_layer import LinearAttention, Attention, AttentionBlock, MaxPool2d, AvgPool2d, MultiInputSequential
from .diffusion_layer import default, prob_mask_like, LearnedSinusoidalPosEmb, SinusoidalPosEmb, GroupNorm32
from .diffusion_layer import feature_z_normalize, z_normalize
from src.model.inception_resnet_v2.common_module.layers import get_act, get_norm
from einops import rearrange, repeat
from .sub_models import MLPSkipNet, Classifier
from src.model.inception_resnet_v2.diffusion.diff_ae.flash_attn import FlashMultiheadAttention

def get_encode_feature_channel(block_size, model_depth):
    feature_channel = block_size * (2 ** model_depth)
    return int(round(feature_channel))

def get_time_emb_dim(block_size):
    # emb_dim = block_size * 16
    # I found 512 is best size for all size
    emb_dim = 512
    time_emb_dim_init = emb_dim // 2
    time_emb_dim = emb_dim * 4
    return emb_dim, time_emb_dim_init, time_emb_dim

class ClassificationHeadSimple(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_proba, act):
        super(ClassificationHeadSimple, self).__init__()
        INPLACE = True
        # Global Average Pooling Layer

        # First fully connected layer
        self.fc_1 = nn.Linear(in_channels, in_channels * 2)
        self.drop_1 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_1 = nn.ReLU6(inplace=INPLACE)

        # Second fully connected layer
        self.fc_2 = nn.Linear(in_channels * 2, in_channels)
        self.drop_2 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_2 = nn.ReLU6(inplace=INPLACE)

        # Dropout layer

        # Third fully connected layer
        self.fc_3 = nn.Linear(in_channels, in_channels // 2)
        self.drop_3 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_3 = nn.ReLU6(inplace=INPLACE)
        # Output layer
        self.fc_out = nn.Linear(in_channels // 2, num_classes)
        self.last_act = get_act(act)

    def forward(self, x):
        x = x.mean(dim=-1)
        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.act_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        x = self.act_2(x)

        x = self.fc_3(x)
        x = self.drop_3(x)
        x = self.act_3(x)
        x = self.fc_out(x)
        x = self.last_act(x)

        return x
    
class VisionEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, init_channel=3, img_dim=2):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = conv_nd(
            dims=img_dim,
            in_channels=init_channel,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, image_tensor: torch.FloatTensor) -> torch.Tensor:
        _, _, *spatial = image_tensor.shape # [Batch_Size, Channels, Height, Width]
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        patch_embeds = self.patch_embedding(image_tensor)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=None, dropout=0.0, use_checkpoint=False):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads, causal=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.layer_norm1 = nn.RMSNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.RMSNorm(embed_dim, eps=1e-6)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint(self._forward_impl, x)
        else:
            x = self._forward_impl(x)
        return x
    
    def _forward_impl(self, x):
        
        residual = x
        hidden_states = self.layer_norm1(x)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        hidden_states = self.layer_norm2(x)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class VisionTransformerMultiTask(nn.Module):
    def __init__(self, in_channel=3, cond_channel=3, img_size=512, patch_size=4, embed_dim=768, 
                 encode_layer_depth=12,
                 use_checkpoint=False, num_head=8, drop_prob=0.0,
                 seg_out_channel=2, seg_act="softmax", seg_decode_fn_str_list=["conv_transpose", "pixel_shuffle"],
                 class_out_channel=2, class_act="softmax",
                 recon_out_channel=None, recon_act="tanh", recon_decode_fn_str_list=["conv_transpose", "pixel_shuffle"],
                 validity_shape=(1, 8, 8), validity_act=None,
                 get_seg=True, get_class=False, get_recon=False, get_validity=False,
                 img_dim=2):
        super().__init__()
        self.use_inception_block_attn = True
        # for compability with diffusion_sample
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.cond_channel = cond_channel
        self.num_head = num_head
        ##################################
        self.padding_3x3 = 1
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.img_dim = img_dim
        self.encode_layer_depth = encode_layer_depth
        self.decode_layer_depth = int(np.log2(patch_size))

        self.drop_prob = drop_prob
        ##################################
        ##################################
        self.use_non_diffusion = get_seg or get_class or get_recon or get_validity
        self.get_seg = get_seg
        self.get_class = get_class
        self.get_recon = get_recon
        self.get_validity = get_validity
        ##################################
        self.image_shape = self.get_image_init_shape()
        ##################################
        self.set_encoder()
        if get_seg:
            self.seg_decoder_list = self.get_decoder(seg_out_channel, seg_act, decode_fn_str_list=seg_decode_fn_str_list)
        if get_class:
            self.class_head = ClassificationHeadSimple(embed_dim, class_out_channel, drop_prob, class_act)
        if get_recon:
            recon_out_channel = recon_out_channel or in_channel
            self.recon_decoder_list = self.get_decoder(recon_out_channel, recon_act, decode_fn_str_list=recon_decode_fn_str_list)
        if get_validity:
            self.validity_head = self.get_validity_block(validity_shape, validity_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = Return()
        encode_feature, encode_feature_2d = self.encode_forward(x)
        
        if self.get_seg:
            seg_decode_feature = self.decode_forward(self.seg_decoder_list, encode_feature_2d)
            output["seg_pred"] = seg_decode_feature
        if self.get_class:
            class_output = self.class_head(encode_feature)
            output["class_pred"] = class_output
        if self.get_recon:
            recon_decode_feature = self.decode_forward(self.recon_decoder_list, encode_feature_2d)
            output["recon_pred"] = recon_decode_feature
        if self.get_validity:
            validitiy_output = self.validity_head(encode_feature_2d)
            output["validity_pred"] = validitiy_output
        return output

    def encode_forward(self, x, *emb_args):
        # skip connection name list
        x = self.init_block(x, *emb_args)
        for encode_block in self.encode_block_list:
            x = encode_block(x, *emb_args)

        x = rearrange(x, "b n d -> b d n")
        x_nd = x.view(-1, self.embed_dim, *self.image_shape)
        return x, x_nd
    
    def decode_forward(self, decoder_list, encode_feature, *args):
        decode_block_list, decode_layer_up_list, decode_final_conv = decoder_list
        decode_feature = encode_feature
        for decode_idx, (decode_block, decode_layer_up) in enumerate(zip(decode_block_list, decode_layer_up_list)):
            decode_feature = decode_block(decode_feature, *args)
            decode_feature = decode_layer_up(decode_feature, *args)
        decode_feature = decode_final_conv(decode_feature)
        return decode_feature
    
    def get_common_kwarg_dict(self, image_shape=None):
        common_kwarg_dict = {
            "norm": nn.RMSNorm,
            "act": "silu",
            "kernel_size": 3,
            "padding" : 1,
            "dropout_proba": self.drop_prob,
            "img_dim": self.img_dim,
            "image_shape": image_shape
        }
        return common_kwarg_dict
        
    def set_encoder(self):
        
        self.init_block = VisionEmbedding(image_size=self.img_size, patch_size=self.patch_size, 
                                       embed_dim=self.embed_dim, init_channel=self.in_channel, img_dim=self.img_dim)
        self.encode_block_list = nn.ModuleList([
            EncoderLayer(embed_dim=self.embed_dim, num_heads=self.num_head, hidden_dim=self.embed_dim * 4,
                          dropout=self.drop_prob, use_checkpoint=self.use_checkpoint)
            for _ in range(self.encode_layer_depth)])
        

    def get_decoder(self, decode_out_channel, decode_out_act, decode_fn_str_list):
        decoder_block_list = []
        decoder_layer_up_list = []
        for decode_idx in range(self.decode_layer_depth):
            common_kwarg_dict = self.get_common_kwarg_dict(self.image_shape * (2 ** decode_idx))
            decode_block_in_channel = self.embed_dim // (2 ** decode_idx)
            decode_block_out_channel = decode_block_in_channel // 2
            decoder_block = ResNetBlockND(decode_block_in_channel, decode_block_out_channel, **common_kwarg_dict)
            del common_kwarg_dict["kernel_size"]
            del common_kwarg_dict["padding"]
            decoder_layer_up = MultiDecoderND_V2(decode_block_out_channel, decode_block_out_channel,
                                                kernel_size=2, decode_fn_str_list=decode_fn_str_list,
                                                use_residual_conv=True, **common_kwarg_dict)
            decoder_block_list.append(decoder_block)
            decoder_layer_up_list.append(decoder_layer_up)
        decoder_block_list = nn.ModuleList(decoder_block_list)
        decoder_layer_up_list = nn.ModuleList(decoder_layer_up_list)
        
        common_kwarg_dict = self.get_common_kwarg_dict(self.image_shape * self.patch_size)
        decode_final_conv = nn.Sequential(
            ResNetBlockND(decode_block_out_channel, decode_block_out_channel, **common_kwarg_dict),
            OutputND(decode_block_out_channel, decode_out_channel, act=decode_out_act, img_dim=self.img_dim),
        )
        return nn.ModuleList([decoder_block_list, decoder_layer_up_list, decode_final_conv])

    def get_validity_block(self, validity_shape, validity_act):
        validity_init_channel = self.embed_dim // 2

        common_kwarg_dict = self.get_common_kwarg_dict(self.image_shape)
        validity_conv_1 = ResNetBlockND(self.embed_dim, validity_init_channel,
                                        **common_kwarg_dict)
        validity_conv_2 = ResNetBlockND(validity_init_channel,
                                            validity_init_channel // 2,
                                            **common_kwarg_dict)
        validity_conv_3 = ResNetBlockND(validity_init_channel // 2,
                                            validity_init_channel // 2,
                                            **common_kwarg_dict)
        gap_layer = None
        if self.img_dim == 1:
            gap_layer = nn.AdaptiveAvgPool1d
        elif self.img_dim == 2:
            gap_layer = nn.AdaptiveAvgPool2d
        elif self.img_dim == 3:
            gap_layer = nn.AdaptiveAvgPool3d
        validity_avg_pool = gap_layer(validity_shape[1:])
        validity_final_conv = ConvBlockND(validity_init_channel // 2, validity_shape[0],
                                                kernel_size=1, act=validity_act, norm=None, dropout_proba=0.0)
        validity_block = nn.Sequential(
            validity_conv_1,
            validity_conv_2,
            validity_conv_3,
            validity_avg_pool,
            validity_final_conv,
        )
        return validity_block
    
    def get_attn_info(self, attn_info, num_heads=None, dim_head=32):
        if attn_info is None:
            return None
        elif attn_info is False:
            return {"num_heads": num_heads, "dim_head": dim_head, "full_attn": False}
        elif attn_info is True:
            return {"num_heads": num_heads, "dim_head": dim_head, "full_attn": True}
    
    def get_attn_layer(self, dim, num_heads, dim_head, use_full_attn, use_checkpoint):
        if use_full_attn is True:
            return AttentionBlock(channels=dim, num_heads=num_heads,
                                  use_checkpoint=use_checkpoint)
        elif use_full_attn is False:
            return LinearAttention(dim=dim, num_heads=num_heads, dim_head=dim_head,
                                   use_checkpoint=use_checkpoint)
        else:
            return nn.Identity()

    def get_image_init_shape(self):
        image_shape = np.array(tuple(self.img_size for _ in range(self.img_dim))) // self.patch_size
        return image_shape