from .base_model_2d import InceptionResNetV2 as InceptionResNetV22D
from .base_model_3d import InceptionResNetV2 as InceptionResNetV23D
from .layers import TransformerEncoder, PositionalEncoding
import torch
from torch import nn
from einops import rearrange


class InceptionResNetV2Transformer3D(nn.Module):
    def __init__(self, n_input_channels, block_size=16,
                 padding='valid', z_channel_preserve=True,
                 dropout_proba=0.1, num_class=2, include_context=False,
                 use_base=False):
        super().__init__()
        self.include_context = include_context
        self.base_model = InceptionResNetV23D(n_input_channels=n_input_channels, block_size=block_size,
                                              padding=padding, z_channel_preserve=z_channel_preserve,
                                              include_context=include_context)

        if self.include_context:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            transformer_layer_list = []
            # current feature shape: [1536 2 14 14]
            # current feature shape: [3072 14 14]
            # current feature shape: [768 28 28]
            # current feature shape: [768 784]
            # current feature shape: [784 768]
            attn_dim_list = [96 for _ in range(6)]
            num_head_list = [8 for _ in range(6)]
            inner_dim = attn_dim_list[0] * num_head_list[0]
            self.positional_encoding = PositionalEncoding(d_model=inner_dim,
                                                          dropout=dropout_proba)
            if not use_base:
                for attn_dim, num_head in zip(attn_dim_list, num_head_list):
                    attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                                    dropout=dropout_proba)
                    transformer_layer_list.append(attn_layer)
                self.transformer_encoder = nn.Sequential(
                    *transformer_layer_list)
            else:

                encoder_layers = nn.TransformerEncoderLayer(d_model=768,
                                                            nhead=num_head_list[0], dim_feedforward=attn_dim_list[0],
                                                            dropout=dropout_proba)
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layers, 6)
            feature_channel_num = 768
        else:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            feature_channel_num = 1536

        self.final_linear_sequence = nn.Sequential(
            nn.Linear(feature_channel_num, 512),  # 512?
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(512, 256),
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(256, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        # shape: (B, 3, 32, 512, 512)
        x = self.base_model(x)
        if self.include_context:
            x = rearrange(x, 'b c z h w -> b (c z) h w')
            x = self.pixel_shuffle(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            x = x.mean(1)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
        output = self.final_linear_sequence(x)

        return output


<<<<<<< HEAD
class InceptionResNetV2Transformer2D(nn.Module):
=======
class InceptionResNetV2Transformer3DWith2D(nn.Module):
>>>>>>> 17c70043fbff6e0b98437e8835775bda952a18c2
    def __init__(self, n_input_channels,
                 padding='valid',
                 dropout_proba=0.1, num_class=2, include_context=False,
                 use_base=False):
        super().__init__()
        self.include_context = include_context
        self.base_model = InceptionResNetV22D(n_input_channels=n_input_channels,
                                              padding=padding,
                                              include_context=include_context)

        if self.include_context:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            transformer_layer_list = []
            # current feature shape: [1536 14 14]
            # current feature shape: [384 28 28]
            # current feature shape: [384 784]
            # current feature shape: [784 384]
            attn_dim_list = [48 for _ in range(6)]
            num_head_list = [8 for _ in range(6)]
            inner_dim = attn_dim_list[0] * num_head_list[0]
            self.positional_encoding = PositionalEncoding(d_model=inner_dim,
                                                          dropout=dropout_proba)
            if not use_base:
                for attn_dim, num_head in zip(attn_dim_list, num_head_list):
                    attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                                    dropout=dropout_proba)
                    transformer_layer_list.append(attn_layer)
                self.transformer_encoder = nn.Sequential(
                    *transformer_layer_list)
            else:

                encoder_layers = nn.TransformerEncoderLayer(d_model=768,
                                                            nhead=num_head_list[0], dim_feedforward=attn_dim_list[0],
                                                            dropout=dropout_proba)
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layers, 6)
            feature_channel_num = 768
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            feature_channel_num = 1536

        self.final_linear_sequence = nn.Sequential(
            nn.Linear(feature_channel_num, 512),  # 512?
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(512, 256),
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(256, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        # shape: (B, 3, 32, 512, 512)
        B, _, Z, _, _ = x.size()
        x = rearrange(x, 'b c z h w -> (b z) c h w')
        x = self.base_model(x)
<<<<<<< HEAD
=======
        x = rearrange(x, '(b z) c h w -> b (z h w) c', z=Z)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(1)

        output = self.final_linear_sequence(x)

        return output


class InceptionResNetV2Transformer2D(nn.Module):
    def __init__(self, n_input_channels,
                 padding='valid',
                 dropout_proba=0.1, num_class=2, include_context=False,
                 use_base=False):
        super().__init__()
        self.include_context = include_context
        self.base_model = InceptionResNetV22D(n_input_channels=n_input_channels,
                                              padding=padding,
                                              include_context=include_context)

        if self.include_context:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            transformer_layer_list = []
            # current feature shape: [1536 14 14]
            # current feature shape: [384 28 28]
            # current feature shape: [384 784]
            # current feature shape: [784 384]
            attn_dim_list = [48 for _ in range(6)]
            num_head_list = [8 for _ in range(6)]
            inner_dim = attn_dim_list[0] * num_head_list[0]
            self.positional_encoding = PositionalEncoding(d_model=inner_dim,
                                                          dropout=dropout_proba)
            if not use_base:
                for attn_dim, num_head in zip(attn_dim_list, num_head_list):
                    attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                                    dropout=dropout_proba)
                    transformer_layer_list.append(attn_layer)
                self.transformer_encoder = nn.Sequential(
                    *transformer_layer_list)
            else:

                encoder_layers = nn.TransformerEncoderLayer(d_model=768,
                                                            nhead=num_head_list[0], dim_feedforward=attn_dim_list[0],
                                                            dropout=dropout_proba)
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layers, 6)
            feature_channel_num = 768
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            feature_channel_num = 1536

        self.final_linear_sequence = nn.Sequential(
            nn.Linear(feature_channel_num, 512),  # 512?
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(512, 256),
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(256, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        # shape: (B, 3, 32, 512, 512)
        x = self.base_model(x)
>>>>>>> 17c70043fbff6e0b98437e8835775bda952a18c2
        if self.include_context:
            x = self.pixel_shuffle(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            x = x.mean(1)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
        output = self.final_linear_sequence(x)

        return output
