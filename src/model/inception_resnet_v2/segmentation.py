from .base_model_2d import InceptionResNetV2 as InceptionResNetV22D
from torch import nn
from .transformer_layers import PositionalEncoding
from base_model_2d import ConvBlock2D
from reformer_pytorch import Reformer

class SegInceptionResNetV22D(nn.Module):
    def __init__(self, n_input_channels, block_size=16,
                 padding='valid', include_cbam=True, include_context=False,
                skip_connect=True, version="reformer", dropout_proba=0.1
                 ):

        self.skip_connect = skip_connect
        self.base_model = InceptionResNetV22D(n_input_channels=n_input_channels, block_size=block_size,
                                              padding=padding, include_cbam=include_cbam, include_context=include_context,
                                              include_skip_connection=skip_connect)
        feature_channel_num = block_size * 96
        if version == "reformer":
            # TBD
            self.positional_encoding = PositionalEncoding(d_model=32 * 14 * 14,
                                                            dropout=dropout_proba)
            self.transformer_encoder = Reformer(
                dim=feature_channel_num,
                depth=6,
                heads=8,
                lsh_dropout=0.1,
                causal=True
            )        
        elif:
            pass
        else:
            raise Exception("not Supported version")
        