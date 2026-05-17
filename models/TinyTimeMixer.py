__all__ = ['Model']

import sys
from pathlib import Path

import torch
import torch.nn as nn

from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction


def _build_config(configs):
    patch_length = getattr(configs, 'patch_len', max(1, configs.seq_len // 8))
    # patch_stride = getattr(configs, 'stride', patch_length)
    patch_stride = patch_length
    return TinyTimeMixerConfig(
        context_length=configs.seq_len,
        patch_length=patch_length,
        num_input_channels=configs.enc_in,
        prediction_length=configs.pred_len,
        patch_stride=patch_stride,
        d_model=getattr(configs, 'd_model', 16),
        num_layers=getattr(configs, 'e_layers', 3),
        dropout=getattr(configs, 'dropout', 0.2),
        mode=getattr(configs, 'mode', 'common_channel'),
        gated_attn=getattr(configs, 'gated_attn', True),
        norm_mlp=getattr(configs, 'norm_mlp', 'LayerNorm'),
        self_attn=getattr(configs, 'self_attn', False),
        self_attn_heads=getattr(configs, 'self_attn_heads', 1),
        use_positional_encoding=getattr(configs, 'use_positional_encoding', False),
        positional_encoding_type=getattr(configs, 'positional_encoding_type', 'sincos'),
        loss=getattr(configs, 'loss', 'mse'),
        use_decoder=getattr(configs, 'use_decoder', False),
        head_dropout=getattr(configs, 'head_dropout', 0.2),
        post_init=False,
    )


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.tinytimemixer = TinyTimeMixerForPrediction.from_pretrained('ibm-granite/granite-timeseries-ttm-r1')

    def forward(self, x, x_mark=None, return_emb=False):
        outputs = self.tinytimemixer(x, return_loss=False, return_dict=True)
        if hasattr(outputs, 'prediction_outputs'):
            return outputs.prediction_outputs
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        return outputs
