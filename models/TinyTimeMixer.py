__all__ = ['Model']

import torch.nn as nn

from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction


def _build_config(configs):
    # Only task dimensions come from args. All architecture constants are fixed to
    # match ibm-granite/granite-timeseries-ttm-r1 so pretrained weights load cleanly.
    # Generic transformer flags in args (d_model=512, patch_len=16, stride=8, …) must
    # NOT be read here — they would cause shape mismatches across the entire backbone.
    return TinyTimeMixerConfig(
        # Task-specific (vary per experiment)
        context_length=configs.seq_len,
        num_input_channels=configs.enc_in,
        prediction_length=configs.pred_len,
        # IBM granite-timeseries-ttm-r1 architecture constants (do not read from args)
        patch_length=64,
        patch_stride=64,
        d_model=192,
        num_layers=2,
        adaptive_patching_levels=3,
        expansion_factor=2,
        dropout=0.2,
        gated_attn=True,
        norm_mlp='LayerNorm',
        self_attn=False,
        self_attn_heads=1,
        use_positional_encoding=False,
        positional_encoding_type='sincos',
        scaling='std',
        loss='mse',
        head_dropout=0.2,
        # Backbone uses common_channel to match IBM pretrained weights.
        # Channel mixing is placed in the decoder (decoder_mode='mix_channel'), which
        # is fine-tuned from scratch and not subject to the backbone-frozen constraint.
        mode=configs.backbone_mode,
        use_decoder=True,
        decoder_num_layers=2,
        decoder_d_model=128,
        decoder_mode=configs.decoder_mode,
        post_init=False,
    )


class Model(nn.Module):
    """
    TinyTimeMixer wrapper for the PROCEED framework.

    Freezing policy (consistent across all phases):
      - tinytimemixer.backbone (TinyTimeMixerModel): ALWAYS frozen — per TTM paper.
      - tinytimemixer.decoder + head: trainable during pretraining when args.freeze=False
        (approach 1: full TTM fine-tuning), frozen when args.freeze=True (approach 2:
        adapter-only training). Both are frozen before val/online via freeze_head().

    The PROCEED adapter wraps the entire model. add_adapters_() may set new Down_Up
    wrapper weights to requires_grad=True; post_proceed_init() re-enforces the backbone
    freeze after adapter injection.
    """

    def __init__(self, configs):
        super().__init__()
        cfg = _build_config(configs)
        pretrained_model_name = getattr(configs, 'pretrained_model_name',
                                        'ibm-granite/granite-timeseries-ttm-r1')
        if pretrained_model_name:
            self.tinytimemixer = TinyTimeMixerForPrediction.from_pretrained(
                pretrained_model_name,
                config=cfg,
                ignore_mismatched_sizes=True,
            )
        else:
            self.tinytimemixer = TinyTimeMixerForPrediction(cfg)

        # Backbone always frozen per TTM paper.
        self.tinytimemixer.backbone.requires_grad_(False)

        # When args.freeze=True (approach 2), freeze decoder+head as well so only
        # PROCEED adapters are trained. When False (approach 1), decoder+head stay
        # trainable for pretraining and are frozen later via freeze_head().
        self._ttm_head_trainable = not getattr(configs, 'freeze', False)
        self._head_frozen_for_online = False
        if not self._ttm_head_trainable:
            self.tinytimemixer.decoder.requires_grad_(False)
            self.tinytimemixer.head.requires_grad_(False)

    # ------------------------------------------------------------------
    # Hooks called by Proceed and Exp_Proceed
    # ------------------------------------------------------------------

    def post_proceed_init(self):
        """Re-enforce backbone freeze after PROCEED's add_adapters_() runs.

        add_adapters_() replaces original layers with Down_Up wrappers whose
        requires_grad follows args.freeze. For approach 1 (args.freeze=False),
        backbone wrappers would be inadvertently set trainable; this corrects that.
        """
        self.tinytimemixer.backbone.requires_grad_(False)

    def freeze_head(self):
        """Freeze decoder+head permanently before val/online phases.

        Called by Exp_Proceed.update_valid() once pretraining is done. After this,
        requires_grad_(True) will no longer re-enable decoder or head.
        """
        self.tinytimemixer.decoder.requires_grad_(False)
        self.tinytimemixer.head.requires_grad_(False)
        self._head_frozen_for_online = True

    # ------------------------------------------------------------------
    # requires_grad_ override
    # ------------------------------------------------------------------

    def requires_grad_(self, requires_grad: bool = True):
        """Override to ensure backbone is never trainable regardless of caller.

        Called by Proceed during update_valid/online to temporarily toggle the
        model's trainability. Backbone always stays frozen; decoder+head respect
        _head_frozen_for_online and _ttm_head_trainable.
        """
        if requires_grad:
            # Restore decoder+head only if not yet frozen for online phases.
            if not self._head_frozen_for_online and self._ttm_head_trainable:
                self.tinytimemixer.decoder.requires_grad_(True)
                self.tinytimemixer.head.requires_grad_(True)
        else:
            self.tinytimemixer.decoder.requires_grad_(False)
            self.tinytimemixer.head.requires_grad_(False)
        # Backbone unconditionally stays frozen.
        self.tinytimemixer.backbone.requires_grad_(False)
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, x_mark=None, return_emb=False):
        outputs = self.tinytimemixer(x, return_loss=False, return_dict=True)
        if hasattr(outputs, 'prediction_outputs'):
            return outputs.prediction_outputs
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        return outputs
