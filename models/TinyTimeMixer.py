__all__ = ['Model']

import torch
import torch.nn as nn

from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction

# Maps the framework's --freq strings to TTM-r2 integer tokens.
# TTM-r2 frequency_token_vocab_size=8, so valid range is [0, 7].
# 't' (Informer convention for ETTm 15-min data) maps to token 5 (15min).
# Frequencies outside the vocab (daily, weekly) fall back to 0 (oov).
_FREQ_MAP = {
    't': 5,      # ~15-min (ETTm convention in this codebase)
    'min': 1,
    '2min': 2,
    '5min': 3,
    '10min': 4,
    '15min': 5,
    '30min': 6,
    'h': 7, 'H': 7,
}

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

        pretrained_model_name = getattr(configs, 'pretrained_model_name', 'ibm-research/ttm-research-r2')
        cfg = TinyTimeMixerConfig.from_pretrained(
            pretrained_model_name,
            revision = configs.revision)
        cfg.context_length = configs.seq_len
        cfg.prediction_length = configs.pred_len
        cfg.num_input_channels = configs.enc_in
        cfg.decoder_mode = configs.decoder_mode
        cfg.mode = configs.backbone_mode
        self.tinytimemixer = TinyTimeMixerForPrediction.from_pretrained(
                pretrained_model_name,
                config=cfg,
                ignore_mismatched_sizes=True,
                local_files_only=configs.run_offline,
                revision = configs.revision
            )

        # Frequency token for resolution_prefix_tuning (TTM-r2 feature).
        # Stored as a non-persistent buffer so it moves with the model to GPU.
        freq_int = _FREQ_MAP.get(getattr(configs, 'freq', 'h'), 0)
        self.register_buffer('_freq_token',
                             torch.tensor(freq_int, dtype=torch.long),
                             persistent=False)

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
        freq_token = self._freq_token.expand(x.shape[0])
        outputs = self.tinytimemixer(x, return_loss=False, return_dict=True,
                                     freq_token=freq_token)
        if hasattr(outputs, 'prediction_outputs'):
            return outputs.prediction_outputs
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        return outputs
