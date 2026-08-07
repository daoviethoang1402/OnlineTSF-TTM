import torch
import torch.nn as nn

from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction


class Model(nn.Module):
    """
    PatchTST-FM wrapper for the PROCEED framework.

    PatchTST-FM (``ibm-research/patchtst-fm-r1``) is a ~258M-param, channel-
    INDEPENDENT time-series foundation model. It forecasts by MASKED
    RECONSTRUCTION at a fixed context_length of 8192: a short input series is
    left-padded/masked and the forecast horizon is appended as a masked region
    that the model fills in. Consequently ANY (context, horizon) with
    ``context + horizon <= 8192`` is supported by the *existing* weights -- no
    new prediction head is initialised. The model emits 99 quantiles; we take
    the median (0.5) as the point forecast for MSE/MAE.

    Freezing policy (mirrors ``models/TinyTimeMixer.py``):
      - transformer core (``backbone.blocks`` + ``pos_embed`` + ``norm_fn``):
        ALWAYS frozen -- the FM "backbone" (~250M params), per FM convention.
      - read-in / read-out projections (``backbone.in_layer``,
        ``backbone.out_layer``): the "head". Trainable during pretraining when
        ``args.freeze=False`` (approach 1: fine-tune the projections), frozen
        when ``args.freeze=True`` (approach 2: adapter-only). ``args.freeze_online``
        permanently freezes them before val/online via ``freeze_head()``.
      - PROCEED's ``add_adapters_()`` wraps every ``nn.Linear`` with a Down_Up
        adapter and freezes the base weights; ``post_proceed_init()`` re-enforces
        the core freeze after injection.

    Calling convention (matches TTM): ``forward(x, x_mark=None)`` with
    ``x = (B, seq_len, C)`` -> point forecast ``(B, pred_len, C)``.

    NOTE ON COST: every forward runs the transformer at the full 8192 context
    (512 patches x 20 layers x d_model 1024), so it is ~10x heavier than TTM per
    step -- relevant for the sequential online loop.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self._median_q = 0.5

        name = getattr(configs, 'pretrained_model_name', 'ibm-research/patchtst-fm-r1')
        kw = {}
        if getattr(configs, 'run_offline', False):
            kw['local_files_only'] = True
        rev = getattr(configs, 'revision', None)
        if rev and rev != 'main':
            kw['revision'] = rev
        self.model = PatchTSTFMForPrediction.from_pretrained(name, **kw)

        # ---- freeze policy ------------------------------------------------
        self._freeze_online = getattr(configs, 'freeze_online', False)
        self._head_trainable = not getattr(configs, 'freeze', False)
        self._head_frozen_for_online = False
        for m in self._core_modules():
            m.requires_grad_(False)
        if not self._head_trainable:
            for m in self._head_modules():
                m.requires_grad_(False)

    # ------------------------------------------------------------------
    # Module groups (core = always frozen; head = the trainable projections)
    # ------------------------------------------------------------------
    def _core_modules(self):
        bb = self.model.backbone
        return [bb.blocks, bb.pos_embed, bb.norm_fn]

    def _head_modules(self):
        bb = self.model.backbone
        return [bb.in_layer, bb.out_layer]

    # ------------------------------------------------------------------
    # Hooks called by Proceed and Exp_Proceed (mirror TTM)
    # ------------------------------------------------------------------
    def post_proceed_init(self):
        """Re-enforce the core freeze after PROCEED's add_adapters_() runs."""
        for m in self._core_modules():
            m.requires_grad_(False)

    def freeze_head(self):
        """Permanently freeze the read-in/out projections before val/online,
        iff args.freeze_online. No-op otherwise (they keep fine-tuning)."""
        if not self._freeze_online:
            return
        for m in self._head_modules():
            m.requires_grad_(False)
        self._head_frozen_for_online = True

    def requires_grad_(self, requires_grad: bool = True):
        """Override so the transformer core is never trainable regardless of
        caller; head respects _head_frozen_for_online and _head_trainable."""
        if requires_grad:
            if not self._head_frozen_for_online and self._head_trainable:
                for m in self._head_modules():
                    m.requires_grad_(True)
        else:
            for m in self._head_modules():
                m.requires_grad_(False)
        for m in self._core_modules():
            m.requires_grad_(False)
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, x_mark=None, return_emb=False):
        # x: (B, L, C). PatchTST-FM is channel-independent and builds its masks
        # on CPU internally (device bug in forecast_single_step), so we pass a
        # list of univariate CPU series; the backbone moves them to its device.
        B, L, C = x.shape
        xc = x.detach().to('cpu')
        series = [xc[b, :, c] for b in range(B) for c in range(C)]
        out = self.model(inputs=series, prediction_length=self.pred_len,
                         quantile_levels=[self._median_q])
        qp = out.quantile_predictions            # (B*C, 1, pred_len) on model device
        med = qp[:, 0, :].reshape(B, C, self.pred_len).permute(0, 2, 1).contiguous()
        return med
