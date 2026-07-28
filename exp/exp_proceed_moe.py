"""Exp_ProceedMoE: PROCEED-MoE online experiment runner.

Identical to Exp_Proceed except it (a) wraps the backbone with `ProceedMoE`
instead of `Proceed`, and (b) adds the router's auxiliary loss (load-balancing +
z-loss) to the training objective.

Selected via `--online_method ProceedMoE`. Because the lowercased method name
still contains "proceed", all of run.py's Proceed setup (checkpoint naming, freeze
handling, etc.) applies unchanged.
"""
import math

import torch
import torch.optim as optim

from adapter import proceed_moe
from exp.exp_proceed import Exp_Proceed
from exp.exp_online import Exp_Online


class Exp_ProceedMoE(Exp_Proceed):
    def __init__(self, args):
        super().__init__(args)
        # ---- online abstention guard -------------------------------------- #
        # A non-gradient control law: on each recent (already-revealed) window we
        # measure the FULL-adaptation error vs the IDENTITY (few-shot) error,
        # keep EMAs of both, and set the model's alpha gate =
        #   sigmoid( ((E_id - E_full) / E_id) / tau ).
        # alpha in (0,1) shrinks the mixture toward identity where adaptation has
        # recently been hurting -> converts the "loses big" tail into ties, while
        # keeping alpha~=1 (full adaptation) where adaptation clearly helps
        # (e.g. Exchange). Uses only past labels -> no leakage. The relative
        # (scale-free) improvement makes a single tau transfer across datasets.
        self.use_guard = getattr(args, 'use_guard', False)
        self.guard_ema = getattr(args, 'guard_ema', 0.95)
        self.guard_tau = getattr(args, 'guard_tau', 0.1)
        self.guard_eps = getattr(args, 'guard_eps', 1e-6)
        self.guard_alpha_min = getattr(args, 'guard_alpha_min', 0.0)
        self.guard_alpha_max = getattr(args, 'guard_alpha_max', 1.0)
        self.E_full = None      # EMA of full-adaptation error on recent windows
        self.E_id = None        # EMA of identity (few-shot) error on recent windows
        self._guard_alpha_sum = 0.0
        self._guard_steps = 0

    def _update_online(self, batch, criterion, optimizer, scaler=None, flag_current=False):
        loss, outputs = super()._update_online(batch, criterion, optimizer, scaler,
                                               flag_current=flag_current)
        # Only recent-window steps (flag_current=False) have revealed labels, so
        # only they may drive the guard. This also warm-starts the EMAs during the
        # validation-update phase before the test phase begins.
        if self.use_guard and not flag_current:
            self._guard_measure(batch, criterion)
        return loss, outputs

    def _guard_measure(self, batch, criterion):
        was_training = self.model.training
        self.model.eval()
        m = self._model
        with torch.no_grad():
            y = batch[self.label_position]
            if not self.args.pin_gpu:
                y = y.to(self.device)
            # full-adaptation error (alpha forced to 1)
            m.flag_identity = False
            m.alpha.fill_(1.0)
            out_f = self.forward(batch)
            if isinstance(out_f, (tuple, list)):
                out_f = out_f[0]
            err_full = criterion(out_f, y).item()
            # identity (few-shot) error
            m.flag_identity = True
            out_i = self.forward(batch)
            if isinstance(out_i, (tuple, list)):
                out_i = out_i[0]
            err_id = criterion(out_i, y).item()
            m.flag_identity = False
        r = self.guard_ema
        self.E_full = err_full if self.E_full is None else r * self.E_full + (1 - r) * err_full
        self.E_id = err_id if self.E_id is None else r * self.E_id + (1 - r) * err_id
        rel = (self.E_id - self.E_full) / (self.E_id + self.guard_eps)   # fractional gain of adapting
        alpha = 1.0 / (1.0 + math.exp(-rel / self.guard_tau))
        alpha = min(self.guard_alpha_max, max(self.guard_alpha_min, alpha))
        m.alpha.fill_(alpha)                    # applied to the NEXT current-window prediction
        self._guard_alpha_sum += alpha
        self._guard_steps += 1
        if was_training:
            self.model.train()

    def _build_model(self, model=None, framework_class=None):
        # Skip Exp_Proceed._build_model (which hardcodes proceed.Proceed) and go
        # straight to the grandparent with our MoE framework class.
        model = Exp_Online._build_model(self, model, framework_class=proceed_moe.ProceedMoE)
        print(model)
        return model

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        # Put the router in its OWN param group so it can use a higher LR than the
        # experts/adapter (which want the tiny online LR). The group is tagged
        # 'is_router' so the online-LR-setting loops (run.py, Exp_Proceed.online)
        # skip it. Without this the router shares the ~3e-6 online LR and stays
        # frozen near its uniform init (see documents/MOE_SSF_DESIGN.md).
        if return_self and self.model_optim is not None:
            return self.model_optim
        router_ids = {id(p) for p in self._model.generator.router.parameters()}
        src = self.model.parameters() if model is None else model.parameters()
        router_params, other_params = [], []
        for p in src:
            if filter_frozen and not p.requires_grad:
                continue
            (router_params if id(p) in router_ids else other_params).append(p)
        rlr = getattr(self.args, 'router_learning_rate', self.args.learning_rate)
        groups = [{'params': other_params, 'lr': self.args.learning_rate}]
        if router_params:
            groups.append({'params': router_params, 'lr': rlr, 'is_router': True})
        opt_name = getattr(self.args, 'optim', 'Adam')
        model_optim = getattr(optim, opt_name)(groups, lr=self.args.learning_rate)
        if return_self:
            self.model_optim = model_optim
        return model_optim

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        aux = getattr(self._model.generator, 'aux_loss', None)
        if aux is not None and torch.is_tensor(aux):
            loss = loss + aux.to(loss.device)
        return loss

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        # Reset routing accumulators per phase, re-assert the router's own LR (in
        # case a training-phase scheduler left it elsewhere), then report how the
        # router split its mass over {experts..., FALLBACK} at the end — the key
        # diagnostic for whether the mixture concentrates / abstains sensibly.
        router = self._model.generator.router
        rlr = getattr(self.args, 'router_learning_rate', None)
        if rlr is not None and self.model_optim is not None:
            for g in self.model_optim.param_groups:
                if g.get('is_router'):
                    g['lr'] = rlr
        router.reset_usage()
        # Reset per-phase guard accumulators, but KEEP E_full/E_id so the guard
        # stays warm from the validation-update phase into the test phase.
        self._guard_alpha_sum = 0.0
        self._guard_steps = 0
        ret = super().online(online_data, target_variate, phase, show_progress)
        print(f'[MoE router usage | phase={phase}] {router.usage_summary()}', flush=True)
        if self.use_guard and self._guard_steps > 0:
            print(f'[MoE guard | phase={phase}] mean alpha={self._guard_alpha_sum / self._guard_steps:.3f} '
                  f'| E_full={self.E_full:.4f} E_id={self.E_id:.4f} steps={self._guard_steps}', flush=True)
        return ret
