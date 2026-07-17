"""Exp_ProceedMoE: PROCEED-MoE online experiment runner.

Identical to Exp_Proceed except it (a) wraps the backbone with `ProceedMoE`
instead of `Proceed`, and (b) adds the router's auxiliary loss (load-balancing +
z-loss) to the training objective.

Selected via `--online_method ProceedMoE`. Because the lowercased method name
still contains "proceed", all of run.py's Proceed setup (checkpoint naming, freeze
handling, etc.) applies unchanged.
"""
import torch
import torch.optim as optim

from adapter import proceed_moe
from exp.exp_proceed import Exp_Proceed
from exp.exp_online import Exp_Online


class Exp_ProceedMoE(Exp_Proceed):
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
        ret = super().online(online_data, target_variate, phase, show_progress)
        print(f'[MoE router usage | phase={phase}] {router.usage_summary()}', flush=True)
        return ret
