"""Exp_ProceedMoE: PROCEED-MoE online experiment runner.

Identical to Exp_Proceed except it (a) wraps the backbone with `ProceedMoE`
instead of `Proceed`, and (b) adds the router's auxiliary loss (load-balancing +
z-loss) to the training objective.

Selected via `--online_method ProceedMoE`. Because the lowercased method name
still contains "proceed", all of run.py's Proceed setup (checkpoint naming, freeze
handling, etc.) applies unchanged.
"""
import torch

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

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        aux = getattr(self._model.generator, 'aux_loss', None)
        if aux is not None and torch.is_tensor(aux):
            loss = loss + aux.to(loss.device)
        return loss

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        # Reset routing accumulators per phase, then report how the router split
        # its mass over {experts..., FALLBACK} at the end — the key diagnostic for
        # whether the mixture concentrates / abstains sensibly.
        router = self._model.generator.router
        router.reset_usage()
        ret = super().online(online_data, target_variate, phase, show_progress)
        print(f'[MoE router usage | phase={phase}] {router.usage_summary()}', flush=True)
        return ret
