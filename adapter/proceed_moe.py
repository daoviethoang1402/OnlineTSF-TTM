"""ProceedMoE: PROCEED with a Mixture-of-SSF-Experts generator.

Subclasses `Proceed` and swaps its single `AdaptGenerator` for a
`MoEAdaptGenerator` (K heterogeneous-capacity experts + shared router + implicit
identity/fallback slot). Everything else — the Down_Up adapter injection, the
mlp1/mlp2 concept extractors, the recent_batch / ema machinery, the forward
application of adaptations — is inherited unchanged.

See adapter/module/moe.py and documents/MOE_SSF_DESIGN.md for the design.
"""
import torch.nn as nn

from adapter.proceed import Proceed
from adapter.module.moe import MoEAdaptGenerator


class ProceedMoE(Proceed):
    def __init__(self, backbone, args):
        # Proceed.__init__ injects Down_Up adapters into the backbone and builds
        # mlp1/mlp2 + a (single) AdaptGenerator. We reuse all of that and then
        # replace the generator with the MoE version, which re-reads the same
        # injected Adaptation modules.
        super().__init__(backbone, args)
        self.generator = MoEAdaptGenerator(
            self.backbone, args.concept_dim,
            activation=nn.Sigmoid if args.act == 'sigmoid' else nn.Identity,
            shared=not args.individual_generator,
            need_bias=self.more_bias,
            mid_dim=args.bottleneck_dim,
            num_experts=getattr(args, 'num_experts', 2),
            expert_bottleneck_dims=getattr(args, 'expert_bottleneck_dims', ''),
            router_hidden_dim=getattr(args, 'router_hidden_dim', 0),
            router_noisy_std=getattr(args, 'router_noisy_std', 0.0),
            lb_coef=getattr(args, 'moe_lb_coef', 0.01),
            z_coef=getattr(args, 'moe_z_coef', 1e-3),
            router_log_every=getattr(args, 'moe_log_every', 0),
        )
        print(f'[ProceedMoE] experts={self.generator.num_experts} '
              f'bottleneck_dims={self.generator.expert_bottleneck_dims} '
              f'(+1 fallback slot), router_hidden={getattr(args, "router_hidden_dim", 0)}, '
              f'lb_coef={self.generator.lb_coef}, z_coef={self.generator.z_coef}')

    # ---- freezing -------------------------------------------------------- #
    # Proceed's freeze_adapter/freeze_bias reach into `bottleneck.weights` /
    # `bottleneck.biases`, which MoEBottleneck does not expose directly (it holds
    # a list of expert Bottlenecks). Override to iterate the experts, and treat
    # the router like an adapter weight (trained on the current-batch phase,
    # frozen on the recent-batch phase — same policy as the generator weights).
    def freeze_adapter(self, freeze=True):
        for module_name in ['mlp1', 'mlp2']:
            if hasattr(self, module_name):
                getattr(self, module_name).requires_grad_(not freeze)
                getattr(self, module_name).zero_grad(set_to_none=True)
        self.generator.router.requires_grad_(not freeze)
        self.generator.router.zero_grad(set_to_none=True)
        for moe in self.generator.bottlenecks.values():
            for adapter in moe.experts:
                adapter.weights.requires_grad_(not freeze)
                adapter.weights.zero_grad(set_to_none=True)
                adapter.biases[:len(adapter.weights) - 1].requires_grad_(not freeze)
                adapter.biases[:len(adapter.weights) - 1].zero_grad(set_to_none=True)

    def freeze_bias(self, freeze=True):
        if self.more_bias:
            for moe in self.generator.bottlenecks.values():
                for adapter in moe.experts:
                    adapter.biases[-1].requires_grad_(not freeze)
                    adapter.biases[-1:].zero_grad(set_to_none=True)
