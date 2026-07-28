"""MoE-SSF: a mixture of scale/shift (SSF) experts for PROCEED.

Replaces PROCEED's single drift -> (scale, shift) generator (`Bottleneck` in
`adapter/module/generator.py`) with K expert bottlenecks of heterogeneous
capacity plus a shared router and an implicit identity/fallback slot.

Motivation (see documents/MOE_SSF_DESIGN.md):
  * The capacity diagnostic showed the optimal generator capacity is CELL-
    dependent (short horizon wants low rank, most long horizon wants full rank,
    with counterexamples) — no single fixed bottleneck is best everywhere.
  * The reg_combo2 full-grid run showed a fixed recipe "wins small, loses big":
    it is net-negative because it cannot turn adaptation OFF where drift is weak
    / adaptation is harmful (Weather, several long horizons). The single highest-
    value capability is therefore ABSTENTION.

This module addresses both: heterogeneous experts cover the capacity range, and
the fallback slot gives the router a first-class "do not adapt" action (routing
mass to it shrinks the adaptation toward the zero-init identity == zero-shot TTM).

MVP scope: the router conditions on the DRIFT vector only. Horizon is NOT an
input because in the current single-(dataset, seq_len, pred_len) pipeline pred_len
is constant within a run, so a horizon feature carries no within-run signal.
Horizon conditioning becomes meaningful only under multi-horizon joint training
(see MOE_SSF_DESIGN.md "Future work"). Routing is dense/soft (no top-k) since SSF
experts are cheap and dense gating sidesteps discrete-routing collapse.
"""
import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from adapter.module import down_up
from adapter.module.base import Adaptation
from adapter.module.generator import Bottleneck, clip


# --------------------------------------------------------------------------- #
# Auxiliary losses
# --------------------------------------------------------------------------- #
def importance_loss(expert_gates: torch.Tensor) -> torch.Tensor:
    """Load-balancing via squared coefficient of variation of per-expert
    importance (Shazeer et al. 2017), computed over the K REAL experts only.

    The fallback slot is deliberately EXCLUDED so the router is free to abstain
    (weight fallback arbitrarily high) without penalty — we only prevent one real
    expert from starving the others (dead-expert collapse).

    expert_gates: (B, K) soft gates for the real experts.
    """
    importance = expert_gates.sum(0)                      # (K,)
    eps = 1e-10
    return importance.var(unbiased=False) / (importance.mean() ** 2 + eps)


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE router z-loss (Zoph et al. 2022): keeps router logits from blowing
    up, which stabilizes online training. logits: (B, num_slots)."""
    return (torch.logsumexp(logits, dim=-1) ** 2).mean()


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
class ExpertRouter(nn.Module):
    """Shared soft router: drift -> distribution over K experts + 1 fallback slot.

    Returns (expert_gates, gates, logits):
      * expert_gates (B, K): weights for the real experts (fed to MoEBottleneck).
      * gates (B, K+1): full softmax incl. fallback (for aux losses / logging).
      * logits (B, K+1): pre-softmax (for z-loss).
    Any mass on the last slot is the fallback: it multiplies a ZERO expert output,
    so it shrinks the adaptation toward identity.
    """
    def __init__(self, concept_dim: int, num_experts: int, hidden_dim: int = 0,
                 noisy_std: float = 0.0, ema_decay: float = 0.99, log_every: int = 0):
        super().__init__()
        self.num_experts = num_experts
        self.num_slots = num_experts + 1                 # +1 = fallback / identity
        self.noisy_std = noisy_std
        self.ema_decay = ema_decay
        self.log_every = log_every                       # >0 = print running usage every N fwd calls
        if hidden_dim and hidden_dim > 0:
            self.gate = nn.Sequential(nn.Linear(concept_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, self.num_slots))
            last = self.gate[-1]
        else:
            self.gate = nn.Linear(concept_dim, self.num_slots)
            last = self.gate
        # Start near-uniform routing; symmetry between experts is broken by their
        # (random) parameter init, not by the router.
        nn.init.zeros_(last.bias)
        nn.init.normal_(last.weight, std=1e-3)
        # instrumentation only (does not affect grads)
        self.register_buffer('usage_ema', torch.ones(self.num_slots) / self.num_slots,
                             persistent=False)
        self.register_buffer('gate_sum', torch.zeros(self.num_slots), persistent=False)
        self.register_buffer('argmax_count', torch.zeros(self.num_slots), persistent=False)
        self.register_buffer('sample_count', torch.zeros(1), persistent=False)
        self._calls = 0

    def reset_usage(self):
        self.gate_sum.zero_(); self.argmax_count.zero_(); self.sample_count.zero_()
        self._calls = 0

    def usage_summary(self) -> str:
        """Human-readable routing summary accumulated since the last reset. Slots
        are [expert_0 .. expert_{K-1}, FALLBACK]."""
        n = float(self.sample_count.item())
        if n == 0:
            return "no samples routed yet"
        mean_gate = (self.gate_sum / n).tolist()
        argmax_frac = (self.argmax_count / n).tolist()
        names = [f"e{i}" for i in range(self.num_experts)] + ["FALLBACK"]
        mg = ", ".join(f"{nm}={v:.3f}" for nm, v in zip(names, mean_gate))
        am = ", ".join(f"{nm}={v:.3f}" for nm, v in zip(names, argmax_frac))
        return (f"samples={int(n)} | mean gate [{mg}] | argmax-frac [{am}] | "
                f"abstain(mean fallback gate)={mean_gate[-1]:.3f}")

    def forward(self, drift: torch.Tensor):
        logits = self.gate(drift)                        # (B, num_slots)
        if self.training and self.noisy_std > 0:
            logits = logits + torch.randn_like(logits) * self.noisy_std
        gates = F.softmax(logits, dim=-1)                # (B, num_slots)
        expert_gates = gates[:, :self.num_experts]       # (B, K)
        with torch.no_grad():
            bsz = gates.shape[0]
            self.usage_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * gates.mean(0))
            self.gate_sum += gates.sum(0)
            self.argmax_count += torch.bincount(gates.argmax(-1), minlength=self.num_slots)
            self.sample_count += bsz
            self._calls += 1
            if self.log_every > 0 and self._calls % self.log_every == 0:
                print(f"[MoE router @ call {self._calls}] {self.usage_summary()}", flush=True)
        return expert_gates, gates, logits


# --------------------------------------------------------------------------- #
# Mixture bottleneck (drop-in for generator.Bottleneck)
# --------------------------------------------------------------------------- #
class MoEBottleneck(nn.Module):
    """K expert `Bottleneck`s combined by per-sample gates from a shared router.

    Output shape matches a single Bottleneck: (n_layers, B, out_dim), so the rest
    of the PROCEED pipeline (Down_Up.assign_adaptation) is unchanged.
    """
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, bottleneck_dims,
                 activation=nn.LeakyReLU, need_bias: bool = False, shared: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.need_bias = need_bias
        self.experts = nn.ModuleList([
            Bottleneck(in_dim, out_dim, n_layers, int(bd), activation,
                       need_bias=need_bias, shared=shared)
            for bd in bottleneck_dims
        ])

    def forward(self, drift: torch.Tensor, expert_gates: torch.Tensor):
        # drift: (B, in_dim); expert_gates: (B, K)
        outs = torch.stack([e(drift) for e in self.experts], dim=0)   # (K, n_layers, B, out_dim)
        g = expert_gates.permute(1, 0).view(expert_gates.shape[1], 1, expert_gates.shape[0], 1)
        return (outs * g).sum(0)                                       # (n_layers, B, out_dim)


# --------------------------------------------------------------------------- #
# Mixture generator (drop-in for generator.AdaptGenerator)
# --------------------------------------------------------------------------- #
class MoEAdaptGenerator(nn.Module):
    """Mirrors `AdaptGenerator` but each per-key generator is a `MoEBottleneck`,
    and a single shared `ExpertRouter` produces the gates once per forward.

    Exposes `self.aux_loss` (load-balancing + z-loss, scaled by lb_coef / z_coef)
    for the training loop to add, and `self.last_gates` for logging.
    """
    def __init__(self, backbone: nn.Module, concept_features: int, activation=nn.Sigmoid,
                 shared: bool = True, need_bias: bool = True, mid_dim: int = None,
                 num_experts: int = 2, expert_bottleneck_dims=None,
                 router_hidden_dim: int = 0, router_noisy_std: float = 0.0,
                 lb_coef: float = 0.01, z_coef: float = 1e-3, router_log_every: int = 0,
                 router_use_norm: bool = True):
        super().__init__()
        self.dim_name_dict = collections.defaultdict(list)
        self.bottlenecks = nn.ModuleDict()
        self.lb_coef = lb_coef
        self.z_coef = z_coef
        self.router_use_norm = router_use_norm
        self.register_buffer('aux_loss', torch.zeros(()), persistent=False)
        self.last_gates = None

        # Same key/grouping logic as AdaptGenerator.
        for name, module in backbone.named_modules():
            if isinstance(module, Adaptation):
                _weight = module.affine_weight if hasattr(module, 'affine_weight') else module.weight
                out_features = _weight.shape[1 if isinstance(module, transformers.Conv1D) else 0]
                if _weight.dim() == 1:
                    in_features = 1
                else:
                    in_features = _weight.shape[0 if isinstance(module, transformers.Conv1D) else 1]
                out_dim = out_features
                if module.bias is not None:
                    out_dim += out_features
                if isinstance(module, down_up.Down_Up):
                    out_dim += in_features
                self.dim_name_dict[name.split('.')[-1] + '_' + str(out_dim)].append(name)

        dims = self._resolve_expert_dims(expert_bottleneck_dims, mid_dim, num_experts)
        self.expert_bottleneck_dims = dims
        self.num_experts = len(dims)
        # The router conditions on the drift direction plus, optionally, the
        # pre-clip drift magnitude ||drift|| (a strong abstention signal: weak
        # drift -> little to adapt). The clip() applied to the experts' input
        # discards that magnitude, so we feed it to the router explicitly.
        router_in_dim = concept_features + (1 if router_use_norm else 0)
        self.router = ExpertRouter(router_in_dim, self.num_experts,
                                   hidden_dim=router_hidden_dim, noisy_std=router_noisy_std,
                                   log_every=router_log_every)

        for key, names in self.dim_name_dict.items():
            out_dim = int(key.split('_')[-1])
            self.bottlenecks[key] = MoEBottleneck(concept_features, out_dim, len(names), dims,
                                                  activation, need_bias=need_bias, shared=shared)

    @staticmethod
    def _resolve_expert_dims(expert_bottleneck_dims, mid_dim, num_experts):
        """If explicit dims are given, they define the experts (num_experts is
        ignored). Otherwise geometrically span [max(4, mid//8) .. mid]."""
        if expert_bottleneck_dims:
            if isinstance(expert_bottleneck_dims, str):
                dims = [int(x) for x in expert_bottleneck_dims.split(',') if x.strip()]
            else:
                dims = [int(x) for x in expert_bottleneck_dims]
            return dims
        mid = int(mid_dim) if mid_dim else 32
        lo = max(4, mid // 8)
        if num_experts <= 1:
            return [mid]
        dims = sorted({int(round(lo * (mid / lo) ** (i / (num_experts - 1))))
                       for i in range(num_experts)})
        return dims

    def forward(self, x, need_clip=False):
        if need_clip:
            x, x_norm = clip(x)          # x_norm = ||drift|| BEFORE clipping, shape (B, 1)
        else:
            x_norm = x.norm(dim=-1, keepdim=True)
        router_in = torch.cat([x, x_norm], dim=-1) if self.router_use_norm else x
        expert_gates, gates, logits = self.router(router_in)
        coefs = {k: bn(x, expert_gates) for k, bn in self.bottlenecks.items()}
        self.aux_loss = (self.lb_coef * importance_loss(expert_gates)
                         + self.z_coef * router_z_loss(logits))
        self.last_gates = gates.detach()
        return coefs


# --------------------------------------------------------------------------- #
# Self-test (core math only — no backbone required):  python -m adapter.module.moe
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    torch.manual_seed(0)
    B, concept_dim, out_dim, n_layers, K = 5, 16, 12, 3, 3
    router = ExpertRouter(concept_dim, K)
    moe = MoEBottleneck(concept_dim, out_dim, n_layers, [4, 8, 32])
    drift = torch.randn(B, concept_dim)

    eg, gates, logits = router(drift)
    assert gates.shape == (B, K + 1)
    assert torch.allclose(gates.sum(-1), torch.ones(B), atol=1e-5), "gates must be a distribution"
    out = moe(drift, eg)
    assert out.shape == (n_layers, B, out_dim), out.shape

    # Fallback check: force all logit mass to the fallback slot -> ~zero adaptation.
    forced_gates = torch.zeros(B, K + 1); forced_gates[:, -1] = 1.0
    out_fb = moe(drift, forced_gates[:, :K])
    assert out_fb.abs().max() < 1e-6, "full fallback must yield ~zero (identity) adaptation"

    # Losses finite and non-negative.
    lb = importance_loss(eg); z = router_z_loss(logits)
    assert torch.isfinite(lb) and torch.isfinite(z) and lb >= 0 and z >= 0

    # Usage tracking: reset -> route a few batches -> summary reflects sample count.
    router.reset_usage()
    for _ in range(4):
        router(torch.randn(B, concept_dim))
    assert int(router.sample_count.item()) == 4 * B
    print("[router usage]", router.usage_summary())
    print(f"[moe self-test OK] gates.sum≈1, out {tuple(out.shape)}, "
          f"fallback_max {out_fb.abs().max():.2e}, lb {lb:.4f}, z {z:.4f}")
    print("auto expert-dim span (mid=32, K=3):",
          MoEAdaptGenerator._resolve_expert_dims(None, 32, 3))
