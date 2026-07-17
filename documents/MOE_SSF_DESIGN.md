# MoE-SSF: Mixture of Scale/Shift Experts for PROCEED + TTM

Branch: `moe-ssf` (off `with-tsfm`). Status: **MVP scaffold — not yet validated on a real run.**

## Why (motivation from the hyperparameter phase)

Two results from `documents/PROCEED_TTM_HYPERPARAM_NOTES.md` make a single fixed
adapter insufficient:

1. **Optimal generator capacity is cell-dependent.** The capacity diagnostic
   showed short horizon prefers a low-rank adapter (`bottleneck_dim≈8`), most
   long horizons prefer full rank (`≈32`), with counterexamples (ETTh2 1536_720
   wanted low). No single fixed bottleneck is best everywhere.
2. **A fixed recipe "wins small, loses big."** reg_combo2 on the full 6-dataset
   grid: 32 wins avg −1.98% vs 40 losses avg +2.64% — net slightly negative,
   because it cannot turn adaptation OFF where drift is weak / adaptation hurts
   (Weather, several long horizons).

MoE-SSF targets both: **heterogeneous-capacity experts** cover the capacity
range, and an **identity/fallback slot** gives the router a first-class
"do not adapt" action. The single highest-value capability is **abstention** —
converting the big-loss tails into ~zero-shot ties.

## Architecture (MVP)

Replaces PROCEED's single `AdaptGenerator` (one `Bottleneck` per layer-group that
maps `drift -> (scale, scale2, shift)`) with a mixture:

```
drift (B, concept_dim)
      │
      ├──────────────► ExpertRouter ──► softmax over K+1 slots
      │                                   ├─ expert_gates (B, K)
      │                                   └─ fallback slot (B, 1)  ← implicit zero expert
      │
      ▼  (per layer-group key)
 MoEBottleneck = { Bottleneck_1(bd=8), Bottleneck_2(bd=32), ... }   # K experts, heterogeneous rank
      │
   adaptation = Σ_k expert_gates[:,k] · Bottleneck_k(drift)         # dense/soft combine
```

- **Experts** (`MoEBottleneck`): K `Bottleneck`s of different `bottleneck_dim`.
  Each outputs `(n_layers, B, out_dim)` exactly like the original, so
  `Down_Up.assign_adaptation` and `Proceed.forward` are unchanged.
- **Fallback/identity slot**: the router emits `K+1` softmax gates; only the
  first `K` are applied. Mass on slot `K+1` multiplies a *zero* expert → the
  adaptation shrinks toward the zero-init identity (`scale=1, shift=0` ==
  zero-shot TTM). No parameters; free zero-shot fallback.
- **Router** (`ExpertRouter`): linear (or 1-hidden-MLP) `drift -> K+1` logits,
  softmax. **Dense/soft routing, no top-k** — SSF experts are cheap, and dense
  gating avoids discrete-routing collapse. Near-uniform init (small logit
  weights); expert symmetry is broken by the experts' own random init.
- **Aux losses**: `importance_loss` (CV² of per-expert importance, over the K
  *real* experts only — fallback excluded so abstention is unpenalized) +
  `router_z_loss` (ST-MoE logit regularizer). Exposed as
  `generator.aux_loss`, added to the objective in `Exp_ProceedMoE.train_loss`.
- **Instrumentation**: `router.usage_ema` (per-slot usage EMA) and
  `generator.last_gates` for logging how much the router abstains vs. uses each
  expert.

### Key MVP scoping decision: router conditions on DRIFT only (not horizon)

The capacity/LR preference flips with horizon, so horizon is a natural routing
signal *across* runs. But the current pipeline trains one model per
`(dataset, seq_len, pred_len)`, so **within a run `pred_len` is constant** and a
horizon feature carries no signal. The MVP router therefore conditions on the
per-instance **drift** vector (which does vary within a run). The value within a
run is per-instance **abstention** — exactly what the "wins small, loses big"
result demands. Horizon routing is deferred to multi-horizon joint training
(see Future work).

## Files

| file | role |
|---|---|
| `adapter/module/moe.py` | `ExpertRouter`, `MoEBottleneck`, `MoEAdaptGenerator`, `importance_loss`, `router_z_loss` + a no-backbone self-test (`python3 -m adapter.module.moe`) |
| `adapter/proceed_moe.py` | `ProceedMoE(Proceed)` — swaps in the MoE generator, overrides `freeze_adapter/freeze_bias` to iterate experts + router |
| `exp/exp_proceed_moe.py` | `Exp_ProceedMoE(Exp_Proceed)` — wraps with `ProceedMoE`, adds `aux_loss` to `train_loss` |
| `exp/__init__.py` | exports `Exp_ProceedMoE` |
| `run.py` | CLI flags (see below) |
| `ttm-scripts/moe-ssf/ETTh1_smoke.sh` | 2-cell smoke test |

## CLI flags (`--online_method ProceedMoE`)

| flag | default | meaning |
|---|---|---|
| `--num_experts` | 2 | K real experts (ignored if `--expert_bottleneck_dims` set) |
| `--expert_bottleneck_dims` | `''` | explicit per-expert `bottleneck_dim`, e.g. `8,32`; empty = geom-span `[max(4, bottleneck_dim//8) .. bottleneck_dim]` |
| `--router_hidden_dim` | 0 | 0 = linear router; >0 = 1-hidden-layer MLP |
| `--router_noisy_std` | 0.0 | Gaussian noise on router logits during training (exploration) |
| `--moe_lb_coef` | 0.01 | load-balancing (importance CV²) coefficient |
| `--moe_z_coef` | 1e-3 | router z-loss coefficient |

Existing flags still apply: `--concept_dim` sizes the drift/router input,
`--bottleneck_dim` is the *upper* end of the auto expert span, `--freeze_online`,
`--online_learning_rate` (use the tuned 3e-6), `--ema` (inert).

## How to run

```bash
# 1) core-math self-test (torch env, no GPU needed) — now also checks usage tracking
python3 -m adapter.module.moe
# 2) end-to-end smoke test (2 corner cells)
bash ttm-scripts/moe-ssf/ETTh1_smoke.sh
# 3) diagnostics
bash ttm-scripts/moe-ssf/ETTh1_512_96_lbcoef.sh   # lb_coef in {0, 0.001}: does the router concentrate?
bash ttm-scripts/moe-ssf/ETTh1_1536_720.sh        # hold-out rerun (needs an IDLE gpu)
```

### Router usage logging

`ExpertRouter` accumulates mean gates / argmax fractions / abstention (mean
fallback gate) per phase. `Exp_ProceedMoE.online` resets at the start of each
phase and prints a summary at the end:

```
[MoE router usage | phase=test] samples=... | mean gate [e0=.., e1=.., FALLBACK=..] | argmax-frac [...] | abstain(mean fallback gate)=..
```

Add `--moe_log_every N` to also print running usage every N router calls during
the loop. Slots are `[expert_0 .. expert_{K-1}, FALLBACK]`.

## Results so far

| cell | config | MSE | vs zero-shot | vs reg_combo2 (fixed) |
|---|---|---|---|---|
| ETTh1 512_96 | experts 8,32; lb=0.01; lr3e-6 | 0.4725 | −1.8% (beats 0.481) | +1.4% (loses to 0.466) |

Read: at 512_96 a *single* capacity (bneck8) is optimal, so the mixture dilutes
and the lb loss (forcing ~50/50) likely blocks concentration — hence the
lb_coef diagnostic. The informative cells (multi-capacity / hold-out /
abstention) are still pending (1536_720 OOM'd on a contended GPU).

## What to look for in the smoke run

1. **It runs** end-to-end and prints the `[ProceedMoE] experts=... bottleneck_dims=...` banner.
2. **Router behavior** (`usage_ema` / `last_gates`): does the router differentiate
   the two cells? Expectation — 512_96 leans on the low-rank expert with little
   fallback; 1536_720 leans on the full-rank expert and/or the fallback. If the
   router collapses to one slot everywhere, revisit init / `moe_lb_coef` /
   `router_noisy_std`.
3. **Metrics vs. the reg_combo2 baseline** at the same cells (from
   `results.xlsx` sheet `no-ft`): does per-instance routing beat the fixed recipe,
   especially recovering the 1536_720 hold-out toward zero-shot via abstention?

This smoke run also answers the **router-feasibility** question folded into the
build: if the router learns to abstain on loss-prone inputs, a feed-forward
router suffices; if it cannot separate them from drift alone, switch to an online
performance guard (see Future work).

## Integration seams / caveats (need a real run to validate)

- **`aux_loss` hook**: added via `Exp_ProceedMoE.train_loss`, which covers the
  standard `_update` path (batch dim==3, which the frozen TTM runs use). The
  leakage/per-sample inline-loss path in `Exp_Online._update_online` (batch
  dim==4) does NOT go through `train_loss` — if that mode is ever used, the aux
  loss must be added there too.
- **Freezing policy for the router**: currently treated as an adapter *weight*
  (trained on the current-batch phase, frozen on the recent-batch phase). This
  is a default choice, not yet validated — the router may prefer to update on
  both phases, or on a slower schedule.
- **`flag_basic` path** in `Proceed.forward` (reads `bottleneck.biases`) is not
  MoE-aware; unused by default but would break under `flag_basic=True`.
- **Optimizer**: all router/expert params are `requires_grad=True` at build, so
  `_select_optimizer` includes them (same as base Proceed); freeze toggles only
  gate grad flow afterward.

## Future work (in priority order)

1. **Abstention analysis** — quantify how often the router picks fallback per
   dataset/horizon; correlate with the reg_combo2 win/loss map. This is the core
   thesis evidence.
2. **Horizon-conditioned routing via multi-horizon training** — train one model
   over multiple `pred_len` so horizon becomes a real routing input (the axis
   along which capacity/LR preference flips).
3. **Online performance guard** — fallback path that compares recent adapted vs.
   zero-shot error and backs off (SOLID-style), for when a feed-forward router
   cannot separate win/loss inputs from drift alone.
4. **Per-expert effective step size** — the LR×capacity interaction (full-rank
   experts tolerate larger LR) suggests experts may want distinct update scales.
5. **Top-k / sparse routing** — only if K grows large enough that dense routing
   cost matters (unlikely for SSF).
