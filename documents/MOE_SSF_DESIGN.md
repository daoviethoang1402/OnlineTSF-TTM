# MoE-SSF: Mixture of Scale/Shift Experts for PROCEED + TTM

Branch: `moe-ssf` (off `with-tsfm`).
Status: **v2 — capacity experts + `‖drift‖` router + online abstention guard.**
The v1 scaffold (soft mixture + learned fallback slot) is documented below as
historical record; the sections from "## v2" down to "## Why" describe the
current design. v1's key negative result — *a gradient-trained softmax router
never selects the identity/fallback slot* — is what motivated v2's guard.

---

## v2: Capacity experts + online abstention guard (CURRENT DESIGN)

### The core problem v1 exposed

The MoE must solve **two separable jobs**, and only one of them is
gradient-learnable:

1. **Which capacity to use when adapting** — learnable. Given the drift, a router
   can learn to prefer a low-rank vs high-rank expert (the capacity optimum is
   cell-dependent per the capacity diagnostic).
2. **Whether to adapt at all (abstention)** — **NOT** gradient-learnable here. On
   any window you fit, *more* adaptation lowers the training MSE, so no phase of
   training ever rewards "don't adapt." Abstention only pays off out-of-sample
   where the drift signal is spurious. A softmax router trained by MSE therefore
   never selects the identity slot — exactly what v1 observed (fallback argmax
   ≈ 0.000; loss-cells recovered 0/10).

v1 asked one softmax to do both and it could only do #1 (weakly). v2 splits them:
the **router** does #1 (gradient-learned), an **online guard** does #2
(non-gradient, from revealed performance).

### Components

```
              x_current, recent_batch
                       │
             drift = concept(x) − concept(recent)         (mlp1/mlp2, as in PROCEED)
                       │
        ┌──────────────┴───────────────┐
        │                              [‖drift‖]                (pre-clip magnitude:
        ▼                               │                        strong abstention signal)
   clip(drift)                    ExpertRouter( [clip(drift), ‖drift‖] ) → softmax(K+1)
        │                               │  gates over {e0..e_{K-1}, FALLBACK}
        ▼                               ▼
 MoEBottleneck = { Bottleneck(bd=8), Bottleneck(bd=32), Bottleneck(bd=64) }   # K=3 experts
        │
  mixture = Σ_k gates[:,k] · expert_k(drift)              # heterogeneous-rank capacity mixture
        │
        ▼
  applied_adaptation = α_guard · mixture                 # α∈[0,1] from the ONLINE GUARD
        │                                                #   α=0 ⇒ identity (few-shot backbone)
        ▼                                                #   α=1 ⇒ full mixture
   Down_Up.assign_adaptation(applied)                    # scale/shift around frozen layers
```

- **Experts** (`MoEBottleneck`, K=3, bottleneck_dims `{8,32,64}`): span the
  characterized capacity range (short horizon ~8, long horizon ~32–64). All
  zero-init to identity, so the mixture *starts* at few-shot.
- **Router** (`ExpertRouter`): input is `[clip(drift), ‖drift‖]` (dim
  `concept_dim+1`). The clip throws away drift magnitude; we feed `‖drift‖`
  (the pre-clip norm) back in explicitly because weak drift ⇒ little to adapt ⇒
  a natural abstention cue. Decoupled router LR (`--router_learning_rate`, 1e-3).
- **Online abstention guard** (`Exp_ProceedMoE._guard_measure`): the load-bearing
  new piece. Per recent (already-revealed) window it measures
  `err_full` (α=1) vs `err_identity` (few-shot backbone), keeps EMAs `E_full`,
  `E_id`, and sets
  `α = sigmoid( ((E_id − E_full) / E_id) / τ )`.
  The **relative** (scale-free) improvement makes a single `τ` transfer across
  datasets with very different MSE magnitudes. α scales the whole mixture toward
  identity where adaptation has recently been hurting → the "loses big" tail
  becomes ties; α≈1 where adaptation clearly helps (Exchange) → the big wins are
  kept. Uses only revealed windows ⇒ **no leakage**. Warm-started during the
  validation-update phase (EMAs are not reset between val and test).

### Why this is not cherry-picking

The guard's `τ`, `guard_ema`, and `concept_dim` are tuned **once on a fixed dev
set** (ETTh1/h2 corners + Exchange 336 + a Weather cell) and **frozen** across the
full held-out grid. The per-step, per-dataset α is computed *online from data*,
never hand-set. Abstention emerges from a uniform control law, not from
per-cell switches chosen after seeing test results.

### New CLI flags (on top of v1's)

| flag | default | meaning |
|---|---|---|
| `--use_guard` | off | enable the online abstention guard |
| `--guard_tau` | 0.1 | guard temperature; lower = sharper abstention (dev-tuned) |
| `--guard_ema` | 0.95 | EMA decay for `E_full` / `E_id` |
| `--guard_alpha_min/max` | 0 / 1 | clamps on α (0 = allow full abstention) |
| `--router_use_norm` | True | feed `‖drift‖` to the router |

`--expert_bottleneck_dims 8,32,64` sets the K=3 experts. Drop `--use_guard` for
the pure-MoE (v1-style) ablation. Roster run scripts:
`ttm-scripts/moe-ssf/<Dataset>.sh` (all datasets incl. `Jiaolong.sh`).

### Expected behavior (the test the guard must pass)

| regime | example | expected α | outcome |
|---|---|---|---|
| few-shot collapses | Exchange 336 | ≈ 1 | keep the big win (−26–35%) |
| adaptation helps a little | ETT short horizon | ~0.5–1 | small win |
| weak drift / adaptation hurts | Weather, wind, Jiaolong, ETT long | → 0 | **tie** few-shot (no big loss) |

---

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
| ETTh1 512_96 | experts 8,32; **lb=0**; lr3e-6 | 0.4668 | −2.9% | ~tie (0.466) |
| ETTh1 1536_720 | experts 8,32; lb=0.01; lr3e-6 | 0.8135 | +4.7% (loses 0.777) | beats fixed @3e-6 (0.845/0.824) |

**Key negative finding (router inert).** In all of the above the router gates
stayed **near-uniform** (~1/3 each) and **fallback was almost never the argmax**
(≈0.004). The gains are from implicit ensembling + constant fallback shrinkage,
**not learned routing**, and abstention did not emerge — even on 1536_720 where
zero-shot beats adaptation. Root cause: the router shared the adapter's tiny
online LR (3e-6) and stayed frozen near its uniform init.

**Fix implemented: decoupled router LR.** The router now lives in its own
optimizer param group (`--router_learning_rate`, default 1e-3; experts/adapter
stay at `--online_learning_rate`). The group is tagged `is_router` so the
online-LR loops (run.py, Exp_Proceed.online) skip it, and `Exp_ProceedMoE.online`
re-asserts it per phase. Diagnostic: `ttm-scripts/moe-ssf/ETTh1_router_lr.sh`
sweeps router LR ∈ {1e-3, 1e-2} on both key cells (lb=0). Watch whether a faster
router concentrates and starts using FALLBACK on 1536_720 — the decisive test of
whether a learned feed-forward router is viable, else fall back to explicit
abstention supervision (see Future work #3).

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
  gate grad flow afterward. `Exp_ProceedMoE._select_optimizer` builds TWO groups
  (adapter + router-tagged `is_router`) so the router can use its own LR; the
  online-LR loops in run.py / Exp_Proceed.online skip `is_router` groups.
- **`remove_frozen_param_from_optim`** (exp_basic.py) only reindexes
  `param_groups[0]` — it assumes a single group. It runs only on an optimizer-
  checkpoint reload ValueError (not hit by these `--pretrain`-free frozen runs),
  but if MoE is ever combined with optimizer-checkpoint loading, that helper must
  be made router-group-aware.

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
