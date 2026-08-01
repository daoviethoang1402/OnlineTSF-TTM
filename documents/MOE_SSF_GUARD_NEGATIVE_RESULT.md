# MoE-SSF + Online Abstention Guard — Negative Result

Status: **explored, built, and FALSIFIED. Abandoned in favor of a single frozen
PROCEED adapter** (see `CAPACITY_SELECTION_STUDY.md`).
Code + full design: branch **`moe-ssf`** (`adapter/module/moe.py`,
`adapter/proceed_moe.py`, `exp/exp_proceed_moe.py`, `documents/MOE_SSF_DESIGN.md`).
This doc records the *conclusion* on the main line so it is not re-attempted.

## What was built (on `moe-ssf`)

The goal was to beat few-shot on both the near-corpus (ETT/Weather) and the
non-stationary (Exchange) datasets with **one** system, by adding two things to
frozen PROCEED:

1. **Mixture-of-SSF-experts:** K heterogeneous-rank experts (bottleneck_dims
   `{8,32,64}`) + a `‖drift‖`-augmented router + a parameter-free identity slot,
   replacing the single adaptation generator. Job: pick *which capacity* to use.
2. **Online abstention guard:** a non-gradient control law that, per revealed
   window, compares full-adaptation vs identity error (EMA) and sets a gate
   `α = sigmoid(((E_id − E_full)/E_id)/τ)` that shrinks the mixture toward identity
   where adaptation has been hurting. Job: decide *whether to adapt at all*
   (abstention is not gradient-learnable — the MSE gradient always rewards
   adapting the window you are fitting).

## The decisive diagnostic (single frozen adapter, 5 configs, 2 cells)

Run on Exchange 512_336 and ETTh2 512_720 (`ttm-scripts/moe-ssf/diag_regression.sh`):

**Exchange 512_336** (few-shot ≈ 1.06, single-PROCEED docs ≈ 0.78):

| cfg | what | MSE |
|---|---|---|
| A | single bd8, cdim64, **no guard** | **0.844** |
| B | single bd8, cdim64, **+guard** | 0.927 |
| C | MoE {8,32,64}, cdim64, +guard | 0.838 |
| D | MoE, cdim200, +guard | 0.932 |
| E | MoE, **cdim128**, no guard | **1.434** |

**ETTh2 512_720** (few-shot 0.475): A 0.500, B 0.561, C 0.581, D 0.515, E 0.483.

## What it proved

1. **No bug.** Config A (single adapter, no guard) reproduces the working baseline
   — Exchange 0.844 (−20% vs few-shot), ETTh2 0.500 (≈ reg_combo2 0.505). The
   `ProceedMoE` code path is sound.
2. **The guard is a NET NEGATIVE — controlled A→B, same adapter ± guard:** Exchange
   **0.844 → 0.927 (+10%)**, ETTh2 **0.500 → 0.561 (+12%)**. Adding the guard
   degrades a *good* adapter on both cells.
3. **The MoE does not beat a single adapter** — best MoE (C, cdim64) 0.838 ≈ A
   0.844; it only *matches*. And **`concept_dim=128` was a bad default** (E blows
   up to 1.434 on Exchange; cdim64 restores it). Nothing beats plain A on both
   cells.

## Why the guard fails (the mechanism)

On Exchange the adapter *helps overall* (0.844 < 1.06) yet the guard measures it
*hurting* on the recent window (`E_full > E_id`) and abstains. That is structural,
not noise: Exchange's adaptation benefit **grows with drift**, and the guard's
signal comes from a window ~pred_len steps back that is **always less drifted**
than the current one. **A trailing-performance guard therefore systematically
under-adapts exactly on the growing-drift datasets where the headroom is.** No τ
fixes an anti-correlated signal. The mixture-router likewise cannot manufacture
representational headroom on the frozen backbone.

## Conclusion

The elaborate machinery — mixture **and** guard — does **not** beat a single frozen
PROCEED adapter, which already delivers the real result (Exchange −20%, ETT-short
wins, Weather tie) and only mildly loses the known long-horizon hold-outs. The
project therefore returned to the single adapter and to **per-dataset capacity +
drift-based lr selected on validation** (`CAPACITY_SELECTION_STUDY.md`). MoE-SSF
and the online guard are **not** to be revived unless a collapse-prone dataset
shows a *single full-capacity* adapter underfitting **and** a mixture beating full
— a condition never observed.
