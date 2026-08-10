# What PROCEED Actually Does on Exchange — Ablation Decomposition

Three ablations on the reduced-noft (frozen PROCEED) adapter, **Exchange only**
(cdim64/bneck8, ema0, lr 3e-6, contexts 512/1024/1536 × horizons 96/192/336,
single seed). They converge on one conclusion and **correct the mechanism story**:
the win is not drift tracking — it is a **learned static, additive, per-layer
shift**. Companion: `PROCEED_ONLINE_LR_MECHANISM.md`, `CAPACITY_SELECTION_STUDY.md`,
`EXCHANGE` results in `PROJECT_SUMMARY.md`. Scripts: `ttm-scripts/ablation-study/`.

## The headline
On Exchange, frozen PROCEED beats few-shot by ~29% (mean MSE 0.5064 vs 0.6524).
The three ablations each strip a different part of the method; each removal costs
almost nothing **except** removing the additive shift:

- **Concept encoder doesn't matter** (dual best; alternatives +2%).
- **Drift-conditioning doesn't matter** (static SSF +0.8%).
- **The additive shift is load-bearing** (shift-only +1.3%; scales without shift +5%).

→ The mechanism reduces to **a static, low-dimensional, gradient-free, per-layer
additive bias that re-centers the frozen backbone's output to fit Exchange** (where
few-shot collapses at long horizon).

## ① Concept encoder — barely matters (`ttm-scripts/ablation-study/encoders.sh`)
| encoder | mean MSE | vs dual |
|---|---|---|
| **dual** (two encoders, drift = c(X_t)−c(X_{t−H})) | 0.5064 | — |
| stats (fixed data-space moment featurizer, no learned encoder) | 0.5159 | +1.9% |
| current (**only c(X_t), no drift subtraction**) | 0.5166 | +2.0% |
| shared (one encoder for both) | 0.5189 | +2.5% |
| few-shot | 0.6524 | +28.8% |

`current` (no drift at all) is within 2%, and `stats` (hand-crafted, no learned
encoder) within 1.9%. Neither the learned concept space nor the drift *subtraction*
is doing the work.

## ② Drift-conditioning vs static SSF (`ssf_vanilla.sh`)
| variant | mean MSE | vs proceed |
|---|---|---|
| proceed (drift-conditioned) | 0.5064 | — |
| **vanilla (zero drift → static SSF)** | 0.5103 | **+0.77%** |

A static scale/shift (same generator at zero drift) captures ~99% of the win. The
drift-conditioning hypernetwork is nearly inert on Exchange.

## ③ SSF component decomposition (`ssf_components.sh`)
| component | mean MSE | vs full |
|---|---|---|
| **full** (scale + scale2 + shift) | 0.5064 | — |
| scale_shift (drop input scale2) | 0.5096 | +0.6% |
| **shift_only** | 0.5132 | +1.3% |
| inscale_only (no shift) | 0.5322 | +5.1% |
| scale_only (no shift) | 0.5334 | +5.3% |

Shift alone ≈ full (within 1.3%); scales *without* the shift are +5% worse. The
additive shift is essential; the scales are marginal refinement.

## Where the extra machinery earns its keep: the hardest corner
At **1536_336** (largest drift), the drift-conditioning and scales DO add value:
- drift beats static: proceed 0.8292 vs vanilla 0.8431 (**+1.7%**)
- scales help: full 0.8292 vs shift_only 0.8553 (**+3.1%**)

So drift + scales concentrate their (small) value exactly where drift is largest —
consistent with "drift matters when drift is big" — but averaged over the grid it is
minor.

## Why this unifies the earlier findings
All prior Exchange results are the *same object* — a static, low-rank, gradient-free
per-layer shift:
- **lr=0 amortization** (win is gradient-free): a static shift needs no online gradient.
- **capacity-invariant** (cd16 ≈ cd200): a per-layer shift is low-dimensional.
- **few-shot collapses at long horizon, adapter rescues**: the shift re-centers the
  collapsed forecast.

## Caveats
- **Single seed.** Individual cells are noisy (e.g. at 1536_192, `scale_only` 0.4515
  and `inscale_only` 0.4495 beat full 0.4726, and few-shot 0.4659 beats every
  adapter). The **aggregate means are robust** (three independent ablations agree),
  but don't over-read a cell. 2–3 seeds on {full, shift_only, vanilla} would firm up
  the ~1% gaps.
- **Exchange-specific.** Exchange is out-of-corpus but its drift may be ~constant
  within the online test window, so a static shift suffices. On rapidly
  time-varying drift the drift-conditioning could matter more — untested here.

## Implications
1. **The headline result is intact** — the frozen adapter still delivers −29% on
   Exchange. The ablation clarifies the *mechanism*; it does not weaken the result.
2. **A simplification result:** on Exchange you could replace the whole drift
   machinery with a static per-layer bias correction and lose ~1%.
3. **Honest reframe (adopt this):** do NOT claim the win comes from tracking drift.
   State it as *"a learned static SSF / per-layer shift that re-calibrates the frozen
   backbone; the drift-conditioning, scale components, and learned encoder each add
   ≤2%, earning their keep only at the most-drifted corner."*
