# Per-Dataset Capacity Selection Study (frozen PROCEED + TTM)

Status: **selection complete on validation; final full-grid runs pending.**
Branch: `with-tsfm`. Scripts: `ttm-scripts/capacity-select/`.
Companions: `PROCEED_ONLINE_LR_MECHANISM.md` (why lr behaves as it does),
`MOE_SSF_GUARD_NEGATIVE_RESULT.md` (why we returned to a single adapter).

## Why this study

After the MoE-SSF + online-guard exploration was **falsified** (see the negative-
result doc: the guard hurts, the mixture only matches a single adapter), the plan
returned to a **single frozen PROCEED adapter** and asked the simpler question:
*pick per-dataset capacity + a learning-rate rule, on validation only, and extend
to the full grid.*

Design knobs decided up front:
- **lr:** intended as a global horizon rule (3e-6 short / 1e-6 long), co-searched
  to validate it transfers.
- **capacity (cdim, bneck):** per dataset, as two coupled presets —
  `reduced {cdim64, bneck8}` vs `full {cdim200, bneck32}`.
- **ema = 0** (documented inert; removes a knob).

## Protocol (anti-cherry-picking)

Per dataset, search `{2 capacity presets} × {2 lr} × {4 grid corners}` =
**16 validation-only runs** (`--do_valid`, which skips test). Corners are
`seq ∈ {512, 1536} × pred ∈ {96, MAXPRED}` (MAXPRED = 720, or 336 for
Exchange/AirQuality). 11 datasets → **176 runs**. Selection metric is
`Best Valid MSE`; **test is never read during selection.** The chosen per-dataset
config is then extended to **every** context/horizon cell — so the interior cells
(seq 1024; pred 192/336) are genuine **held-out** generalization tests. Fairness:
compare on test against the best *fixed* config (reduced-noft **and** full-noft,
already in `results.xlsx` run-2) **and** few-shot — not few-shot alone.

Scripts: `corner_select.sh` (the 176-run sweep) → `parse_corners.py` (validate lr
rule + pick capacity) → `emit_final.py` (final full-grid scripts under `final/`).

## Finding 1 — the lr horizon-rule is FALSIFIED

The ETT-derived rule (3e-6 short / 1e-6 long) does **not** transfer to the roster:

> short (pred≤192) prefers 3e-6 in **9/44**; long prefers 1e-6 in **21/44**.

- **Short horizon: 1e-6 wins ~35/44** — the *opposite* of "3e-6 for short."
- **Long horizon: a coin flip.**
- **Magnitude:** lr is a **minor lever (~1–3%) for 10 of 11 datasets** — consistent
  with the amortized/frozen regime (see the lr-mechanism doc: online lr only
  moves the adapter's output bias). The **exception is Exchange**, where it is
  *major*: at seq1536/pred336, reduced is **0.633 (3e-6) vs 0.793 (1e-6)** — a 25%
  swing.

**Reframe (more defensible + mechanistic): lr tracks *drift*, not *horizon*.**
Genuinely non-stationary data (Exchange) benefits from *more* online adaptation
(higher lr); near-stationary data wants *less* (avoid over-adapting). So the rule
is **1e-6 everywhere, 3e-6 for Exchange** — not a horizon schedule.

## Finding 2 — per-dataset capacity: mostly reduced, full only where it bites

Mean validation MSE over the 4 corners (rule-lr per corner):

| dataset | reduced | full | pick | margin |
|---|---|---|---|---|
| **Exchange** | 0.2928 | **0.1939** | full | **33.8%** |
| **ETTh2** | 0.2695 | **0.2399** | full | **11.0%** |
| Energy | 0.3671 | 0.3592 | full | 2.2% |
| ETTm2 | 0.1254 | 0.1227 | full | 2.1% |
| Energy/ETTm2 … | | | | *(borderline)* |
| ETTh1 | 1.3660 | 1.3913 | reduced | 1.8% |
| BeijingAQ | 0.6824 | 0.6766 | full | 0.9% |
| wind | 0.3422 | 0.3392 | full | 0.9% |
| Jiaolong | 0.2479 | 0.2500 | reduced | 0.8% |
| Weather | 0.6072 | 0.6052 | full | 0.3% |
| ETTm1 | 0.7740 | 0.7764 | reduced | 0.3% |
| AirQuality | 1.0364 | 1.0379 | reduced | 0.1% |

Applying "**>2% margin is real, else default reduced**": only **Exchange, ETTh2**
are clear full-preferrers; **Energy** is borderline (2.2%). The sub-1% "full" picks
(BeijingAQ/wind/Weather) are washes → reduced. This matches run-2's "reduced ≈ full
except Energy," now joined by Exchange and ETTh2.

## Finding 3 — the big picks are CORNER-DOMINATED (verified)

Exchange→full is driven **entirely by the single hardest corner (seq1536/pred336)**:
reduced *underfits* there (early-stops at epoch 4, val 0.633) while full keeps
improving to epoch 13 (val 0.325). But at the *easy* corner (seq512/pred336),
**reduced is better** (0.194 vs 0.220). Same shape for ETTh2 (full only wins at
seq1536/pred720). So a per-dataset *fixed* "full" is a compromise: it helps the
hard cell and can slightly hurt the easy cells. The right unit is arguably
(context × horizon), but neither a pure per-dataset nor a pure global-difficulty
rule is clean (ETTh1/ETTm1 prefer reduced *even at* their hard corner). **Decision:
accept per-dataset flat capacity and let the held-out test cells reveal any
easy-cell cost.**

## The decided final config (extend to full grid)

| capacity | lr | datasets |
|---|---|---|
| **full** (cdim200/bneck32) | **3e-6** | Exchange *(reuse existing full-noft@3e-6)* |
| **full** | **1e-6** | ETTh2, Energy |
| **reduced** (cdim64/bneck8) | **1e-6** | ETTh1, ETTm1, ETTm2, Weather, Jiaolong, wind, AirQuality, BeijingAQ |

`ema = 0` throughout. Single frozen PROCEED (`--freeze_online`, no MoE/guard).
Emitted to `ttm-scripts/capacity-select/final/<Dataset>.sh`.

## Caveats to check when the test results land

1. **Validation ≠ test on the flagship.** Exchange val here is 0.19–0.63; test was
   ~0.78–1.06 (drift grows into the test region). The picks are *hypotheses* until
   the final TEST grid confirms them — that is what the held-out interiors are for.
2. **Corner-domination watch-list:** for the full datasets, watch their *easy* cells
   (Exchange 512_×, ETTh2 512_× / 1536_96) where val preferred reduced; if full
   costs there on test, horizon-gate capacity for just those cells.
3. **Energy is the weakest full pick** (~2% val, mixed across its own corners) —
   essentially a near-wash; low-risk either way.
4. **Report per cell vs the best fixed config (reduced *and* full) + few-shot.**
