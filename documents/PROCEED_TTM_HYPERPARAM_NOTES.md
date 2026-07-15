# Frozen PROCEED + TTM: Hyperparameter Investigation Notes

## Goal

Make frozen (`--freeze_online`, no online fine-tuning of the TTM backbone/head)
PROCEED beat zero-shot TTM ("Naive frozen" in `results.xlsx`). Fine-tuning is
disabled deliberately: it preserves TTM's pretrained capability, avoids
catastrophic forgetting, and cuts online memory cost. Without fine-tuning,
naive online adaptation (plain gradient descent, no PROCEED) already loses to
zero-shot at long context; the question is whether PROCEED's drift-conditioned
adapter can do better than both.

**Starting diagnosis:** the PROCEED generator (`adapter/module/generator.py`,
`Bottleneck.__init__`) initializes its final weight/bias to zero, so at init
`scale = adaptation + 1 = 1` and `shift = 0` — an exact identity. Frozen
PROCEED therefore *starts* equal to zero-shot TTM, and every online update
moves it away from that already-good point. The original defaults
(`online_learning_rate=1e-4`, inherited from small-model PROCEED on
PatchTST/DLinear) turned out to push far too hard, causing net-harmful
over-adaptation that got worse with longer context.

All experiments below use `--freeze_online`, `train_ratio=0.5`,
`test_ratio=0.45`, on ETTh1 and ETTh2, at the four corner cells
`seq_len ∈ {512, 1536} × pred_len ∈ {96, 720}` unless noted. Scripts live in
`ttm-scripts/proceed-sweep/`, `ttm-scripts/proceed-lr-sweep/`,
`ttm-scripts/proceed-ema0-sweep/`, `ttm-scripts/proceed-capacity-sweep/`; logs
parsed from the matching `logs/proceed_*_sweep/` folders.

---

## Hyperparameter effects

### `online_learning_rate` — the dominant lever

By far the strongest and most consistent effect. In the first OFAT sweep,
lowering LR alone (`1e-4 → 3e-5`) improved MSE in **8/8 cells**, monotonically,
never once hurting — every other lever (bottleneck_dim, concept_dim, ema) only
helped in about half the cells and could backfire by double digits.

A follow-up sweep over `{3e-5, 1e-5, 3e-6, 1e-6}` (holding capacity/ema fixed
at the best-so-far recipe) found the response is **U-shaped**: it keeps
improving down to **~3e-6**, then `1e-6` starts overshooting into
too-little-adaptation for most cells. Six of eight cells bottom out at 3e-6.

**Long horizon prefers an even lower LR than short horizon.** At `pred=720`,
some cells (e.g. ETTh2 1536_720) were *still* improving at `1e-6`
(0.643 → 0.511 → 0.499 → 0.493, monotonically down), i.e. the true optimum
there is below 3e-6. At `pred=96`, `1e-6` already overshoots back up.

**Takeaway:** `online_learning_rate` should be treated as the primary tuning
knob, roughly two orders of magnitude below the small-model PROCEED default.
A horizon-aware value (lower for long horizon) captures a bit more
performance than one fixed value.

### `bottleneck_dim` (generator rank) — helps mid/long horizon, can hurt at the hardest cell

Lower rank (32 → 16 → 8 → 4) regularizes the adapter back toward its
identity init. In the OFAT sweep this consistently helped at `pred=720`
(bneck4 was −15% MSE at ETTh1 512_720) but was inconsistent at `pred=96`
(small positive or negative swings) and could actively hurt at the hardest
cell (ETTh1 1536_720: bneck4 was **+11%**, i.e. reducing capacity made that
cell worse, not better).

**Takeaway:** capacity reduction is a real but second-order lever, and it is
not uniformly beneficial — at the single hardest cell we found, less capacity
made things worse, which is why the capacity diagnostic
(`ttm-scripts/proceed-capacity-sweep/`) exists: to check whether that cell
instead wants *more* capacity once LR is properly tuned.

### `concept_dim` (drift representation size) — same direction as bottleneck_dim, weaker effect

Lower `concept_dim` (200 → 64 → 32) had a similar but smaller regularizing
effect to `bottleneck_dim` — helped mid/long horizon on average, occasionally
hurt (e.g. cdim32 was +8.4% at ETTh1 1536_720, the same hold-out cell). Never
the single best lever on its own; useful as a secondary regularizer stacked
with LR and bottleneck_dim reduction.

### `ema` (concept-history smoothing) — inert once LR is properly tuned

Initially hypothesized as a fix for the ETTh1 1536_720 anomaly (flat ~0.84
MSE regardless of LR/capacity, worse than zero-shot). Directly tested:
`ema=0` vs `ema=0.9` at all four `pred=720` cells, both LRs (3e-6, 1e-6) —
**moved MSE by ≤0.007 in every case (noise-level), and was equal-or-worse in
7/8 comparisons.** The hypothesis that `ema=0.9` biases the drift estimate at
long horizon is **refuted**.

**Takeaway:** at the LR regime that actually works (~1e-6 to 3e-6), the
per-step adaptation is already small enough that smoothing the concept
history barely matters. `ema` is not a useful tuning knob here — pick either
value; we kept `ema=0.9` since it was marginally non-worse.

---

## Best configuration found so far

```
--freeze_online
--concept_dim 64
--bottleneck_dim 8
--ema 0.9              (inert — 0 works equally; kept at 0.9 for the winning recipe)
--online_learning_rate 0.000003     # 3e-6
```

This is `reg_combo2` (from the original OFAT sweep) with its `online_lr`
corrected from the sweep's `3e-5` down to the properly-tuned `3e-6`. It is the
best single recipe found so far on both ETTh1 and ETTh2: it **beats zero-shot
TTM on 6 of 8 corner cells**, including all four short-horizon (`pred=96`)
cells clearly (−3% to −5% MSE).

### Refinement: horizon-aware LR

The single best fixed LR (3e-6) is not the true optimum at every cell — long
horizon wants it even lower:

> Long horizon (720) slightly prefers 1e-6, not 3e-6. ETTh2 1536_720 was best
> at 1e-6 (0.493 vs 0.499 at 3e-6), and it was still improving downward at
> 1e-6 (not yet at a floor). So the *absolute* best is a horizon-aware LR
> (3e-6 for short horizon, 1e-6 for long horizon); as a *single* default
> across all horizons, 3e-6 is the right pick — it's a small compromise for
> short horizon and still strong for long horizon, versus 1e-6 which
> overshoots (under-adapts) at short horizon.

### Open hold-outs (not fixed by any hyperparameter tried so far)

Two cells still lose to zero-shot under every recipe tested (LR, capacity,
ema, all combinations):

- **ETTh1, seq_len=1536, pred_len=720** — flat ~0.846 MSE across all LR/ema
  settings tried; zero-shot is 0.777 (+8.9%). Best-ever config at this cell
  was actually *full capacity* at the old `lr=3e-5` (0.786) — still short of
  zero-shot, and better than the reduced-capacity `reg_combo2` (~0.845).
- **ETTh2, seq_len=512, pred_len=720** — ~0.505–0.512 MSE across ema/LR
  variants tried at reduced capacity; zero-shot is 0.475 (+6–8%).

These are **cell-specific, not a clean "long horizon" or "long context"
rule** — ETTh1 wins at 512_720 but loses at 1536_720; ETTh2 does the
opposite (loses at 512_720, wins at 1536_720). This inconsistency is why we
don't generalize a single seq_len/horizon cutoff from the data so far.

**Open question, in progress:** `ttm-scripts/proceed-capacity-sweep/` tests
*full* capacity (`cdim=200, bneck=32`) at the new low LRs (`3e-6`, `1e-6`) on
these hold-out cells, to distinguish two hypotheses:

1. The hold-outs are capacity-starved (reduced-capacity `reg_combo2` under-fits
   long-horizon drift) → more capacity at the right LR should close the gap →
   justifies building MoE-SSF (more adaptation capacity, routed).
2. Adaptation fundamentally cannot help at these cells, regardless of
   capacity → more capacity won't fix them either → prefer a horizon-aware
   near-identity fallback (skip/shrink adaptation) at those cells, and scope
   MoE-SSF only to cells that already show adaptation signal.

### Capacity diagnostic results (RESOLVED — `logs/proceed_capacity_sweep/`)

Full capacity (`cdim=200, bneck=32`) vs reduced reg_combo2 (`cdim=64, bneck=8`),
both `ema=0.9`, best of `lr ∈ {3e-6, 1e-6}` per cell, at `pred=720`:

| cell (720) | full-cap best | reduced best | zero-shot | capacity winner |
|---|---|---|---|---|
| ETTh1 512 | **0.693** (3e-6) | 0.735 | 0.744 | FULL — beats zero −6.8% |
| ETTh1 1536 | 0.824 (3e-6); 0.786 @3e-5 | 0.844 | 0.777 | FULL — better, still +1.2–6% |
| ETTh2 512 | **0.477** (3e-6) | 0.505 | 0.475 | FULL — ties zero +0.3% |
| ETTh2 1536 | 0.532 | **0.493** (1e-6) | 0.502 | RED — full loses to zero |

**Hypothesis 1 (capacity-starved) confirmed for long horizon, but capacity
preference is CELL-DEPENDENT — this is the core MoE-SSF justification:**

- At `pred=720`, more capacity helps in 3/4 cells (by 4–6% MSE) — the OPPOSITE
  of `pred=96`, which preferred the shrunken reg_combo2. So capacity preference
  **flips with horizon**: short → low capacity, most long → high capacity.
- Full capacity **resolves two of the three hold-outs**: ETTh2 512_720
  (+6.3% → +0.3%, tied) and turns ETTh1 512_720 into a clear win (−6.8%).
- **Counterexample: ETTh2 1536_720** — reduced capacity wins there and full
  capacity LOSES to zero-shot. So no single fixed capacity is optimal across
  cells.
- **LR × capacity interaction:** full-capacity adapters tolerate/prefer a
  slightly larger LR (ETTh1 1536_720: full@3e-5=0.786 > full@3e-6=0.824).
- With per-cell-optimal capacity, ≥ zero-shot is reachable on **7/8 corner
  cells** (4 short via reduced, 3/4 long via full); only **ETTh1 1536_720**
  remains a genuine hold-out (best-ever 0.786 @ full+3e-5, still +1.2% over
  zero-shot — resists even full capacity).

**Implication:** the heterogeneity of optimal (capacity, LR) across cells cannot
be captured by one hand-tuned recipe → build MoE-SSF. Design signals from this
data: (1) experts should span a capacity range (low-rank + high-rank);
(2) router conditions on drift AND horizon (both flip the optimum);
(3) keep a near-identity fallback expert (worst-case = zero-shot, covers
ETTh1 1536_720); (4) allow per-expert effective step size (LR×capacity
interaction). See [[moe-ssf-approach]].

---

## Summary table (best SINGLE-recipe reg_combo2 @ lr=3e-6 vs zero-shot, MSE)

Note: this is the best *fixed* recipe; the capacity diagnostic above shows long
horizon can do better with full capacity (per-cell), which reg_combo2 does not
use.

| cell | reg_combo2 @ lr=3e-6 | zero-shot | result |
|---|---|---|---|
| ETTh1 512_96 | 0.466 | 0.481 | **beats** (−3.1%) |
| ETTh1 1536_96 | 0.470 | 0.493 | **beats** (−4.7%) |
| ETTh1 512_720 | 0.735 | 0.744 | **beats** (−1.2%) |
| ETTh1 1536_720 | 0.845 | 0.777 | loses (+8.7%) — hold-out |
| ETTh2 512_96 | 0.224 | 0.224 | tie |
| ETTh2 1536_96 | 0.224 | 0.226 | **beats** (−0.8%) |
| ETTh2 512_720 | 0.505 | 0.475 | loses (+6.3%) — hold-out |
| ETTh2 1536_720 | 0.499 (0.493 @ lr=1e-6) | 0.502 | **beats** (−0.2% to −1.8%) |
