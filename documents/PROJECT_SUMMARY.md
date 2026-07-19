# Project Summary — Frozen PROCEED + TTM against Concept Drift

Full research arc from the opening question to the current state. Companion docs:
`PROCEED_TTM_HYPERPARAM_NOTES.md` (hyperparameter details), `MOE_SSF_DESIGN.md`
(the parked MoE exploration, on branch `moe-ssf`).

---

## 1. The goal

Improve a **time-series foundation model (TinyTimeMixer / TTM)** against
**concept drift** using **PROCEED** (an adaptation framework), under a
**frozen-backbone constraint** — no fine-tuning of the foundation model — chosen
originally to save memory and avoid catastrophic forgetting.

**Terminology (corrected mid-project):**
- **few-shot** ("Naive frozen" baseline) = TTM head fine-tuned on the historical
  split, then frozen. *Not* zero-shot (pure zero-shot was never measured).
- **PROCEED-frozen / reg_combo2** = few-shot + PROCEED adapter; backbone frozen,
  adapter adapts online.
- **PROCEED-ft / naive-ft** = same but the model (or head) is updated online.

## 2. Hyperparameter phase (frozen PROCEED on ETT/Weather)

**Root cause:** the PROCEED adapter initializes to identity, so frozen PROCEED
*starts* equal to few-shot; online updates move it away. The inherited online LR
(1e-4) was ~30x too aggressive -> net-harmful over-adaptation.

Levers:
- **`online_learning_rate` — dominant.** Optimal ~**3e-6** (U-shaped).
- **`bottleneck_dim` / `concept_dim`** — capacity, secondary. Optimal capacity is
  **cell-dependent** (short horizon -> reduced; most long horizon -> full; with
  counterexamples). LR x capacity interaction: full capacity tolerates higher LR.
- **`ema` — inert** (hypothesis refuted).
- Best recipe: **reg_combo2 = cdim64, bneck8, ema0.9, lr3e-6.**

**Result:** on ETT/Weather, reg_combo2 beat few-shot on ~half the cells but by
tiny margins — **net slightly negative on the full grid** (32 wins avg -2.0%,
40 losses avg +2.6%; "wins small, loses big").

## 3. Headroom analysis — the ceiling and the pivot

- **Oracle ceiling** (best possible adapt-or-abstain selector) over few-shot on
  these benchmarks = **~0.9% MSE.** Marginal.
- **Drift statistics do NOT predict headroom** — decisive natural experiment:
  ETTh2 and ETTm2 are the *same data* (different sampling), identical drift stats,
  yet 4.6x different headroom (0.41% vs 1.88%).
- **Conclusion:** the ceiling is an artifact of **near-corpus benchmarks**
  (ETT/Weather resemble TTM's pretraining data). **Pivot: select datasets
  behaviorally** — where the few-shot TSFM actually degrades — not by drift stats.
  Target **out-of-corpus + non-stationary** data.

## 4. MoE-SSF — explored, then parked (branch `moe-ssf`)

Mixture of scale/shift experts + identity fallback + router:
- Router inert until we **decoupled its LR**; then it learned **expert selection**
  but **never abstention** (fallback never wins — the online MSE gradient rewards
  recent-window adaptation, so it can't learn "don't adapt").
- On loss-prone cells, expert-selection **recovered 0/10** (adapted harder where
  adaptation hurt).
- **Verdict:** the mixture isn't the answer; the ceiling problem was **datasets,
  not method.**

## 5. The breakthrough — Exchange + the behavioral probe

**Phase 0 (headroom, out-of-corpus datasets):**

| cell | few-shot | PROCEED-frozen | vs few-shot |
|---|---|---|---|
| Exchange 192 | 0.500 | 0.471 | **-5.8%** |
| **Exchange 336** | 1.089 | 0.806 | **-26.0%** |
| wind 96 / 720 | | | -1.8% / -2.4% |

Frozen PROCEED beats few-shot by **20-30x the ETT ceiling** on Exchange
(out-of-corpus, non-stationary financial). **Horizon reversal:** headroom *grows*
with horizon on Exchange (few-shot collapses at long horizon on non-stationary
data, 0.22->1.09, leaving room the adapter rescues) — the opposite of ETT.

**Phase 1 (forgetting, LR-robustness):**

| online_lr | ETTh1 naive-ft | ETTh1 PROCEED-ft |
|---|---|---|
| 1e-4 | +25% | +17% |
| 1e-2 | **+960,901%** (MSE 4563, destroyed) | +76% (bounded) |

Naive fine-tuning **catastrophically diverges** as LR rises; PROCEED-ft stays
bounded (frozen backbone can't be overwritten). **But both fine-tune variants
lose to few-shot at every LR** -> fine-tuning is a dead end; only the **frozen**
backbone both avoids forgetting *and* beats few-shot.

**Attribution control (`lr=0` on PROCEED-frozen, Exchange 336):** -25.8% ~=
the -26% at lr=3e-6. So **the headroom is gradient-free** — from the trained
adapter generating drift-conditioned scale/shift at inference (amortized
adaptation), not from online gradient updates. *Remaining confound:*
PROCEED-frozen trains head+adapter vs few-shot head-only, so a **capacity-matched
static-SSF control** is still needed to rule out "just more params."

## 6. The crystallized thesis

> On **out-of-distribution, non-stationary data** where a pretrained TSFM
> degrades, **naive online fine-tuning catastrophically forgets**, and even
> PROCEED-fine-tune only *fails safely*. A **frozen-backbone PROCEED adapter**
> avoids forgetting entirely *and* captures the drift, **substantially beating
> few-shot (up to -26%)**. The frozen constraint isn't a memory compromise — it's
> the mechanism. Motivation and result are the same story.

## 7. Current state (branch `with-tsfm`)

- **Datasets registered** (`settings.py`): BeijingAQ, Energy, AirQuality
  (+ existing Exchange, wind, Jiaolong). reg_combo2 scripts written for all
  (`ttm-scripts/proceed-regcombo2/`).
- On disk, unrun: exchange@usa (2nd financial), THU-Concept-Drift (controlled
  synthetic — the causal study), Weather 2024-26 (a "recency != drift" control,
  expected ~0%).
- **Selection rule:** out-of-corpus + non-stationary, verified with a cheap
  behavioral probe (few-shot vs PROCEED-frozen); promote only cells clearing
  ~few%.
- **Probe scripts:** `ttm-scripts/probe/` (Phase 0 headroom, Phase 1 forgetting,
  `analyze_probe.py`).
- **Branches:** `with-tsfm` = main line; `moe-ssf` = parked MoE exploration.

## 8. Open questions / next steps

1. **Confirm & expand headroom** — reg_combo2 + few-shot on Exchange (full grid)
   and the new datasets; does -26% replicate beyond one dataset?
2. **Load-bearing control** — static-SSF (capacity-matched, no drift conditioning)
   vs PROCEED-frozen on Exchange 336: is the win the *drift mechanism* or just
   *capacity*?
3. **Full- vs reduced-capacity ablation** on Exchange (long-horizon OOD is where
   ETT hinted full might help).
4. **Seeds** — Exchange 336 has notable run-to-run variance; pin it.
5. **THU controlled synthetic study** — show the advantage scales with drift
   severity (causal proof, dataset-independent).
6. **Contrast set** — keep ETT/Weather (+ Weather-2024-26) as the "no headroom"
   baseline that makes the OOD result meaningful.

---

**One-line version:** we started trying to tune/architect a better adapter,
discovered the real problem was that standard benchmarks give a strong TSFM
nothing to adapt to, pivoted to behaviorally-selected out-of-corpus non-stationary
data, and found frozen PROCEED delivers large gains (-26%) exactly where
fine-tuning catastrophically forgets — vindicating the frozen design as the
mechanism, not a compromise.
