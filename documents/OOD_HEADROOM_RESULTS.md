# Out-of-Corpus Headroom Study — Results (run-2)

Results for the behaviorally-selected, out-of-corpus / non-stationary datasets,
testing the thesis: *frozen PROCEED beats few-shot where the few-shot TSFM
collapses.* Companion: `PROJECT_SUMMARY.md` (full arc), `MOE_SSF_DESIGN.md`.

## Setup

Three variants compared, MSE + MAE, averaged over contexts seq_len ∈
{512, 1024, 1536}:

- **fewshot** — TTM head fine-tuned on the historical split, then frozen (the
  "Naive frozen" baseline).
- **reduced-noft** — PROCEED-frozen, reg_combo2 recipe (cdim64, bneck8, ema0.9,
  lr3e-6), backbone frozen.
- **full-noft** — PROCEED-frozen, full capacity (cdim200, bneck32).

Datasets cleaned first (missing values / sentinels → interpolated; RAIN dropped
from BeijingAQ, NMHC(GT) dropped from AirQuality; see
`ttm-scripts/probe/clean_datasets.py`). Source table: `results.xlsx` sheet
`run-2`. pred=720 not run for Exchange/AirQuality (val split too small).

## Headline (MSE, mean over contexts; best PROCEED vs fewshot)

| dataset | pred | fewshot | reduced | full | best vs fewshot |
|---|---|---|---|---|---|
| **Exchange** | 96 | 0.246 | 0.243 | 0.241 | −1.9% |
| **Exchange** | 192 | 0.505 | 0.486 | 0.485 | −3.9% |
| **Exchange** | **336** | **1.207** | 0.787 | **0.782** | **−35.2%** |
| AirQuality | 96 | 1.171 | 1.159 | 1.155 | −1.4% |
| AirQuality | 192 | 1.298 | 1.284 | 1.274 | −1.9% |
| **AirQuality** | 336 | 1.441 | 1.380 | 1.388 | **−4.2%** |
| BeijingAQ | 96 | 0.649 | 0.648 | 0.647 | −0.3% |
| BeijingAQ | 336 | 0.733 | 0.730 | 0.722 | −1.5% |
| BeijingAQ | 720 | 0.790 | 0.831 | 0.805 | **+1.9% (worse)** |
| wind | 96 | 0.188 | 0.186 | 0.188 | −1.2% |
| wind | 720 | 0.220 | 0.228 | 0.229 | **+3.6% (worse)** |
| Energy | 336 | 0.619 | 0.625 | **0.586** | **−5.5% (full)** |
| Energy | 720 | 0.886 | 0.900 | **0.834** | **−5.8% (full)** |

## Per-dataset verdict

- **Exchange — strong clean win.** −35% at pred 336, driven by few-shot
  *collapsing* with context: at seq 512 few-shot MSE is 1.06 (−26%), but at seq
  1536 few-shot blows up to **1.65** while PROCEED stays 0.78 (−53%). reduced ≈
  full. The rock-solid anchor.
- **AirQuality — modest clean win.** −1 to −4% (best −4.2% @336). reduced ≈ full.
  Genuine second data point, weaker.
- **Energy — mixed / capacity-sensitive.** **Full** wins MSE (−5.5%/−5.8% at
  336/720) but *loses* MAE (+2.1%/+2.7%); **reduced loses both.** MSE↓ but MAE↑
  means full capacity fixes large/tail errors while worsening typical points —
  consistent with spiky appliance data. **First dataset where capacity matters
  (full ≫ reduced, −7.3% at 720).** Not a clean headroom win.
- **BeijingAQ, wind — null.** ~tie at short/mid, slightly *worse* than few-shot
  at 720 (over-adaptation, like ETT). Few-shot is already at its ceiling here.

**Tally: 1 strong (Exchange) + 1 modest (AirQuality) + 1 mixed (Energy) +
2 null (BeijingAQ, wind).** The story is Exchange-carried.

## Key observations

1. **Headroom ⟺ few-shot behaviorally collapses** (representational-ceiling
   frame), NOT "OOD". Where few-shot MSE explodes with horizon/context (Exchange
   → 1.65, AirQuality → 1.44) the adapter recovers the gap; where few-shot is
   stable (BeijingAQ ~0.65, wind ~0.2) it's already at ceiling and PROCEED
   slightly over-adapts. Refined selection rule: **screen candidates by
   *few-shot fails*, not by *is it OOD*.**
2. **Capacity: reduced ≈ full almost everywhere** (Exchange/AirQuality/BeijingAQ/
   wind), so the lightweight recipe loses nothing → default to **reduced**. The
   exception is **Energy** (full ≫ reduced at long horizon), echoing the old ETT
   capacity diagnostic (long horizon can prefer full capacity).
3. **reduced ≈ full where headroom exists** is mild evidence the win isn't raw
   capacity → the **static-SSF attribution control** (capacity-matched, no drift
   conditioning) on Exchange 336 is still the clean test of drift-mechanism vs
   capacity.
4. **MSE/MAE can disagree** (Energy) — decide the headline metric; if reporting
   both, Energy is a wash and belongs in the capacity-ablation discussion, not
   the headroom table.

## MoE-SSF: still NOT justified (re-examined against these results)

- Where headroom exists (Exchange), a **single** adapter captures it and
  reduced ≈ full → mixture/capacity is not the bottleneck.
- Where headroom doesn't exist (BeijingAQ/wind), MoE **cannot break the
  representational ceiling**: SSF experts are scale/shift on the *frozen*
  backbone — they modulate, they don't add representational capacity.
- The mixed win/over-adapt picture justifies **abstention** as a goal, but the
  gradient-learned router provably won't abstain (see `MOE_SSF_DESIGN.md`);
  needs an explicit online performance guard, and its payoff is small.
- **MoE trigger condition (not yet observed):** a collapse-prone dataset where a
  single *full-capacity* adapter underfits AND a *mixture* beats full. Energy
  shows the opposite (single full already beats reduced). Revive MoE only if
  that condition appears.

## Next

1. **More collapse-prone (financial-type) datasets** — Exchange is financial;
   add exchange@usa, crypto. Screen by "few-shot MSE grows pathologically with
   horizon/context" on *both* metrics.
2. **Static-SSF attribution control** on Exchange 336 (drift mechanism vs
   capacity).
3. **Decide the headline metric** (MSE vs MAE) — matters for Energy's placement.
4. Default runs to **reduced capacity**, `common_channel` decoder.
