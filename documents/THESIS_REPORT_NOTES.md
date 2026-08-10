# Thesis Report & Defense Notes

Working guide for the bachelor thesis *"Thích ứng thay đổi đặc tính thống kê đối
với mô hình nền tảng cho dữ liệu chuỗi thời gian"* (T826). English working notes —
translate/adapt into the Vietnamese report. Companion: `PROJECT_SUMMARY.md`,
`CAPACITY_SELECTION_STUDY.md`, `PROCEED_ONLINE_LR_MECHANISM.md`.

---

## 0. Staging (safe-first)

- **Core (safe) thesis = PROCEED + ONLINE fine-tuning** (TTM-only vs TTM-P). This
  is what the current draft defends. Keep it as the spine.
- **Frozen/no-fine-tune (noft) work + Exchange + PatchTST-FM are HELD IN RESERVE.**
  Add them only after the **interim (1st) reviewer** signals the frozen direction
  is welcome. Until then they live in the oral-defense "preliminary results / future
  work" slide, not the written core.

## 1. How to state the novelty (do NOT say "we combined two models")

Lead with the **research question**, not the integration:

> *First empirical study of whether a proactive drift-adaptation module (designed
> for small, per-dataset forecast models) still works when the backbone already
> carries broad pretrained knowledge (a time-series foundation model).*

Then claim the two findings the **original PROCEED paper never showed** (it only
tested horizons 24/48/96 on small models):

1. The proactive advantage **grows with forecast horizon** (out to 720; Weather
   +29%, ETTh2 +26% MSE).
2. The advantage **grows with context length** (gain at ctx 1536 > 512 on 21/24
   configs), with a mechanism: longer context ⇒ larger drift between the two ends
   of the window ⇒ TTM-only degrades while PROCEED holds.

Plus the analytical contributions already in the draft: the **MSE-vs-MAE
decomposition** (advantage concentrated at high-error/drift moments) and the
**parameter-complexity analysis** (added params depend only on rank r, not layer
size → deployability).

## 2. The ablation result — read it as a TWO-LEVEL finding (honest)

Measured ordering (error ascending, lower = better):

> **few-shot < reduced/full-noft < TTM-P < TTM-only**

- **Level 1 — frozen ≫ online.** Both frozen approaches (few-shot, noft) beat both
  online approaches (TTM-P, TTM-only). This is **catastrophic forgetting**:
  continuous online fine-tuning degrades the foundation model. *Strongest, cleanest
  finding.*
- **Level 2 — the adapter is a CONDITIONAL effect.** noft ≈ few-shot on average
  (slightly loses), but **wins where there is headroom** (OOD/non-stationary:
  Exchange −35%, AirQuality, BeijingAQ) and **ties where the FM is at ceiling**
  (in-corpus: ETT/Weather = do-no-harm).

The thesis's internal claim **TTM-P < TTM-only still holds** → keep the defensible
statement: *"PROCEED reduces the forgetting damage when online fine-tuning is
required."*

## 3. Report results per-REGIME, never as an average or a win-count

The average across datasets is the **wrong metric** — it mixes two regimes and is
dominated by the no-headroom majority, hiding the real result. Group instead:

| Group | Datasets | Expectation | Result |
|---|---|---|---|
| **OOD / non-stationary** (FM challenged) | Exchange, AirQuality, BeijingAQ | headroom → win | **large gains (Exchange −35%)** |
| **In-corpus / stationary** (FM near ceiling) | ETT×4, Weather, wind, Energy, Jiaolong | no headroom → tie | ≈tie, **no harm** |

- **KEEP wind and Energy in the report.** Dropping them to raise a win-rate is
  cherry-picking, doesn't even fix the average (few-shot still wins on the subset),
  and throws away the do-no-harm contrast that makes the OOD wins credible. In the
  per-regime frame they are not losses — they are the "no-headroom" control.
- Lead the results section with **magnitude + mechanism** (Exchange −35%, and the
  forgetting contrast where naive online-FT diverges), not with cell counts.

## 4. Defense talking points (bachelor bar = rigor + understanding, not SOTA)

- **"Isn't your gain just from online fine-tuning?"** → The ablation isolates it:
  PROCEED-frozen matches/beats PROCEED-online at a fraction of the memory (no
  backbone gradients). PROCEED's contribution is the drift-conditioning, shown to
  be gradient-free (lr=0 ≈ lr=3e-6).
- **"Few-shot beats your method — why fine-tune at all?"** → On near-corpus
  stationary data you shouldn't — that is our finding (online-FT forgets). The value
  appears on OOD non-stationary data (Exchange), where the FM degrades: the frozen
  adapter recovers −35% while online fine-tuning diverges.
- **"Are you just winning by adding parameters?"** → No — capacity sweep shows
  cd16 ≈ cd200; the win is capacity-invariant (a tiny adapter captures it).
- General principle: **own the weakness, explain it mechanistically, show it led
  somewhere** (the frozen approach). Honesty + understanding scores higher than a
  hidden weakness.

## 5. Unified thesis statement

> *Continuous online fine-tuning of a time-series foundation model is
> counterproductive — it induces catastrophic forgetting and is beaten by a simple
> frozen (few-shot) baseline. Concept drift is better handled by keeping the
> backbone frozen and attaching an amortized, drift-conditioned adapter: it does no
> harm where the model is already accurate, and recovers large accuracy losses
> (Exchange −35%) where the model degrades under out-of-distribution non-stationary
> drift.*

## 6. Fixes to make before submission

- **"14%–188%" gain figure (Mục 5.1)** doesn't match the tables (max cell ≈ 29%).
  Looks like a decimal-comma extraction issue (`1,4%`→`14%`, `18,8%`→`188%`).
  Reconcile with Bảng 4.3.
- **"Low memory" claim** is only honest for the frozen/adapter-only variant (no
  online head fine-tuning). Qualify it, or present the frozen variant that earns it.

## 6b. Mechanism (ablation result — adopt this framing)

The Exchange ablations are in (`documents/SSF_ABLATION_MECHANISM.md`): PROCEED's win
is a **learned static, per-layer additive SHIFT** re-calibrating the frozen backbone,
NOT drift tracking. Static SSF (zero drift) is within +0.8% of the drift-conditioned
adapter; shift-only within +1.3%; the encoder choice within ≤2%; drift/scales earn
their keep only at the most-drifted corner (1536_336). **Do not claim "tracks drift"**
— say "a static SSF/shift re-calibration; drift-conditioning adds ≤1% on Exchange."
The −29% headline and the forgetting story are unchanged; only the mechanism wording
changes. This also unifies the earlier findings (lr=0 gradient-free, capacity-invariant
cd16≈cd200 — all the same low-dimensional static shift).

## 7. Ablation plan (implemented; safe/online-FT first)

New flag `--concept_mode {dual,shared,current,stats}` (see `adapter/proceed.py`):
- `dual` (default): two encoders, drift = c(X_t) − c(X_{t−H}). *Current method.*
- `shared`: one shared encoder E′ for both X_t and X_{t−H} (recent truncated to
  seq_len), drift = c_t − c_{t−H}. *Does the second encoder matter?*
- `current`: only E′(X_t) = c_t fed to the generator, **no drift subtraction**.
  *Does the drift signal matter, or is the current concept enough?*
- `stats`: data-space per-channel moment featurizer (mean/std/last/trend), drift =
  s(X_t) − s(X_{t−H}), generator in_dim = enc_in·n_stats. *Is the learned concept
  encoder necessary, or does a hand-crafted drift descriptor suffice?*

Plus capacity sweep `cdim{32,64,128} × bneck{8,16,32}` (existing flags).

Scripts: `ttm-scripts/ablation-concept/` (run on the online-FT setting, i.e.
`--online_method Proceed` WITHOUT `--freeze_online`). Reserved for the frozen phase
(after reviewer OK): lr=0 amortization, the {head-training}×{adapter} factorial,
and the drift=0 control.
