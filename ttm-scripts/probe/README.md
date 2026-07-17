# Probe experiments — does online adaptation of TTM have headroom, and is PROCEED's edge forgetting mitigation?

Motivated by the headroom analysis: on the standard benchmarks the oracle
adapt-or-abstain ceiling over few-shot TTM is only **~0.9%**, and input drift
statistics do **not** predict which cells have headroom (ETTh2 vs ETTm2 are the
same data with identical drift stats but 4.6× different headroom). So we select
datasets **behaviorally** (does the strong few-shot TTM degrade?), not by drift.

Terminology (confirmed against the freeze logic): **"few-shot" = TTM head
fine-tuned on the train split then frozen** — this is the "Naive (frozen)"
baseline, *not* zero-shot. reg_combo2 = few-shot + PROCEED adapter.

## Phase 0 — headroom (`headroom_probe.sh` → `analyze_probe.py`)

Four variants on untried, likely-out-of-corpus datasets (Exchange, wind; heavy
Traffic/ECL optional), seq 512, pred {96, 720}:

| variant | what | flags |
|---|---|---|
| `fewshot` | head fine-tuned, no online updates | *(no online_method)* — writes the checkpoint naive-ft loads |
| `naiveft` | plain online fine-tune (whole model) | `--online_method Online` |
| `proceedfrz` | reg_combo2 (frozen backbone) | `--online_method Proceed --freeze_online --concept_dim 64 --bottleneck_dim 8 --ema 0.9 --online_learning_rate 3e-6` |
| `proceedft` | PROCEED online fine-tune | `--online_method Proceed --concept_dim 200 --bottleneck_dim 32 --online_learning_rate 1e-4` |

**Verdict:** a dataset is a keeper if the best adaptation variant beats few-shot
by **more than ~1.5%** (clearly above the ETT/Weather ceiling). If nothing does,
these domains are as saturated as ETT/Weather and we go to controlled synthetic
drift (THU-Concept-Drift / FSNet S-A,S-G).

## Phase 1 — forgetting (`forgetting.sh` → `analyze_probe.py --forget`)

Tests the hypothesis that PROCEED-ft > naive-ft is **catastrophic-forgetting
mitigation**, via LR-robustness: sweep `online_lr ∈ {1e-4, 1e-3, 1e-2}` for
naive-ft (updates whole TTM → should forget at high LR) vs PROCEED-ft (frozen
backbone → forgetting-resistant), against the few-shot anchor. ETTh1 (control) +
Exchange (probe), pred 96.

**Signature:** naive-ft MSE climbs with LR while PROCEED-ft stays ~flat →
divergence = PROCEED mitigates forgetting.

**Limitation:** this is mechanism-targeted but indirect. A *direct* measure
(backward transfer — evaluate the post-online model on the source/train split)
needs a small hook: the framework doesn't persist the online-adapted model
(only an in-memory copy during vali), so it requires saving that model + an
eval-on-train mode. Not yet added — ask if the LR-robustness result warrants it.

## Notes

- All variants use `decoder_mode common_channel` for cross-variant consistency.
- `fewshot` must run before `naiveft` (naive-ft loads its checkpoint); verify on
  the first cell that the checkpoint actually loads rather than silently
  retraining — that handoff is the one fragile link.
- Illness (966 rows) is excluded: < seq_len 512, no training windows fit TTM.
