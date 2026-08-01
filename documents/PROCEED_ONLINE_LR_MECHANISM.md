# What `online_learning_rate` Actually Adjusts (frozen PROCEED + TTM)

A mechanistic reference explaining what the online lr touches, why it is
near-inert in the frozen regime, and why the working rule is "lr tracks *drift*,
not *horizon*." Ties together `PROCEED_TTM_HYPERPARAM_NOTES.md` (the empirical lr
sweeps) and `CAPACITY_SELECTION_STUDY.md` (the falsified horizon rule).

## The adaptation generator's parameters

Each layer-group generator is a `Bottleneck` (`adapter/module/generator.py`):

`adaptation = W_up · act( W_down · drift + b_0 ) + b_out`

| symbol | code | shape | role |
|---|---|---|---|
| W_down | `weights[0]` | in→bneck, **shared** across layers | drift → hidden |
| W_up | `weights[-1]` | bneck→out, **shared** | hidden → adaptation |
| b_0 | `biases[0]` = `biases[:-1]` | (n_layers, 1, bneck), **per-layer** | inside the mapping |
| b_out | `biases[-1]` | (n_layers, 1, out), **per-layer** | additive offset on the output |

Two shared weight matrices + two per-layer bias tensors. `b_out` exists only when
`more_bias = not args.freeze` is True.

## The freezing split (deliberate)

- `freeze_adapter(True)` freezes `mlp1/mlp2`, **both weights**, and **`biases[0]`**
  (`biases[:len(weights)-1]`).
- `freeze_bias(freeze)` controls **`biases[-1]`** separately; `update_valid` leaves
  it **trainable** entering the online loop.

So during the online **test** loop (`Exp_Proceed.online` sets `freeze_adapter(True)`
for the whole loop; the current window is eval-only, only the recent window is
updated):

> **`online_learning_rate` moves exactly one thing: `biases[-1]`** — the
> drift-**independent** per-layer output offset. Everything else (backbone, head,
> concept MLPs, both weights, `biases[0]`) is frozen.

Edge cases:
- `--freeze` (⇒ `more_bias=False`, no `biases[-1]`) **or** `freeze_bias(True)` →
  `online_lr` adjusts **nothing**; the run is pure amortized inference.
- **Original PROCEED** (no `--freeze_online`): the decoder/head keep fine-tuning
  online, so `online_lr` *additionally* updates those backbone params — the
  dominant and risky effect (this is where naive online fine-tuning
  catastrophically forgets). The adapter *mapping* (weights, `biases[0]`) stays
  frozen during the online-test loop in **every** variant; it is learned in
  pretraining + the validation-update phase, then applied.

## Why lr is near-inert — and why it still tracks drift

The **drift → scale/shift mapping** (`W_down`, `b_0`, `W_up`) is frozen. The
adaptation still changes every step, but only because the **drift** changes (the
`recent_batch` buffer + current input feed a fresh drift into the frozen
generator). That is amortized inference in the forward pass — **`online_lr` has
nothing to do with it.**

Because `online_lr` only nudges a small constant offset (`biases[-1]`):
- On **stationary** data it is **near-inert** — the `lr=0 ≈ lr=3e-6` attribution
  result. The win is entirely amortized.
- On **strongly-drifting** data (Exchange) that offset slowly **tracks the drift
  level** across the long test, so a **higher lr tracks faster and helps**. This
  is the mechanism behind the capacity-study finding that **lr tracks the
  dataset/drift, not the horizon** — Exchange wanting 3e-6 is the one place online
  bias-tracking earns its keep; everywhere else 1e-6 (minimal adaptation) is best.

## Design note (not adopted): unfreezing `biases[0]` online

Making `biases[0]` (the in-mapping bias) trainable online would dial plasticity
into the *mapping* itself — more power to track strong drift, but it re-tunes the
drift→adaptation shape on the recent window each step. Given the whole project
shows over-adaptation hurts on stationary data (U-shaped lr; "wins small, loses
big"), this is expected to help only where drift is real (Exchange) and hurt
elsewhere — the same asymmetry the guard hit. **Not adopted**; the amortized,
frozen-mapping design is what carries the results.
