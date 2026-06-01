# PROCEED + TinyTimeMixer Integration

## Overview

This document describes all changes made to integrate **TinyTimeMixer (TTM)** with the **PROCEED** online adaptation framework, supporting two training approaches across all three pipeline phases (pretraining → validation → online testing).

### Two Training Approaches

| | Approach 1: Full TTM | Approach 2: Adapter-only |
|---|---|---|
| CLI flag | *(no `--freeze`)* | `--freeze` |
| TTM Backbone | Frozen (always) | Frozen (always) |
| TTM Decoder + Head | Fine-tuned during pretraining | Frozen always |
| PROCEED Adapters | Trained | Trained |
| Bias term (`more_bias`) | Yes | No |

**TTM Backbone is always frozen** per the TTM paper, in all phases and both approaches.

---

## Files Changed

### 1. `models/TinyTimeMixer.py` — Full rewrite

**Before:** Always loaded IBM pretrained weights with no config, ignored `_build_config()`, no freezing logic.

**After:**

#### `_build_config(configs)`

Builds a `TinyTimeMixerConfig` with only three values from `args` (the task dimensions) and all architecture constants hardcoded to match `ibm-granite/granite-timeseries-ttm-r1`.

```
Task-specific (from args):    context_length, num_input_channels, prediction_length
Architecture constants:       patch_length=64, patch_stride=64, d_model=192,
                               num_layers=2, adaptive_patching_levels=3, …
```

**Critical:** Generic transformer flags in `args` (`d_model=512`, `patch_len=16`, `stride=8`) must **not** be read here. Doing so causes shape mismatches across the entire backbone because the default CLI values differ from the IBM model's architecture.

#### Channel-mixing design

The IBM backbone is `mode='common_channel'`. Setting `mode='mix_channel'` would add randomly-initialised, frozen channel-mixing layers to the backbone — useless.

Instead, channel mixing is placed in the **decoder** via `decoder_mode='mix_channel'`. The decoder is separate from the frozen backbone (`tinytimemixer.decoder`) and is fine-tuned from scratch. This satisfies the channel-mixing requirement without touching pretrained backbone weights.

```
tinytimemixer.backbone  (TinyTimeMixerModel)      common_channel, ALWAYS FROZEN
tinytimemixer.decoder   (TinyTimeMixerDecoder)     mix_channel,    fine-tuned
tinytimemixer.head      (TinyTimeMixerForPredHead) —               fine-tuned
```

#### `Model.__init__`

```python
cfg = _build_config(configs)
self.tinytimemixer = TinyTimeMixerForPrediction.from_pretrained(
    pretrained_model_name, config=cfg,
    local_files_only=True, ignore_mismatched_sizes=True,
)
self.tinytimemixer.backbone.requires_grad_(False)   # always frozen
```

`ignore_mismatched_sizes=True` handles shape differences between the IBM checkpoint and the task-specific config (different `context_length`, `enc_in`, `prediction_length`, and the new `mix_channel` decoder layers). Matching layers load from IBM weights; new/reshaped layers are randomly initialised.

`pretrained_model_name` is read from `args.pretrained_model_name` (added to `run.py`). Pass an empty string to train from scratch.

#### Three hooks for the pipeline

| Hook | Called by | Purpose |
|---|---|---|
| `post_proceed_init()` | `Proceed.__init__` | Re-freezes backbone after `add_adapters_()` — for approach 1, Down_Up wrappers are created with `freeze_weight=False` which inadvertently marks backbone parameters as trainable. |
| `freeze_head()` | `Exp_Proceed.update_valid()` | Permanently freezes decoder+head before val/online phases. Sets `_head_frozen_for_online=True` so `requires_grad_(True)` no longer restores them. |
| `requires_grad_(bool)` | `Proceed` (val/online loop) | Overrides `nn.Module` default. Backbone is unconditionally kept frozen. Decoder+head follow the call only during pretraining; once `freeze_head()` has been called they remain frozen regardless. |

#### Why `requires_grad_` must be overridden

In `Exp_Proceed.update_valid()`, PROCEED calls `backbone.requires_grad_(False)` before updating on `current_batch`, then `backbone.requires_grad_(True)` after, to restore the model for the next `recent_batch` iteration. Without the override, the second call would unfreeze the TTM backbone. The override intercepts this and:

- `requires_grad_(False)` → freezes decoder+head (backbone was already frozen)
- `requires_grad_(True)` → restores decoder+head only if `_head_frozen_for_online` is `False` AND `_ttm_head_trainable` is `True`

---

### 2. `adapter/proceed.py` — 4-line addition

```python
self.backbone = add_adapters_(backbone, args)
# Let the backbone re-enforce its own internal freeze policy after adapters
# are injected (e.g. TTM always keeps its pretrained backbone frozen).
if hasattr(self.backbone, 'post_proceed_init'):
    self.backbone.post_proceed_init()
```

`add_adapters_()` walks the model and replaces `Linear`/`Conv1d`/`LayerNorm`/`BatchNorm1d` layers with `Down_Up` wrappers. For approach 1 (`args.freeze=False`), these wrappers are created with `freeze_weight=False`, setting the wrapped weight's `requires_grad=True` — including weights inside the frozen TTM backbone. The `post_proceed_init()` hook call immediately corrects this.

The hook is guarded by `hasattr`, so it is a no-op for all other models.

---

### 3. `exp/exp_proceed.py` — 4-line addition

```python
# Freeze TTM decoder+head (and any model with a freeze_head hook) before the
# val/online loop so they are excluded from the optimizer and never updated
# during adaptation phases, regardless of args.freeze.
if hasattr(self._model.backbone, 'freeze_head'):
    self._model.backbone.freeze_head()
```

Inserted at the top of `update_valid()`, before `_select_optimizer()` is called. Because `_select_optimizer()` filters by `requires_grad`, calling `freeze_head()` first ensures decoder+head parameters are excluded from the val/online optimizer in both approaches.

`update_valid()` always runs before `online()` in `run.py`, so the freeze carries through to the online phase automatically.

---

### 4. `settings.py` — Three additions

```python
# hyperparams — empty dict means no architecture override; TTM ignores
# generic transformer flags (d_model, e_layers, …) in _build_config()
hyperparams['TinyTimeMixer'] = {}

# Online learning rates for PROCEED+TTM (starting point; tune per dataset)
pretrain_lr_online_dict['TinyTimeMixer'] = {
    'ETTh1': 0.0001, 'ETTh2': 0.0001, 'ETTm1': 0.0001, 'ETTm2': 0.0001,
    'Weather': 0.0001, 'ECL': 0.0001, 'Traffic': 0.0001,
    'Exchange': 0.0001, 'Illness': 0.0001,
}

# Pretraining learning rates for standalone TTM pretraining
pretrain_lr_dict['TinyTimeMixer'] = {
    'ETTh1': 0.0001, 'ETTh2': 0.0001, 'ETTm1': 0.0001, 'ETTm2': 0.0001,
    'Weather': 0.0001, 'ECL': 0.0001, 'Traffic': 0.0001,
    'Exchange': 0.0001, 'Illness': 0.0001,
}
```

Without these entries, `run.py` raises a `KeyError` when `--pretrain` is passed (it looks up `pretrain_lr_online_dict[model][dataset]` at line 394).

---

### 5. `run.py` — One argument added

```python
parser.add_argument('--pretrained_model_name', type=str,
                    default='ibm-granite/granite-timeseries-ttm-r1',
                    help='HuggingFace model id or local path for TinyTimeMixer '
                         'pretrained weights. Set to empty string to train from scratch.')
```

Passed through `args` to `Model.__init__` via `getattr(configs, 'pretrained_model_name', ...)`.

---

## Running the Pipeline

### Approach 1 — Fine-tune decoder+head + PROCEED adapters

```bash
# First run: trains from IBM weights, saves checkpoint
python run.py --model TinyTimeMixer --dataset ETTh1 \
  --seq_len 512 --pred_len 96 \
  --online_method Proceed \
  --concept_dim 200 --bottleneck_dim 32 \
  --online_learning_rate 0.0001

# Subsequent runs: load saved checkpoint, skip retraining
python run.py --model TinyTimeMixer --dataset ETTh1 \
  --seq_len 512 --pred_len 96 \
  --online_method Proceed \
  --concept_dim 200 --bottleneck_dim 32 \
  --online_learning_rate 0.0001 --pretrain
```

### Approach 2 — Adapter-only (frozen TTM)

```bash
python run.py --model TinyTimeMixer --dataset ETTh1 \
  --seq_len 512 --pred_len 96 \
  --online_method Proceed --freeze \
  --concept_dim 200 --bottleneck_dim 32 \
  --online_learning_rate 0.0001
```

### Train from scratch (no IBM weights)

```bash
python run.py --model TinyTimeMixer --dataset ETTh1 \
  --seq_len 512 --pred_len 96 \
  --online_method Proceed \
  --pretrained_model_name "" \
  --concept_dim 200 --bottleneck_dim 32 \
  --online_learning_rate 0.0001
```

---

## Freezing State Table (all phases, both approaches)

| Phase | TTM Backbone | TTM Decoder + Head | PROCEED mlp1/mlp2/generator weights | PROCEED generator bias |
|---|---|---|---|---|
| Pretraining (approach 1) | Frozen | **Trainable** | Trainable | Trainable |
| Pretraining (approach 2) | Frozen | Frozen | Trainable | — |
| Val — recent batch | Frozen | Frozen | Frozen | **Trainable** (approach 1 only) |
| Val — current batch | Frozen | Frozen | **Trainable** | Frozen |
| Online — recent batch | Frozen | Frozen | Frozen | **Trainable** (approach 1 only) |
| Online — inference | Frozen | Frozen | Frozen | Frozen |

---

## Known Constraints

- `seq_len=512` matches the IBM model's `context_length`. Other values (`96`, `192`, `336`) will cause the patcher and head to reinitialise from scratch since `patch_length=64` and `patch_stride=64` are fixed. The backbone MLP layers (shape `[d_model, ...]`) are unaffected by `seq_len` and always load from IBM weights.
- The decoder's `mix_channel` layers are always randomly initialised (IBM decoder is `common_channel`), so they benefit from the pretraining phase regardless of approach.
- Learning rates in `pretrain_lr_online_dict` and `pretrain_lr_dict` are conservative starting points (`1e-4`). Tune per dataset.
