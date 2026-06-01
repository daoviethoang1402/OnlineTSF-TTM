# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OnlineTSF-TTM is an online time series forecasting framework implementing **PROCEED** (Proactive Model Adaptation Against Concept Drift), published at KDD 2025. The key constraint is **no information leakage**: ground truth for a window is only used after the full forecast horizon has passed.

Paper: https://arxiv.org/pdf/2412.08435  
Original repo: https://github.com/SJTU-DMTai/OnlineTSF

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Pretraining (example)
python run.py --model PatchTST --dataset ETTh1 --seq_len 96 --pred_len 96 \
  --learning_rate 0.0001 --train_epochs 100

# Online testing with PROCEED
python run.py --model PatchTST --dataset ETTh1 --seq_len 96 --pred_len 96 \
  --online_method Proceed --concept_dim 200 --bottleneck_dim 32 \
  --online_learning_rate 0.001 --pretrain

# Using pre-configured scripts
bash scripts/pretrain/PatchTST/ETTh2.sh
bash scripts/online/PatchTST/Proceed/ETTh2.sh

# Multi-GPU
python -m torch.distributed.launch --nproc_per_node=2 run.py \
  --model PatchTST --dataset ETTh1 --online_method Proceed --pretrain
```

Key `run.py` flags:
- `--model`: PatchTST, iTransformer, DLinear, TCN, TinyTimeMixer, etc.
- `--dataset`: ETTh1/h2, ETTm1/m2, ECL, Weather, Traffic, Exchange, Illness
- `--online_method`: Proceed, Online, OneNet, SOLID, FSNet, DER++, ER, Naive
- `--border_type online`: 20/5/75 train/val/test split (default)
- `--pretrain`: Load pretrained checkpoint before online phase

## Architecture

### Three-Phase Pipeline

**Phase 1 — Pretraining** (`exp/exp_main.py`): Standard supervised training on 20% historical data. Backbone model is optionally wrapped with PROCEED adapter. Best checkpoint saved via early stopping.

**Phase 2 — Validation Update** (`exp/exp_proceed.py:update_valid()`): Model processes validation data sequentially with an alternating freeze/unfreeze strategy to prime the adapter for the distribution before the online phase.

**Phase 3 — Online Testing** (`exp/exp_online.py`): Test data is processed strictly sequentially (one sample at a time). For each step, the model updates on `(x_recent, y_recent)` then evaluates on `(x_current, y_current)`. The two windows never overlap — this is the no-leakage guarantee.

### PROCEED Adapter (`adapter/proceed.py`)

Wraps any backbone model without modifying it:

1. **Concept extraction**: Two MLPs (`mlp1`, `mlp2`) map the current input and recent batch to concept vectors.
2. **Drift detection**: `drift = concept_current - concept_historical`
3. **Adaptation generation**: `AdaptGenerator` (`adapter/module/generator.py`) maps drift through a bottleneck projection to per-layer scale/shift parameters.
4. **Application**: `Down_Up` adapters (`adapter/module/down_up.py`) apply `scale_in`, `scale2`, and `shift` around each frozen `Linear`/`Conv1d`/`LayerNorm`/`BatchNorm1d` layer.

The backbone weights are frozen during online phases. Only adapter weights update.

### Freezing Strategy (critical for PROCEED correctness)

| Phase | Adapter weights | Adapter bias | Backbone |
|---|---|---|---|
| Pretraining | Update | Update | Update |
| Val/Online recent batch | Frozen | Update | Frozen |
| Val/Online current batch | Update | Frozen | Frozen |

Methods: `Proceed.freeze_adapter()`, `Proceed.freeze_bias()`, `backbone.requires_grad_(False)`.

### Data Pipeline (`data_provider/data_loader.py`)

`Dataset_Recent` wraps any base dataset to emit paired samples `((x_recent, y_recent), (x_current, y_current))` with no temporal overlap. The factory in `data_factory.py` selects the right dataset class based on `--online_method`.

### Configuration (`settings.py`)

Centralized lookup tables for:
- `data_settings`: per-dataset CSV filename, target column, feature dimensions
- `hyperparams`: per-model default d_model, e_layers, n_heads, etc.
- `pretrain_lr_online_dict`: per-model, per-dataset tuned online learning rates

`run.py` merges CLI args with these defaults before constructing the Exp class.

## Key PROCEED Hyperparameters

| Arg | Paper symbol | Default | Effect |
|---|---|---|---|
| `--concept_dim` | d_c | 200 | Concept vector size; higher = more drift sensitivity |
| `--bottleneck_dim` | r | 32 | Generator bottleneck; lower = stronger regularization |
| `--ema` | — | 0 | EMA weight for smoothing concept history |
| `--tune_mode` | — | down_up | Adapter variant: `down_up` or `all_down_up` |
| `--act` | — | sigmoid | Activation on adapter output |
| `--freeze` | — | True | Freeze backbone during online phase |

## Adding a New Model

1. Add `models/YourModel.py` with a `Model` class matching the `forward(x_enc, x_mark_enc, x_dec, x_mark_dec)` signature.
2. Register in `run.py` under `Exp_Main._build_model()` (the model dict lookup).
3. Add default hyperparams to `settings.py:hyperparams`.
4. Add per-dataset online LRs to `settings.py:pretrain_lr_online_dict` if needed.
5. The PROCEED adapter wraps the model automatically; no adapter-specific changes required unless the model has unusual layer types not handled by `Down_Up`.

## Adding a New Dataset

1. Place the CSV in `./dataset/`.
2. Add an entry to `settings.py:data_settings` with filename, target column, and `[enc_in, dec_in, c_out]` dims.
3. If the format differs from standard ETT/ECL, add a new `Dataset_*` class in `data_provider/data_loader.py` and register it in `data_factory.py`.
