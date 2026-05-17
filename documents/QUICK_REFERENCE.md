# PROCEED + PatchTST - Quick Reference Guide

## What is this pipeline?

An **online time series forecasting system** that:
1. Uses PatchTST (transformer-based) as the forecast backbone
2. Wraps it with PROCEED adapters to detect and adapt to concept drift
3. Updates continuously on new data **without information leakage**

---

## Three Execution Phases

### 🔧 Phase 1: Pretraining
```
Full historical data → Standard supervised learning → Pretrained checkpoint
```
- Train backbone + adapters on historical data
- Use early stopping on validation set
- Save best model checkpoint

### 🎯 Phase 2: Validation Adaptation (Optional)
```
Validation data → Sequential adaptation → Updated model state
```
- Process validation data chronologically
- Freeze/unfreeze strategy prevents forgetting
- Model adapts to validation period distribution

### 📊 Phase 3: Online Testing
```
Test data (streaming) → Sequential prediction + adaptation → Performance metrics
```
- Process each test sample sequentially
- Update model on recent context (historical patterns)
- Predict on current window (concept drift detected here)
- NO information leakage (ground truth only available after forecast)

---

## Core Technical Innovation: Concept Drift Detection

```
┌─────────────────────────────────────────────────┐
│ Current data (t)                                │
│ ↓                                               │
│ MLP1 → Current Concept (what's happening now)   │
│                          ↓                      │
│                   Drift = Δ Concept             │
│                          ↑                      │
│ MLP2 ← Recent Concept (historical patterns)     │
│ ↑                                               │
│ Recent accumulated data (t-recent)              │
└─────────────────────────────────────────────────┘
       ↓
Adaptation Generator
       ↓
Layer-specific parameters (scale, shift, bias)
       ↓
Apply Down-Up Adapters to frozen backbone
```

**Key insight**: Drift = Current - Recent. When significant, adapters adjust model behavior.

---

## Architecture Layers

### PatchTST Backbone
```
Input → Patching → Transformer Encoder → Output Projection → Forecast
```
- Divides series into patches (default 16 timesteps)
- Transformer captures patch-level and temporal dependencies
- Optional trend/residual decomposition

### PROCEED Wrapper
```
Backbone (frozen or partially frozen)
    ↓
Down-Up Adapters (applied to every Linear/Conv1d layer)
    ↓
Adaptive scaling parameters (learned from concept drift)
```

Down-Up Adapter:
```
Input → [scale by (1 + scale_in)] → Weight × (1 + scale_out) → [shift] → Output
```

---

## Data Flow in Online Phase

```
Dataset_Recent wrapper provides paired batches:

Batch 1: (recent_context, current_window)
         ↓
         ├─→ Train on recent_context (adapt to patterns)
         │   [freezes adapter weights, updates backbone and adapter bias]
         │
         └─→ Test on current_window (measure performance)
             [unfreezes adapter weights, detects drift]

Batch 2: (recent_context, current_window)
         ↓ (recent_context = accumulated previous predictions + current)
         ...
```

**No Leakage**: recent_context uses only past data, current_window uses only available labels.

---

## Key Hyperparameters

| Parameter | Effect | Default |
|-----------|--------|---------|
| `--concept_dim` | Drift detection sensitivity | 200 |
| `--bottleneck_dim` | Adaptation model compression | 32 |
| `--ema` | Smoothing for concepts (0=off) | 0 |
| `--freeze` | Freeze backbone, train only adapters | True |
| `--tune_mode` | Adapter type (down_up, all_down_up) | down_up |
| `--online_learning_rate` | Learning rate in online phase | varies |
| `--seq_len` | Input window size | 96 |
| `--pred_len` | Forecast horizon | 96 |

---

## Freezing Strategy (PROCEED-specific)

Prevents **catastrophic forgetting** while enabling **targeted adaptation**:

| Stage | Adapter Weights | Adapter Bias | Backbone Weights | Backbone Bias | Purpose |
|-------|-----------------|--------------|------------------|---------------|---------|
| Recent | ❌ Frozen | ✓ Update | ❌ Frozen | ✓ Update | Learn temporal patterns |
| Current | ✓ Update | ✓ Update | ❌ Frozen | ✓ Update | Detect & adapt to drift |

**Note**: Adapter bias is the extra learnable bias that PROCEED adds to each layer, NOT backbone bias.

This balance:
- Preserves learned features (frozen backbone)
- Adapts to changing distributions (active adapter)
- Prevents overfitting to noise (selective bias updates)

---

## Data Processing Pipeline

```
Raw Time Series
    ↓
[Train : Val : Test] split (20:5:75 for online mode)
    ↓
─────────────────────────────────────────
│ TRAIN (Pretraining)                    │
│ Standard DataLoader                    │
│ Single samples: (x_t, y_t)             │
└─────────────────────────────────────────
    ↓
─────────────────────────────────────────
│ VAL + TEST (Online)                    │
│ Dataset_Recent wrapper                 │
│ Paired samples: (recent, current)      │
└─────────────────────────────────────────
    ↓
Sequential processing without shuffling
    ↓
Performance metrics (MSE, MAE)
```

---

## Running the Pipeline

```bash
# 1. Pretraining phase
python run.py \
  --model PatchTST \
  --dataset ETTh1 \
  --online_method Proceed \
  --pretrain \
  --learning_rate 0.0001 \
  --seq_len 96 --pred_len 96

# 2. Online testing phase (automatic if pretraining succeeds)
# Evaluation happens automatically in exp.online()
```

---

## File Structure

```
adapter/
  ├── proceed.py (main PROCEED wrapper)
  └── module/
      ├── down_up.py (Down-Up adapter implementation)
      ├── generator.py (concept → parameters mapping)
      └── base.py (base adapter class)

exp/
  ├── exp_main.py (Exp_Main: pretraining)
  ├── exp_online.py (Exp_Online: online loop)
  └── exp_proceed.py (Exp_Proceed: PROCEED-specific)

models/
  └── PatchTST.py (backbone model)

data_provider/
  ├── data_loader.py (Dataset_Recent for pairing)
  └── data_factory.py (data loading utilities)
```

---

## Summary Table

| Aspect | PROCEED | PatchTST |
|--------|---------|----------|
| **Role** | Adaptation mechanism | Forecasting backbone |
| **Learns** | Concept drift → adaptations | Time series patterns |
| **Updates During Online** | Always (if unfrozen) | Conditionally (if unfrozen) |
| **Key Inputs** | Concept drift vector | Patched time series |
| **Key Outputs** | Per-layer parameters | Forecast values |
| **Frozen by Default** | No | Yes (in online phase) |

---

## Performance Characteristics

- **Pretraining**: Standard training speed (batch processing)
- **Online Phase**: Slower (sequential processing), but low memory footprint
- **Adaptation Overhead**: Minimal (MLP forward pass + Down-Up scaling)
- **Information Leakage**: Zero (sequential processing ensures this)

---

## Next Steps

1. **Understand Concept**: Read the drift computation in `Proceed.generate_adaptation()`
2. **Trace Online Flow**: Follow `Exp_Proceed.online()` → `Exp_Online.online()`
3. **Inspect Adapters**: Check `Down_Up` layer in `adapter/module/down_up.py`
4. **Experiment**: Vary `--concept_dim`, `--bottleneck_dim`, `--online_learning_rate`
