# PROCEED + PatchTST Pipeline Architecture

## Overview
This codebase implements **PROCEED** (Proactive Model Adaptation Against Concept Drift), an online time series forecasting framework combined with **PatchTST** as the backbone model. The pipeline focuses on:
- **No information leakage**: Updates happen with ground truth data only after the forecast horizon
- **Online model adaptation**: Continuous model updates as new data arrives
- **Concept drift handling**: Detects and adapts to changes in data distribution

---

## Architecture Components

### 1. **Backbone Model: PatchTST**
**Location**: `models/PatchTST.py`

PatchTST (Patch Time Series Transformer) is a transformer-based forecasting model that:
- Divides input time series into **patches** (configurable via `--patch_len`)
- Uses transformer encoder to capture temporal dependencies
- Optionally decomposes series into trend and residual components
- Applies RevIN (Reversible Instance Normalization) for normalization

**Key architecture**:
```
Input (B, seq_len, enc_in)
  ↓
[Optional: Series Decomposition]
  ↓
Permute (B, enc_in, seq_len)
  ↓
PatchTST_backbone (patching, transformer encoder, head)
  ↓
Output (B, pred_len, enc_in)
```

---

### 2. **PROCEED Adapter Layer**
**Location**: `adapter/proceed.py`

PROCEED wraps the backbone model with **proactive adaptation** capabilities:

#### Core Components:

**a) Concept Generation**
- **MLP1**: Extracts current timestep concept from input `x`
  - Input: `(B, seq_len, enc_in)`
  - Output: `(B, concept_dim)`
  - Formula: `concept = MLP1(x).mean(-2)`

- **MLP2**: Extracts historical concept from recent batch
  - Input: `(B, seq_len + pred_len, enc_in)` (accumulated recent context)
  - Output: `(B, concept_dim)`

**b) Concept Drift Computation**
```python
drift = concept_current - concept_recent
# Optional EMA smoothing for stability
```

**c) Adaptation Generator**
- Maps concept drift → adaptation parameters for each layer
- Uses bottleneck projection: `concept_dim → bottleneck_dim → layer_params`
- Clipping ensures bounded adaptation magnitudes

**d) Adapter Application**
- **Down-Up Adapters**: Applied to Linear, Conv1d, LayerNorm, BatchNorm1d layers
- Each adapter scales input features (down) and output features (up)
- Formula: `output = (weight * scale2) * input * scale + bias + shift`

---

### 3. **Experimental Framework**

#### Class Hierarchy:
```
Exp_Basic (base utilities)
  ↓
Exp_Main (standard training)
  ↓
Exp_Online (online learning loop)
  ↓
Exp_Proceed (PROCEED-specific logic)
```

**Key Experiment Classes**:

| Class | Purpose |
|-------|---------|
| `Exp_Main` | Standard pretraining on historical data |
| `Exp_Online` | Online learning with dataset streaming |
| `Exp_Proceed` | PROCEED-specific phases and freezing |

---

## Full Pipeline Execution

### Phase 1: Pretraining (if `--pretrain`)

**Location**: `exp/exp_main.py`

```python
# Setup
1. Load backbone model (PatchTST)
2. Wrap with PROCEED adapter → add_adapters_()
3. Initialize optimizer (default: Adam)
4. Load training data

# Training loop (epochs)
for epoch in range(train_epochs):
  for batch in train_loader:
    # Standard supervised learning
    loss = criterion(model(batch), labels)
    loss.backward()
    optimizer.step()
  
  # Validation
  val_loss = evaluate(val_loader)
  early_stop_check(val_loss)

# Save checkpoint
```

**Important Flags**:
- `--freeze`: Freeze backbone weights, only train adapter
- `--pretrain`: Load pretrained weights before online phase
- Training hyperparams from `settings.py`

---

### Phase 2: Validation Data Update (if `--do_valid`)

**Location**: `exp/exp_proceed.py` → `update_valid()`

Adapts the model to validation data with **special handling**:

```python
# Setup
valid_loader = get_dataloader(valid_data, 'online')
# Wraps data as (recent_batch, current_batch) pairs

# Update loop
for recent_batch, current_batch in valid_loader:
  # Step 1: Learn from recent historical context
  freeze_adapter(True)      # Fix adapter weights (mlp1, mlp2, down-up weights frozen)
  freeze_bias(False)        # Update adapter bias (extra per-layer bias in PROCEED adapters)
  _update_online(recent_batch)
  freeze_bias(True)         # Fix adapter bias
  
  # Step 2: Learn from current data
  freeze_adapter(False)     # Update adapter weights (mlp1, mlp2, down-up weights unfrozen)
  backbone.requires_grad_(False)  # if frozen model
  _update_online(current_batch)

# Result: Model sees validation data without leakage
```

---

### Phase 3: Online Testing (if `--online_method`)

**Location**: `exp/exp_online.py` → `online()`

Processes test data sequentially, mimicking real-time deployment:

```python
# Setup
online_loader = get_dataloader(test_data, 'online')
# Data comes as (recent_batch, current_batch) pairs

# Online loop
for i, (recent_batch, current_batch) in enumerate(online_loader):
  # Train on recent context (no gradient for prediction)
  model.train()
  _update_online(recent_batch)  # Learn temporal patterns
  
  # Predict on current window
  model.eval()
  with torch.no_grad():
    outputs = forward(current_batch)
  
  # Compute metrics
  mse, mae = evaluate(outputs, labels)
  update_metrics(...)

# Return: aggregated MSE, MAE
```

**Data Flow in Online Phase**:
```
recent_batch: (B, recent_seq_len, enc_in)
              Historical context for learning
              ↓
              Used to update adapter weights
              
current_batch: (B, pred_len, enc_in)
               Current window for prediction
               ↓
               Used to evaluate model performance
```

---

### Phase 4: Information Leakage Handling

**Location**: `exp/exp_online.py` → `online_information_leakage_PatchTST()`

Special path for PatchTST when `--leakage` flag:

```python
# Alternative (not recommended, for benchmarking):
for recent_data, current_data in online_loader:
  # Forward pass on recent data (no gradient)
  with torch.no_grad():
    outputs = forward(recent_data)
  
  # Train on current data
  _update_online(current_data)
```

This allows "leakage" of future information (for comparative analysis).

---

## Data Flow Details

### Dataset Classes:

**`Dataset_Recent`** (data_provider/data_loader.py)
- Wraps standard dataset to provide recent context pairs
- Creates (recent_sample, current_sample) tuples
- Ensures no overlap between recent and current windows
- Parameters:
  - `recent_num`: How many recent samples to include
  - `gap`: Gap between recent and current windows

**Online Data Loader**:
```python
# Standard: single samples
loader[i] = (x_t, y_t)

# Online: paired samples
loader[i] = ((x_recent, y_recent), (x_current, y_current))
```

---

## Key Configuration Parameters

### Model Configuration (`settings.py`):
```python
--seq_len 96          # Input lookback window
--pred_len 96         # Forecast horizon
--patch_len 16        # Patch size for PatchTST
--stride 8            # Stride between patches

--d_model 128         # Transformer hidden dimension
--n_heads 16          # Attention heads
--e_layers 3          # Encoder layers
--dropout 0.2         # Dropout rate
```

### PROCEED Configuration (`run.py`):
```python
--concept_dim 200           # d_c: concept representation dimension
--bottleneck_dim 32         # r: bottleneck for adaptation
--tune_mode 'down_up'       # Adapter type
--act 'sigmoid'             # Activation for concept drift
--ema 0                     # EMA smoothing for concepts (0 = off)
--wo_clip False             # Clip adaptation magnitude
--individual_generator False # Share generator across layers
```

### Training Configuration:
```python
--learning_rate 0.0001          # Pretraining LR
--online_learning_rate 0.001    # Online phase LR
--train_epochs 100              # Pretraining epochs
--patience 5                    # Early stopping patience
--batch_size 32                 # Training batch size
```

### Data Split (`--border_type online`):
```
Default: 20% training | 5% validation | 75% online testing
Borders controlled by get_borders() in settings.py
```

---

## Adapter Freezing Strategy

**In PROCEED**:

| Phase | Adapter | Bias (if not frozen) | Backbone |
|-------|---------|----------------------|----------|
| Pretraining | Update | Update | Update |
| Val Recent | Frozen | Update | Frozen |
| Val Current | Update | Frozen | Frozen |
| Online Recent | Frozen | Update | Frozen |
| Online Current | Update | Frozen | Frozen |

This **prevents catastrophic forgetting** while allowing targeted updates.

---

## Concept Drift Detection

**Mechanism**:
```python
# Continuous concepts
concept_current = MLP1(x_current).mean()
concept_recent = MLP2(accumulated_recent_data).mean()

# Drift magnitude
drift = ||concept_current - concept_recent||

# Adaptation strength (learned per layer)
adaptation = generator(drift)

# Application
layer_output = layer(input, adaptation)
```

**EMA Smoothing** (optional):
```python
if args.ema > 0:
  recent_concept = α * recent_concept_prev + (1-α) * recent_concept_new
```

---

## No Information Leakage Guarantee

**Key Insight**: Validation and test data are processed sequentially:

1. **Recent Batch Update**: Uses historical context **before** current prediction window
   - No labels for future time steps
   - Safe for online deployment

2. **Current Batch Evaluation**: Uses ground truth **only after** forecast
   - Labels available because forecast period has passed
   - Matches real-time deployment scenario

3. **RecenetBatch Accumulation**: `recent_batch = cat([batch[0], batch[1]], -2)`
   - Accumulates predictions to build historical context
   - Ensures moving window alignment

---

## Execution Flow Diagram

```
run.py
  ↓
  1. Setup args, seeds, data splits
  ↓
  2. For each iteration (itr):
    ├─ [Optional] Pretraining Phase
    │  └─ exp.train() → Exp_Main
    │     ├─ Load training data
    │     ├─ Standard supervised learning
    │     └─ Save checkpoint
    │
    ├─ [Optional] Validation Update
    │  └─ exp.update_valid() → Exp_Proceed
    │     ├─ Adapt to validation data
    │     └─ Freeze/unfreeze strategy
    │
    └─ Online Testing Phase
       └─ exp.online() → Exp_Proceed/Exp_Online
          ├─ Sequential processing
          ├─ Update on recent context
          ├─ Evaluate on current window
          └─ Report MSE, MAE
```

---

## Summary

**PROCEED + PatchTST = Online Time Series Forecasting with Concept Drift Adaptation**

| Component | Purpose |
|-----------|---------|
| **PatchTST** | Powerful transformer backbone with patching |
| **PROCEED Adapter** | Detects drift, generates adaptive parameters |
| **Concept MLPs** | Extracts temporal patterns and detects changes |
| **Down-Up Adapters** | Efficiently applies adaptations to each layer |
| **Online Loop** | Processes data sequentially without leakage |
| **Freezing Strategy** | Prevents catastrophic forgetting during adaptation |

The pipeline ensures no information leakage while maintaining continuous adaptation to non-stationary time series data.
