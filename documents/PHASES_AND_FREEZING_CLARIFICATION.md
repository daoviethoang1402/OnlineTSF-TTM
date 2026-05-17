# PROCEED Training Phases & Freezing Strategy - Clarification

**Last Updated**: 2026-05-16  
**Status**: Addresses all 3 questions with corrections to documentation

---

## Question 1: Are there 3 phases: pre-training, validation, and online?

### ✅ Answer: YES, there are **exactly 3 main phases**

```
Phase 1: Pretraining       (20% of data)
    ↓
Phase 2: Validation Adapt  (5% of data)  ← Optional, controlled by --do_valid
    ↓
Phase 3: Online Testing    (75% of data)
```

### Phase Details:

#### **Phase 1: Pretraining** (`Exp_Main`)
- **Location**: `exp/exp_main.py`
- **Data**: Historical training data (batch_size=32, shuffled)
- **Freezing**: ✅ Both backbone and adapters are **trainable** (no freezing)
- **Purpose**: Learn baseline features from historical distribution

#### **Phase 2: Validation Adaptation** (`Exp_Proceed` or `Exp_Online`)
- **Location**: `exp/exp_proceed.py::update_valid()` (if using Proceed) or `exp/exp_online.py::update_valid()`
- **Data**: Validation data (single samples, sequential)
- **Freezing**: ❌ **Alternating freeze strategy** (see Question 2)
- **Purpose**: Prepare model for online testing by adapting to validation distribution
- **Control**: `--do_valid` flag (default: True)

#### **Phase 3: Online Testing** (`Exp_Proceed` or `Exp_Online`)
- **Location**: `exp/exp_proceed.py::online()` or `exp/exp_online.py::online()`
- **Data**: Test data (single samples, sequential)
- **Freezing**: ❌ **Same alternating freeze strategy as Phase 2**
- **Purpose**: Continuous evaluation and adaptation in online setting
- **Control**: `--online_method` flag (e.g., "Proceed")

### Code Evidence:

From `run.py`:
```python
# Phase 1
if args.pretrain:
    exp.train()  # exp_main.py

# Phase 2
if args.do_valid:
    exp.update_valid()  # exp_proceed.py or exp_online.py

# Phase 3
results = exp.online()  # exp_proceed.py or exp_online.py
```

---

## Question 2: Why in update_valid, if joint_update_valid is true, the bias are NOT frozen but the adapter weights ARE?

### ✅ Answer: Two Different Strategies

**Location**: `exp/exp_proceed.py` lines 34-78 (`update_valid()`)

### What is `joint_update_valid`?

A boolean flag controlling which update strategy to use during validation:

- **`joint_update_valid=False`** (default): Uses **alternating freeze strategy**
- **`joint_update_valid=True`**: Uses **unified update strategy**

### Strategy 1: Alternating (Default, `joint_update_valid=False`)

**Lines 46-62 in `exp/exp_proceed.py`:**

```python
for recent_batch, current_batch in valid_loader:
    # ========== RECENT BATCH PROCESSING ==========
    self._model.freeze_bias(False)         # Unfreeze adapter bias
    self._model.freeze_adapter(True)       # FREEZE adapter weights + mlp1/mlp2
    self._update_online(recent_batch)      # Update: backbone, adapter bias
    self._model.freeze_bias(True)          # Freeze adapter bias
    if not self.args.freeze:
        self._model.backbone.requires_grad_(False)  # Freeze backbone weights
    
    # ========== CURRENT BATCH PROCESSING ==========
    self._model.freeze_adapter(False)      # UNFREEZE adapter weights + mlp1/mlp2
    self._update_online(current_batch)     # Update: adapter weights (+ adapter bias)
    if not self.args.freeze:
        self._model.backbone.requires_grad_(True)   # Unfreeze backbone for next iteration
```

**What this does:**

| Step | Adapter Weights | Adapter Bias | Backbone |
|------|-----------------|--------------|----------|
| Recent | ❌ **FROZEN** | ✓ Can update | ❌ Frozen |
| Current | ✓ **UNFROZEN** | ✓ Can update | ❌ Frozen |

**Why?**
- Recent: Keep adapter weights fixed (as "low-rank corrections"), only adapt the extra bias
- Current: Let adapters adjust to detected drift, while backbone stays frozen

### Strategy 2: Unified (`joint_update_valid=True`)

**Lines 63-73 in `exp/exp_proceed.py`:**

```python
for recent_batch, current_batch in valid_loader:
    # No explicit freeze calls - use default trainable status
    self._update_online(recent_batch)  # Updates all trainable params
    
    # Evaluate on current batch
    if self.args.do_predict:
        self.model.eval()
        with torch.no_grad():
            outputs = self.forward(current_batch)
        self.model.train()
```

**What this does:**

| Component | Status |
|-----------|--------|
| Adapter weights | ✓ Can update |
| Adapter bias | ✓ Can update |
| Backbone | ❌ Frozen |

**Why?**
- Simpler approach: everything that can be updated, gets updated
- No alternating freeze pattern
- Single update loop instead of separate recent/current handling

### Why This Design?

The **alternating strategy** prevents catastrophic forgetting:

1. **Recent batch** (historical context):
   - Backbone frozen → preserves learned features
   - Adapter weights frozen → keeps low-rank corrections stable
   - Adapter bias updated → allows small adjustments

2. **Current batch** (prediction window):
   - Backbone frozen → preserves learned features
   - Adapter weights active → allows drift adaptation
   - Adapter bias updated → allows small adjustments

This balance ensures:
- ✅ Backbone knowledge preserved (frozen)
- ✅ Drift-aware updates enabled (active adapters)
- ✅ No overfitting (selective updates)

---

## Question 3: Based on the paper, at validation phase, which are frozen: backbone or adapter?

### ✅ Answer: **BOTH are frozen selectively**

### Complete Freezing Timeline During Validation

```
┌─────────────────────────────────────────────────────────┐
│ Validation Adaptation Phase (Alternating Strategy)      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ For Each Validation Batch Pair:                         │
│                                                         │
│ ┌─────────────────────────────────────────────────┐   │
│ │ Recent Batch Processing:                        │   │
│ ├─────────────────────────────────────────────────┤   │
│ │ Adapter Weights:  ❌ FROZEN                      │   │
│ │ Adapter Bias:     ✓ Can update                   │   │
│ │ Backbone Weights: ❌ FROZEN                      │   │
│ │ Backbone Bias:    ✓ Can update                   │   │
│ │                                                 │   │
│ │ → Updates backbone and adapter bias only        │   │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
│ ┌─────────────────────────────────────────────────┐   │
│ │ Current Batch Processing:                       │   │
│ ├─────────────────────────────────────────────────┤   │
│ │ Adapter Weights:  ✓ Can update                   │   │
│ │ Adapter Bias:     ✓ Can update                   │   │
│ │ Backbone Weights: ❌ FROZEN                      │   │
│ │ Backbone Bias:    ✓ Can update                   │   │
│ │                                                 │   │
│ │ → Updates adapters (weights + bias)             │   │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Code Implementation:

**From `exp/exp_proceed.py` lines 46-57:**

```python
# Recent Batch
self._model.freeze_bias(False)              # Line 48: Unfreeze adapter bias
self._model.freeze_adapter(True)            # Line 49: FREEZE adapter weights/mlp1/mlp2
self._update_online(recent_batch)           # Update with frozen adapters
self._model.freeze_bias(True)               # Line 51: Freeze adapter bias
if not self.args.freeze:
    self._model.backbone.requires_grad_(False)  # Line 53: Ensure backbone frozen

# Current Batch
self._model.freeze_adapter(False)           # Line 54: UNFREEZE adapter weights/mlp1/mlp2
if not self.args.freeze:
    self._model.backbone.requires_grad_(False)  # Backbone stays frozen
self._update_online(current_batch)          # Update with unfrozen adapters
if not self.args.freeze:
    self._model.backbone.requires_grad_(True)   # Restore for next iteration
```

### Key Insight: What Gets Updated in Each Step

**Recent Batch Step:**
- Backbone weights: ❌ FROZEN
- Backbone bias: ✓ Updated (not frozen)
- Adapter weights: ❌ FROZEN
- Adapter bias: ✓ Updated (not frozen)

**Current Batch Step:**
- Backbone weights: ❌ FROZEN
- Backbone bias: ✓ Updated (not frozen)
- Adapter weights: ✓ Updated (not frozen)
- Adapter bias: ✓ Updated (not frozen)

### What About Online Testing Phase?

**The online phase uses the exact same freezing strategy as validation!**

From `exp/exp_proceed.py` line 20:
```python
self.online_phases = ['val', 'test', 'online']  # All use same strategy
```

---

## Critical Documentation Errors Found & Fixed

### ❌ Error 1: Confusing "freeze_bias" with "backbone bias"

**Location**: `paper-code-analysis-qa.md` line 39

❌ **OLD**: `freeze_bias(True/False)` - Controls **backbone bias** updates

✅ **FIXED**: `freeze_bias(True/False)` - Controls **adapter bias** updates (the extra per-layer bias added by PROCEED, NOT backbone bias)

**Why this matters:**
- `freeze_bias()` is about the **extra learnable adapter bias**, not the backbone
- The backbone bias is controlled by `backbone.requires_grad_()`
- Confusion leads to incorrect understanding of the freezing strategy

### ❌ Error 2: Misleading freezing timeline in PIPELINE_EXPLANATION.md

**Location**: `PIPELINE_EXPLANATION.md` lines 145, 147

❌ **OLD**:
```python
freeze_bias(False)        # Update backbone bias
freeze_bias(True)         # Fix backbone bias
```

✅ **FIXED**:
```python
freeze_bias(False)        # Update adapter bias (extra per-layer bias in adapters)
freeze_bias(True)         # Fix adapter bias
```

### ❌ Error 3: Oversimplified freezing table in QUICK_REFERENCE.md

**Location**: `QUICK_REFERENCE.md` lines 134-137

❌ **OLD**: Single "Adapter" column that didn't distinguish weights vs. bias

✅ **FIXED**: Separate columns for "Adapter Weights" and "Adapter Bias"

---

## Summary Table: Complete Freezing Across All Phases

| Phase | Backbone Weights | Backbone Bias | Adapter Weights | Adapter Bias | Context |
|-------|------------------|---------------|-----------------|--------------|---------|
| **Pretraining** | ❌ Updated | ❌ Updated | ❌ Updated | ❌ Updated | Standard training on full historical data |
| **Validation - Recent** | ❌ Frozen | ✓ Updated | ❌ Frozen | ✓ Updated | Learn temporal patterns with fixed adapters |
| **Validation - Current** | ❌ Frozen | ✓ Updated | ✓ Updated | ✓ Updated | Detect drift with active adapters |
| **Online - Recent** | ❌ Frozen | ✓ Updated | ❌ Frozen | ✓ Updated | Learn temporal patterns with fixed adapters |
| **Online - Current** | ❌ Frozen | ✓ Updated | ✓ Updated | ✓ Updated | Detect drift with active adapters |

**Legend**: ❌ = Frozen (no gradients), ✓ = Can be updated (receives gradients)

---

## Key Architecture Details

### What is "Adapter Bias" in PROCEED?

PROCEED adds the following to the standard PatchTST backbone:

```
Standard Layer (e.g., Linear)
  ├─ Weight matrix W
  └─ Bias b

PROCEED Down-Up Adapter Layer
  ├─ Weight matrix W
  ├─ Bias b (original)
  ├─ scale_in (learned from drift)
  ├─ scale_out (learned from drift)
  ├─ shift (learned from drift)
  └─ biases[-1] = "extra adapter bias" (additional learnable parameter)
```

So "adapter bias" refers to `biases[-1]`, which is **not the backbone bias**, but an **additional parameter added by the adapter**.

### Code Evidence:

From `adapter/proceed.py` lines 90-94:

```python
def freeze_bias(self, freeze=True):
    if self.more_bias:
        for adapter in self.generator.bottlenecks.values():
            # This is the "extra bias" added by adapters, NOT backbone bias!
            adapter.biases[-1].requires_grad_(not freeze)
            adapter.biases[-1:].zero_grad(set_to_none=True)
```

---

## References

**Code Files**:
- `exp/exp_proceed.py`: Lines 34-78 (update_valid), lines 92-99 (_update_online)
- `adapter/proceed.py`: Lines 79-94 (freeze_adapter, freeze_bias)
- `exp/exp_online.py`: Lines 66-100 (update_valid in base online class)

**Documentation Files** (now corrected):
- `paper-code-analysis-qa.md`: Updated with correct explanations
- `PIPELINE_EXPLANATION.md`: Updated with clarified freezing comments
- `QUICK_REFERENCE.md`: Updated with detailed freezing table

---

## Summary

1. ✅ **3 phases exist**: Pretraining → Validation → Online
2. ✅ **`joint_update_valid` controls**: Alternating (default) vs. unified update strategy
3. ✅ **Validation freezing**: Both backbone and adapters are selectively frozen using alternating strategy
4. ✅ **Documentation corrected**: All references to "freeze_bias = backbone bias" have been fixed

