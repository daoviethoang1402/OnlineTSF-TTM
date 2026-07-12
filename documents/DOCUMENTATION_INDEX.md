# PROCEED + PatchTST Documentation Index

## Quick Navigation

### 📚 Three Documentation Levels

#### 1. **QUICK_REFERENCE.md** (Start here - 5 min read)
- High-level overview of what PROCEED+PatchTST does
- Three execution phases explained visually
- Key hyperparameters and freezing strategy
- Quick command to run the pipeline
- **Best for**: Getting oriented quickly

#### 2. **PIPELINE_EXPLANATION.md** (Deep dive - 20 min read)
- Complete architecture component breakdown
- Detailed phase-by-phase execution flow
- Data flow and information leakage prevention
- Concept drift detection mechanism
- Adapter application details
- Configuration parameters explained
- **Best for**: Understanding the system thoroughly

#### 3. **ARCHITECTURE_DIAGRAM.txt** (Reference - visual diagrams)
- ASCII diagrams of overall pipeline
- PROCEED wrapper architecture
- PatchTST backbone structure
- Online learning loop flow
- Down-Up adapter application
- Freezing strategy timeline
- File dependencies
- **Best for**: Visual learners and reference

---

### 🧩 TTM-Specific Documentation (branch `no-finetune`)

Everything above documents the generic PROCEED+PatchTST pipeline. TinyTimeMixer (TTM) was
integrated later and adds its own freezing policy on top of the generic strategy described in
**PHASES_AND_FREEZING_CLARIFICATION.md** — read that first if you're unfamiliar with the
generic backbone/adapter freezing alternation, then read these for what TTM changes/adds:

- **[TinyTimeMixer_INTEGRATION.md](TinyTimeMixer_INTEGRATION.md)** — original integration notes (partially outdated; see `../CHANGES.md` for the current state).
- **[TTM_FREEZE_FLAGS_CLARIFICATION.md](TTM_FREEZE_FLAGS_CLARIFICATION.md)** — what `--freeze` vs `--freeze_online` each do for TTM, why they're independent axes, and why `--normalization RevIN` should not be combined with TTM (it already normalizes internally).
- **[../CHANGES.md](../CHANGES.md)** — full change history for the TTM+PROCEED integration, including the `--freeze_online` flag and the `ForecastModel` hook-forwarding bug fix.
- **Best for**: Anyone running `--model TinyTimeMixer`, or deciding between `--freeze` and `--freeze_online`.

---

## Document Quick Links

```
QUICK_REFERENCE.md
├── What is this pipeline?
├── Three execution phases
├── Core technical innovation
├── Architecture layers
├── Data flow in online phase
├── Key hyperparameters
├── Freezing strategy
├── Data processing pipeline
├── File structure
└── Next steps

PIPELINE_EXPLANATION.md
├── Overview
├── Architecture components
│  ├── PatchTST backbone
│  ├── PROCEED adapter
│  └── Experimental framework
├── Full pipeline execution
│  ├── Phase 1: Pretraining
│  ├── Phase 2: Validation data update
│  ├── Phase 3: Online testing
│  └── Phase 4: Information leakage handling
├── Data flow details
├── Key configuration parameters
├── Adapter freezing strategy
├── Concept drift detection
├── No information leakage guarantee
└── Execution flow diagram

ARCHITECTURE_DIAGRAM.txt
├── Overall pipeline
├── PROCEED wrapper architecture
├── PatchTST backbone architecture
├── Online learning loop
├── Down-Up adapter application
├── Freezing strategy timeline
├── Key equations
└── File dependencies

TTM_FREEZE_FLAGS_CLARIFICATION.md
├── What --freeze does (3 effects, generic to any backbone)
├── What --freeze_online does (TTM-only, val/online cutoff)
├── Side-by-side comparison table
├── --normalization RevIN: don't combine it with TTM (redundant internal scaler)
└── FAQ
```

---

## Core Concepts Summary

### What Makes PROCEED Unique?

**Problem**: Concept drift in time series (data distribution changes over time)

**Solution**: 
1. Detect drift via concept difference (MLP1 vs MLP2)
2. Generate per-layer adaptations from drift signal
3. Apply lightweight Down-Up adapters
4. Prevent catastrophic forgetting via freezing strategy

### The Three-Phase Pipeline

| Phase | Data | Type | Purpose |
|-------|------|------|---------|
| **Pretraining** | Historical (full) | Batch processing | Learn features |
| **Validation Adapt** | Validation (sequential) | Online loop | Prepare for test |
| **Online Test** | Test (sequential) | Online loop | Evaluate performance |

### Key Innovation: No Information Leakage

Each test sample is processed as a pair:
- **Recent batch**: Historical context (for learning)
- **Current batch**: Current prediction window (for evaluation)

Guarantees labels are only available after forecast horizon.

---

## How to Read the Documentation

### If you want to...

**Understand the big picture (5 min)**
→ Read: QUICK_REFERENCE.md → Section "What is this pipeline?"

**Understand concept drift detection (10 min)**
→ Read: QUICK_REFERENCE.md → Section "Core Technical Innovation"
→ OR: ARCHITECTURE_DIAGRAM.txt → Equation "DRIFT COMPUTATION"

**Understand the online loop (10 min)**
→ Read: PIPELINE_EXPLANATION.md → Section "Phase 3: Online Testing"
→ OR: ARCHITECTURE_DIAGRAM.txt → Section "ONLINE LEARNING LOOP"

**Understand why no information leakage (5 min)**
→ Read: PIPELINE_EXPLANATION.md → Section "No Information Leakage Guarantee"
→ OR: QUICK_REFERENCE.md → "Data Flow in Online Phase"

**Understand the freezing strategy (10 min)**
→ Read: PIPELINE_EXPLANATION.md → Section "Adapter Freezing Strategy"
→ OR: QUICK_REFERENCE.md → Section "Freezing Strategy"
→ OR: ARCHITECTURE_DIAGRAM.txt → Section "FREEZING STRATEGY TIMELINE"

**Run an experiment (2 min)**
→ Read: QUICK_REFERENCE.md → Section "Running the Pipeline"

**Decide between `--freeze` and `--freeze_online` for TTM (5 min)**
→ Read: TTM_FREEZE_FLAGS_CLARIFICATION.md → Full document (it's short)
→ OR: ../CHANGES.md → §6 for the code-level diff

**Deep dive into implementation (30 min)**
→ Read: PIPELINE_EXPLANATION.md → Full document
→ Then: ARCHITECTURE_DIAGRAM.txt → For visual reference

---

## Key Files in the Codebase

### Main Entry Points
- `run.py` - Command-line interface and pipeline orchestration
- `settings.py` - Configuration and hyperparameters

### PROCEED Components
- `adapter/proceed.py` - Main PROCEED wrapper
- `adapter/module/down_up.py` - Down-Up adapter implementation
- `adapter/module/generator.py` - Concept drift → adaptation mapping
- `adapter/module/base.py` - Base adapter class

### Experimental Framework
- `exp/exp_main.py` - Pretraining (Exp_Main)
- `exp/exp_online.py` - Online learning (Exp_Online)
- `exp/exp_proceed.py` - PROCEED-specific logic (Exp_Proceed)
- `exp/exp_basic.py` - Base experiment utilities

### Models & Data
- `models/PatchTST.py` - PatchTST backbone
- `layers/PatchTST_backbone.py` - PatchTST implementation details
- `data_provider/data_loader.py` - Dataset_Recent wrapper
- `data_provider/data_factory.py` - Data loading utilities

---

## Configuration Parameters by Category

### Model Architecture (PatchTST)
```
--seq_len 96            # Input window
--pred_len 96           # Forecast horizon
--patch_len 16          # Patch size
--stride 8              # Patch stride
--d_model 128           # Hidden dimension
--n_heads 16            # Attention heads
--e_layers 3            # Encoder layers
--dropout 0.2           # Dropout rate
```

### PROCEED Adaptation
```
--concept_dim 200       # Drift detection dimension
--bottleneck_dim 32     # Bottleneck compression
--tune_mode 'down_up'   # Adapter type
--act 'sigmoid'         # Activation
--ema 0                 # EMA smoothing (0=off)
--freeze True           # Freeze backbone
--wo_clip False         # Clip adaptations
```

### Training Setup
```
--learning_rate 0.0001          # Pretraining LR
--online_learning_rate 0.001    # Online LR
--train_epochs 100              # Pretraining epochs
--batch_size 32                 # Batch size
--patience 5                    # Early stopping
```

### Data & Evaluation
```
--dataset ETTh1         # Dataset name
--features 'M'          # Features mode
--border_type 'online'  # Data split strategy
--pretrain True         # Load pretrained
--do_valid True         # Validate on validation set
```

---

## Common Workflows

### Workflow 1: Full Pipeline from Scratch
```bash
python run.py \
  --model PatchTST \
  --dataset ETTh1 \
  --online_method Proceed \
  --pretrain \
  --do_valid \
  --learning_rate 0.0001 \
  --online_learning_rate 0.001 \
  --concept_dim 200 \
  --bottleneck_dim 32
```

### Workflow 2: Test with Existing Checkpoint
```bash
python run.py \
  --model PatchTST \
  --dataset ETTh1 \
  --online_method Proceed \
  --only_test
```

### Workflow 3: Hyperparameter Tuning
```bash
# Vary concept_dim
for dim in 100 200 300; do
  python run.py --concept_dim $dim ...
done

# Vary bottleneck_dim
for btl in 16 32 64; do
  python run.py --bottleneck_dim $btl ...
done
```

---

## Debugging & Understanding Code

### To understand data flow:
1. Start in `run.py` main loop
2. Follow `Exp(args)` initialization
3. Trace `exp.train()` if pretraining
4. Trace `exp.update_valid()` for validation
5. Trace `exp.online()` for online testing

### To understand concept drift:
1. Look at `Proceed.generate_adaptation()` in `adapter/proceed.py`
2. See how MLP1 and MLP2 extract concepts
3. Compute drift as difference
4. Map to layer parameters via generator

### To understand adaptation application:
1. Check `Down_Up.assign_adaptation()` in `adapter/module/down_up.py`
2. See how scale_in, scale_out, shift are computed
3. Understand how they modify weights in-place

### To understand no leakage:
1. Check `Dataset_Recent` in `data_provider/data_loader.py`
2. Understand (recent_batch, current_batch) pairing
3. See online loop in `exp/exp_online.py` lines 170-200
4. Verify recent_batch doesn't use future labels

---

## Performance Characteristics

- **Pretraining**: Standard training speed
- **Online Phase**: ~5-10% slower than baseline (due to sequential processing)
- **Adaptation Overhead**: <1% (MLP forward pass only)
- **Memory**: Low (no replay buffer by default)

---

## FAQ

**Q: Why freeze the backbone in online phase?**
A: Preserves learned features, prevents catastrophic forgetting. Only adapters change.

**Q: Why use Down-Up adapters?**
A: Lightweight, parameter-efficient, learnable per-layer. Minimal overhead.

**Q: Why no information leakage?**
A: Recent batch from past, current batch labels from available future. Perfect sequential simulation.

**Q: Can I use a different backbone?**
A: Yes! Just replace PatchTST in models and ensure it works with Down-Up adapters (Linear/Conv1d layers).

**Q: What's the difference between validation and online testing?**
A: Validation: adapting to validation data. Online: same process on test data. Both use same freeze strategy.

**Q: How do I increase adaptation sensitivity?**
A: Increase `--concept_dim` (more detailed drift detection) or decrease `--bottleneck_dim` (more direct mapping).

**Q: For TTM, what's the difference between `--freeze` and `--freeze_online`?**
A: `--freeze` is generic (any backbone) and lifelong — if set, decoder/head never fine-tune, not even during pretraining. `--freeze_online` is TTM-only — decoder/head always fine-tune during pretraining, but this flag locks them frozen before val/online instead of letting that fine-tuning continue. See TTM_FREEZE_FLAGS_CLARIFICATION.md for the full breakdown.

---

## Next Steps

1. **Read QUICK_REFERENCE.md** for overview
2. **Read PIPELINE_EXPLANATION.md** for details
3. **Review ARCHITECTURE_DIAGRAM.txt** for visual understanding
4. **Trace code** following the file dependencies
5. **Run experiments** with different hyperparameters
6. **Implement extensions** based on understanding

---

## Document Versions

- **Created**: 2026-05-16
- **Coverage**: PROCEED paper (KDD 2025)
- **Codebase**: OnlineTSF framework
- **Backbone**: PatchTST

---

## Support

For questions about specific components, refer to:
- Component → Find in architecture diagram
- Architecture diagram → Find detailed explanation in PIPELINE_EXPLANATION.md
- Code implementation → See file list in ARCHITECTURE_DIAGRAM.txt

