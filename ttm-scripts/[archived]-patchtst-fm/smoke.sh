#!/usr/bin/env bash
# =============================================================================
# PatchTST-FM integration smoke test (Task 3)  --  RUN ON THE 16 GB GPU.
# -----------------------------------------------------------------------------
# Validates the FULL pipeline (pretrain -> val-update -> online) for the new
# PatchTST-FM backbone on ONE cheap cell before scaling to the grid.
#
# Was validated on CPU up to a forward pass (shape (B,pred_len,C) correct,
# 86 PROCEED adapters injected, gradient reaches the adapter). The full
# pretrain+backward could NOT be tested in the 2 GB sandbox -- hence this.
#
# COST WARNING: PatchTST-FM always runs at 8192 context (512 patches x 20 layers
# x d_model 1024), ~10x heavier than TTM per step. Keep batch small; expect the
# online loop to be slow. Bump batch_size once you see memory headroom.
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/patchtst-fm" ]; then mkdir ./logs/patchtst-fm; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data=Exchange; freq=d; seq_len=512; pred_len=96
cdim=64; bneck=8; lr=0.000003          # reduced-noft @ Exchange's deployed lr
fn=logs/patchtst-fm/smoke_${data}_${seq_len}_${pred_len}.log

echo "[ptfm-smoke] $data seq=$seq_len pred=$pred_len cap=reduced lr=$lr (batch small; heavy model)"
python run.py --model PatchTST_FM --freeze_online --decoder_mode common_channel \
  --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \
  --train_ratio 0.5 --test_ratio 0.45 \
  --online_method Proceed --concept_dim $cdim --bottleneck_dim $bneck --ema 0 \
  --batch_size 8 --online_learning_rate $lr --itr 1 2>&1 | tee "$fn"

echo "[ptfm-smoke] done -> $fn  (look for a final 'mse:..., mae:...' line)"
