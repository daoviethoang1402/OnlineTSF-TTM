#!/usr/bin/env bash
# =============================================================================
# Task 2 -- ZERO-SHOT baseline: the pretrained TTM backbone tested with NO
# fine-tuning on the target data (no head fine-tune, no adapter).
#
# Uses the new --zero_shot flag: no --online_method, training is skipped, and
# Exp_Main.test() evaluates the in-memory pretrained (IBM ft-r2) checkpoint. This
# is the SAME test path as few-shot, so zero-shot / few-shot / adapter-only /
# PROCEED-online are all directly comparable.
#
# 11 datasets, full context x horizon grid, --itr 1.
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/zeroshot" ]; then mkdir ./logs/zeroshot; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ_LENS=(512 1024 1536)
DATASETS=(
  "ETTh1|h|96 192 336 720"
  "ETTh2|h|96 192 336 720"
  "ETTm1|15min|96 192 336 720"
  "ETTm2|15min|96 192 336 720"
  "Weather|10min|96 192 336 720"
  "Jiaolong_DSMS|s|96 192 336 720"
  "wind|h|96 192 336 720"
  "Energy|10min|96 192 336 720"
  "BeijingAQ|h|96 192 336 720"
  "Exchange|d|96 192 336"
  "AirQuality|h|96 192 336"
)

for spec in "${DATASETS[@]}"; do
  IFS='|' read -r data freq preds <<< "$spec"
  for seq_len in "${SEQ_LENS[@]}"; do
    for pred_len in $preds; do
      fn=logs/zeroshot/${data}_${seq_len}_${pred_len}_zeroshot.log
      if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then
        echo "[zeroshot] SKIP (done) $data $seq_len $pred_len"; continue
      fi
      echo "[zeroshot] data=$data seq=$seq_len pred=$pred_len (no fine-tuning)"
      python run.py --model TinyTimeMixer --zero_shot --decoder_mode common_channel \
        --dataset "$data" --seq_len "$seq_len" --pred_len "$pred_len" --freq "$freq" \
        --train_ratio 0.5 --test_ratio 0.45 \
        --batch_size 64 --itr 1 >> "$fn" 2>&1
    done
  done
done
echo "[zeroshot] DONE -> logs/zeroshot/"
