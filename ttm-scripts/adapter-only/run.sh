#!/usr/bin/env bash
# =============================================================================
# Task 2 -- "adapter-only" (approach 2): freeze TTM head from the START, train
# ONLY the PROCEED adapter. Compared against zero-shot (zeroshot.sh) and the
# existing few-shot numbers.
#
# Mechanism: --freeze  => TTM decoder+head frozen during pretraining (only the
# adapter/generator learns). Under --freeze there is NO biases[-1], so
# online_learning_rate is INERT -> a single amortized setting (the lr below is
# passed but does nothing; kept for uniformity).
#
# Runs BOTH capacities (reduced cd64/bn8, full cd200/bn32), all 11 datasets,
# full context x horizon grid. --itr 1 (single seed) to match the main results.
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/adapter-only" ]; then mkdir ./logs/adapter-only; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ_LENS=(512 1024 1536)
LR=0.000001            # inert under --freeze; kept for uniform logging
# capacity spec:  tag | cdim | bneck
CAPS=(
  "reduced|64|8"
  "full|200|32"
)
# dataset spec:  name | freq | pred_lens
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
  for capspec in "${CAPS[@]}"; do
    IFS='|' read -r captag cdim bneck <<< "$capspec"
    for seq_len in "${SEQ_LENS[@]}"; do
      for pred_len in $preds; do
        fn=logs/adapter-only/${data}_${seq_len}_${pred_len}_${captag}.log
        if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then
          echo "[adapter-only] SKIP (done) $data $seq_len $pred_len $captag"; continue
        fi
        echo "[adapter-only] data=$data seq=$seq_len pred=$pred_len cap=$captag (--freeze, adapter-only)"
        python run.py --model TinyTimeMixer --freeze --decoder_mode common_channel \
          --dataset "$data" --seq_len "$seq_len" --pred_len "$pred_len" --freq "$freq" \
          --train_ratio 0.5 --test_ratio 0.45 \
          --online_method Proceed --concept_dim "$cdim" --bottleneck_dim "$bneck" --ema 0 \
          --batch_size 64 --online_learning_rate $LR --itr 1 >> "$fn" 2>&1
      done
    done
  done
done
echo "[adapter-only] DONE -> logs/adapter-only/"
