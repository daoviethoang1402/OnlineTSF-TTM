#!/usr/bin/env bash
# =============================================================================
# Concept-encoder ABLATION on the ONLINE fine-tuning (SAFE) approach.
# Holds capacity + lr fixed; varies only --concept_mode:
#   dual    : two encoders, drift = c(X_t) - c(X_{t-H})   [= current method]
#   shared  : one shared encoder for both (recent truncated to seq_len)
#   current : only c(X_t) fed to the generator, NO drift subtraction
#   stats   : data-space per-channel moment featurizer (mean/std/last/trend)
#
# Online-FT setting = --online_method Proceed WITHOUT --freeze_online (decoder+head
# keep fine-tuning online), i.e. the thesis "TTM-P".  MATCH CDIM/BNECK/OLR/EMA below
# to your thesis TTM-P run so the ablation is apples-to-apples.
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/ablation-concept" ]; then mkdir ./logs/ablation-concept; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- online-FT config (match thesis TTM-P) ----
OLR=0.0001; CDIM=200; BNECK=32; EMA=0
SEQ_LENS=(512 1024 1536)
MODES=(dual shared current stats)
# Representative subset. Add ETTh1/ETTm1/ETTm2 to cover the full thesis set.
DATASETS=(
  "ETTh2|h|96 192 336 720"
  "Weather|10min|96 192 336 720"
  "Jiaolong_DSMS|s|96 192 336 720"
)

for spec in "${DATASETS[@]}"; do
  IFS='|' read -r data freq preds <<< "$spec"
  for mode in "${MODES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
      for pred_len in $preds; do
        fn=logs/ablation-concept/${data}_${seq_len}_${pred_len}_${mode}.log
        if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then
          echo "[concept] SKIP (done) $data $seq_len $pred_len $mode"; continue
        fi
        echo "[concept] data=$data seq=$seq_len pred=$pred_len mode=$mode (online-FT)"
        python run.py --model TinyTimeMixer --decoder_mode common_channel \
          --dataset "$data" --seq_len "$seq_len" --pred_len "$pred_len" --freq "$freq" \
          --train_ratio 0.5 --test_ratio 0.45 \
          --online_method Proceed --concept_mode "$mode" \
          --concept_dim $CDIM --bottleneck_dim $BNECK --ema $EMA \
          --batch_size 64 --online_learning_rate $OLR --itr 1 >> "$fn" 2>&1
      done
    done
  done
done
echo "[concept] DONE -> logs/ablation-concept/  (compare mse/mae across the 4 modes)"
