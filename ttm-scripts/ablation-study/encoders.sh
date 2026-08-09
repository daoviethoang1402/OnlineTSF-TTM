#!/usr/bin/env bash
# =============================================================================
# Ablation 1 -- concept ENCODER, on reduced-noft (frozen PROCEED), Exchange only.
#   dual    : two encoders, drift = c(X_t) - c(X_{t-H})   [= current method]
#   shared  : one shared encoder for both
#   current : only c(X_t), NO drift subtraction
#   stats   : data-space per-channel moment featurizer (mean/std/last/trend)
# Purpose: does a different encoder / drift formulation work better?
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/ablation-study" ]; then mkdir ./logs/ablation-study; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# reduced-noft @ Exchange
data=Exchange; freq=d; PREDS="96 192 336"; LR=0.000003; CDIM=64; BNECK=8; EMA=0
SEQ_LENS=(512 1024 1536)
MODES=(dual shared current stats)
ITR=1

for mode in "${MODES[@]}"; do
  for seq_len in "${SEQ_LENS[@]}"; do
    for pred_len in $PREDS; do
      fn=logs/ablation-study/encoders_${data}_${seq_len}_${pred_len}_${mode}.log
      if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then echo "SKIP $fn"; continue; fi
      echo "[enc] $data seq=$seq_len pred=$pred_len mode=$mode (reduced-noft)"
      python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
        --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \
        --train_ratio 0.5 --test_ratio 0.45 \
        --online_method Proceed --concept_mode "$mode" \
        --concept_dim $CDIM --bottleneck_dim $BNECK --ema $EMA \
        --batch_size 64 --online_learning_rate $LR --itr $ITR >> "$fn" 2>&1
    done
  done
done
echo "[enc] DONE -> logs/ablation-study/encoders_*"
