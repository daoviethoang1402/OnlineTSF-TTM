#!/usr/bin/env bash
# =============================================================================
# Ablation 3 (ablation-study-2) -- CONCEPT ENCODER, on reduced-noft (frozen
# PROCEED), Exchange + Jiaolong.
#
# ANCHOR (held fixed everywhere else in ablation-study-2):
#   SSF      : Hypernetwork + affine beta-gamma (out-scale + shift, no input-scale)
#   Capacity : concept_dim=32, bottleneck_dim=32
#   Encoders : dual
#
# This script is where the ENCODER is the studied component -> SSF and capacity
# are held at the anchor (--ssf_no_input_scale, cd=32/bn=32), --concept_mode
# varies across 4 values (--concept_mode's CLI choice for "share" is "shared"):
#   dual    : two encoders, drift = c(X_t) - c(X_t-H)                [= anchor]
#   shared  : one shared encoder for both (recent truncated to seq_len)
#   current : only c(X_t), NO drift subtraction
#   strip_y : like dual, but the historical encoder sees ONLY X_t-H, not
#             (X_t-H, Y_t-H) -- isolates whether the label Y_t-H itself
#             contributes to the historical concept (see adapter/proceed.py)
#
# Design decisions (stated, not guessed):
#   * Regime = the deployed config: --freeze_online, ema=0, decoder=common_channel.
#   * online_learning_rate is UNIFORM at 3e-6 for both datasets (see
#     ssf-version.sh header for the rationale).
#   * Exchange: seq_len in {512,1024,1536} x pred_len in {96,192,336}. Jiaolong:
#     same seq_lens, pred_len also includes 720.
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/ablation-study-2" ]; then mkdir ./logs/ablation-study-2; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ_LENS=(512 1024 1536)
LR=0.000003
ITR=1
GPU=${GPU:-0}   # override per-invocation: GPU=1 bash encoders.sh
MODES=(dual shared current strip_y)

# dataset spec: name | freq | pred_lens
DATASETS=(
  "Exchange|d|96 192 336"
  # "Jiaolong_DSMS|s|96 192 336 720"
)

for dspec in "${DATASETS[@]}"; do
  IFS='|' read -r data freq preds <<< "$dspec"
  for mode in "${MODES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
      for pred_len in $preds; do
        fn=logs/ablation-study-2/encoders_${data}_${seq_len}_${pred_len}_${mode}.log
        if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then echo "SKIP $fn"; continue; fi
        echo "[encoders] $data seq=$seq_len pred=$pred_len mode=$mode"
        python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
          --dataset "$data" --seq_len "$seq_len" --pred_len "$pred_len" --freq "$freq" \
          --train_ratio 0.5 --test_ratio 0.45 --gpu $GPU \
          --online_method Proceed --concept_mode "$mode" --ssf_no_input_scale \
          --concept_dim 32 --bottleneck_dim 32 --ema 0 \
          --batch_size 64 --online_learning_rate $LR --itr $ITR >> "$fn" 2>&1
      done
    done
  done
done
echo "[encoders] DONE -> logs/ablation-study-2/encoders_*"
