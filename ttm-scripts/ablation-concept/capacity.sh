#!/usr/bin/env bash
# =============================================================================
# Capacity ABLATION on the ONLINE fine-tuning (SAFE) approach.
# concept_mode = dual (current method); varies concept_dim x bottleneck_dim.
#   cdim  in {32, 64, 128}
#   bneck in {8, 16, 32}
# Same online-FT setting as concept_mode.sh (thesis TTM-P).
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/ablation-capacity" ]; then mkdir ./logs/ablation-capacity; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OLR=0.0001; EMA=0
SEQ_LENS=(512 1024 1536)
CDIMS=(32 64 128)
BNECKS=(8 16 32)
DATASETS=(
  "ETTh2|h|96 192 336 720"
  "Weather|10min|96 192 336 720"
  "Jiaolong_DSMS|s|96 192 336 720"
)

for spec in "${DATASETS[@]}"; do
  IFS='|' read -r data freq preds <<< "$spec"
  for cdim in "${CDIMS[@]}"; do
    for bneck in "${BNECKS[@]}"; do
      for seq_len in "${SEQ_LENS[@]}"; do
        for pred_len in $preds; do
          fn=logs/ablation-capacity/${data}_${seq_len}_${pred_len}_cd${cdim}_bn${bneck}.log
          if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then
            echo "[capacity] SKIP (done) $data $seq_len $pred_len cd$cdim bn$bneck"; continue
          fi
          echo "[capacity] data=$data seq=$seq_len pred=$pred_len cd=$cdim bn=$bneck (online-FT)"
          python run.py --model TinyTimeMixer --decoder_mode common_channel \
            --dataset "$data" --seq_len "$seq_len" --pred_len "$pred_len" --freq "$freq" \
            --train_ratio 0.5 --test_ratio 0.45 \
            --online_method Proceed --concept_mode dual \
            --concept_dim $cdim --bottleneck_dim $bneck --ema $EMA \
            --batch_size 64 --online_learning_rate $OLR --itr 1 >> "$fn" 2>&1
        done
      done
    done
  done
done
echo "[capacity] DONE -> logs/ablation-capacity/"
