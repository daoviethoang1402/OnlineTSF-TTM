#!/usr/bin/env bash
# =============================================================================
# Ablation 2 (ablation-study-2) -- CAPACITY sensitivity, on reduced-noft
# (frozen PROCEED), Exchange + Jiaolong.
#
# ANCHOR (held fixed everywhere else in ablation-study-2):
#   SSF      : Hypernetwork + affine beta-gamma (out-scale + shift, no input-scale)
#   Capacity : concept_dim=32, bottleneck_dim=32
#   Encoders : dual
#
# This script is where CAPACITY is the studied component -> SSF and encoder are
# held at the anchor (--ssf_no_input_scale, --concept_mode dual), capacity
# varies across concept_dim x bottleneck_dim. The anchor cell (cd=32, bn=32) is
# included in the grid, so ssf-version.sh's "beta_gamma" row and encoders.sh's
# "dual" row are directly re-derivable from this sweep's (cd32,bn32) cell --
# same config, different script, should match modulo run-to-run noise.
#
# Design decisions (stated, not guessed):
#   * Regime = the deployed config: --freeze_online, ema=0, decoder=common_channel.
#   * online_learning_rate is UNIFORM at 3e-6 for both datasets (see
#     ssf-version.sh header for the rationale).
#   * Exchange: seq_len in {512,1024,1536} x pred_len in {96,192,336}. Jiaolong:
#     same seq_lens, pred_len also includes 720.
#
# Grid: cdim in {16,32,64,128} x bneck in {8,16,32} = 12 capacities.
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/ablation-study-2" ]; then mkdir ./logs/ablation-study-2; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ_LENS=(512 1024 1536)
CDIMS=(16 32 64 128)
BNECKS=(8 16 32)
LR=0.000003
ITR=1
GPU=${GPU:-0}   # override per-invocation: GPU=1 bash capacity-sensitivity.sh

# dataset spec: name | freq | pred_lens
DATASETS=(
  "Exchange|d|96 192 336"
  # "Jiaolong_DSMS|s|96 192 336 720"
)

for dspec in "${DATASETS[@]}"; do
  IFS='|' read -r data freq preds <<< "$dspec"
  for seq_len in "${SEQ_LENS[@]}"; do
    for pred_len in $preds; do
      for cdim in "${CDIMS[@]}"; do
        for bneck in "${BNECKS[@]}"; do
          fn=logs/ablation-study-2/capacity_${data}_${seq_len}_${pred_len}_cd${cdim}_bn${bneck}.log
          if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then
            echo "SKIP (done) $data seq=$seq_len pred=$pred_len cd=$cdim bn=$bneck"
            continue
          fi
          echo "[capacity] $data seq=$seq_len pred=$pred_len cd=$cdim bn=$bneck"
          python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
            --dataset "$data" --seq_len "$seq_len" --pred_len "$pred_len" --freq "$freq" \
            --train_ratio 0.5 --test_ratio 0.45 --gpu $GPU \
            --online_method Proceed --concept_mode dual --ssf_no_input_scale \
            --concept_dim "$cdim" --bottleneck_dim "$bneck" --ema 0 \
            --batch_size 64 --online_learning_rate $LR --itr $ITR >> "$fn" 2>&1
        done
      done
    done
  done
done
echo "[capacity] DONE -> logs/ablation-study-2/capacity_*"
