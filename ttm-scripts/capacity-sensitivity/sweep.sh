#!/usr/bin/env bash
# =============================================================================
# Capacity-sensitivity sweep (Task 1)
# -----------------------------------------------------------------------------
# Question: how small can the drift-conditioned adapter be and still hold the
#           win?  Sweep concept_dim x bottleneck_dim on the datasets that HAVE
#           headroom (where capacity is even observable): Exchange, AirQuality,
#           Jiaolong.  3 seeds each -> mean +/- std.
#
# Design decisions (stated, not guessed):
#   * Regime = the DEPLOYED config: --freeze_online, ema=0, decoder=common_channel,
#     single Proceed adapter.  Capacity is the ONLY variable.
#   * online_learning_rate is HELD at each dataset's deployed value
#     (Exchange 3e-6; AirQuality/Jiaolong 1e-6) so the sweep isolates capacity.
#   * Full context x horizon grid per dataset (interiors are informative for a
#     sensitivity curve).  Exchange/AirQuality cap at pred 336 (no 720).
#   * 3 seeds via --itr 3  (run.py uses fix_seed = 2021 + ii -> 2021/2022/2023).
#
# Grid:  cdim in {16,32,64,128} x bneck in {8,16,32} = 12 capacities.
# Runs:  Exchange 3ctx*3pred*12 = 108 ; AirQuality 108 ; Jiaolong 3*4*12 = 144
#        => 360 invocations, each running 3 seeds internally.
#
# To trim: shrink SEQ_LENS / CDIMS / BNECKS below, or comment out a dataset row.
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/capacity-sensitivity" ]; then mkdir ./logs/capacity-sensitivity; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- knobs -----------------------------------------------------------------
SEQ_LENS=(512 1024 1536)
CDIMS=(16 32 64 128)
BNECKS=(8 16 32)
ITR=3
TRAIN_RATIO=0.5
TEST_RATIO=0.45
BATCH=64

# dataset spec:  name | freq | pred_lens | online_lr
DATASETS=(
  "Exchange|d|96 192 336|0.000003"
  "AirQuality|h|96 192 336|0.000001"
  "Jiaolong_DSMS|s|96 192 336 720|0.000001"
)
# ----------------------------------------------------------------------------

for spec in "${DATASETS[@]}"; do
  IFS='|' read -r data freq preds lr <<< "$spec"
  for seq_len in "${SEQ_LENS[@]}"; do
    for pred_len in $preds; do
      for cdim in "${CDIMS[@]}"; do
        for bneck in "${BNECKS[@]}"; do
          fn=logs/capacity-sensitivity/${data}_${seq_len}_${pred_len}_cd${cdim}_bn${bneck}_lr${lr}.log
          # resume: skip only if the log already holds ITR completed test lines ("mse:...")
          if [ -s "$fn" ] && [ "$(grep -c 'mse:' "$fn" 2>/dev/null)" -ge "$ITR" ]; then
            echo "[capsens] SKIP (done) $data seq=$seq_len pred=$pred_len cd=$cdim bn=$bneck"
            continue
          fi
          echo "[capsens] data=$data seq=$seq_len pred=$pred_len cd=$cdim bn=$bneck lr=$lr itr=$ITR"
          python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
            --dataset "$data" --seq_len "$seq_len" --pred_len "$pred_len" --freq "$freq" \
            --train_ratio $TRAIN_RATIO --test_ratio $TEST_RATIO \
            --online_method Proceed --concept_dim "$cdim" --bottleneck_dim "$bneck" --ema 0 \
            --batch_size $BATCH --online_learning_rate "$lr" --itr $ITR >> "$fn" 2>&1
        done
      done
    done
  done
done
echo "[capsens] DONE. Parse with: python ttm-scripts/capacity-sensitivity/parse_sweep.py"
