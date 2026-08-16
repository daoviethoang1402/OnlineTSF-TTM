#!/usr/bin/env bash
# =============================================================================
# Ablation 1 (ablation-study-2) -- SSF VERSION, on reduced-noft (frozen PROCEED),
# Exchange + Jiaolong.
#
# ANCHOR (held fixed everywhere else in ablation-study-2):
#   SSF      : Hypernetwork + affine beta-gamma (out-scale + shift, no input-scale)
#   Capacity : concept_dim=32, bottleneck_dim=32
#   Encoders : dual
#
# This script is where SSF is the studied component -> capacity and encoder are
# held at the anchor, SSF varies across:
#   beta_gamma       : Hypernetwork + affine beta-gamma       (out-scale + shift)
#                       = the anchor itself                    [--ssf_no_input_scale]
#   alpha_beta_gamma : Hypernetwork + affine alpha-beta-gamma (in-scale + out-scale
#                       + shift, i.e. full PROCEED SSF, no ablation flags)
#   canonical        : Canonical SSF (raw learnable out-scale + shift, NO
#                       hypernetwork, NO drift, NO concept encoders --
#                       adapter/canonical_ssf.py, --online_method CanonicalSSF)
#
# Design decisions (stated, not guessed):
#   * Regime = the deployed config: --freeze_online, ema=0, decoder=common_channel.
#   * online_learning_rate is UNIFORM at 3e-6 for both datasets (matches the
#     Final-Run MIX-vs-ALL analysis: per-dataset lr tuning bought ~nothing and
#     uniform 3e-6 was never worse -- see documents/THESIS_REPORT_NOTES.md).
#   * canonical has no concept encoder / generator / drift signal at all, so
#     --concept_mode / --concept_dim / --bottleneck_dim / --ema / --ssf_no_*
#     are meaningless for it and are simply omitted.
#   * Exchange: seq_len in {512,1024,1536} x pred_len in {96,192,336} (no 720:
#     val split too small at long horizon). Jiaolong: same seq_lens, pred_len
#     also includes 720.
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/ablation-study-2" ]; then mkdir ./logs/ablation-study-2; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ_LENS=(512 1024 1536)
LR=0.000003
ITR=1
GPU=${GPU:-0}   # override per-invocation: GPU=1 bash ssf-version.sh

# dataset spec: name | freq | pred_lens
DATASETS=(
  "Exchange|d|96 192 336"
  # "Jiaolong_DSMS|s|96 192 336 720"
)

# variant spec: name | extra flags (empty = pure default, no ablation flags)
VARIANTS=(
  "beta_gamma|--online_method Proceed --concept_mode dual --concept_dim 32 --bottleneck_dim 32 --ema 0 --ssf_no_input_scale"
  "alpha_beta_gamma|--online_method Proceed --concept_mode dual --concept_dim 32 --bottleneck_dim 32 --ema 0"
  "canonical|--online_method CanonicalSSF --tune_mode down_up"
)

for dspec in "${DATASETS[@]}"; do
  IFS='|' read -r data freq preds <<< "$dspec"
  for vspec in "${VARIANTS[@]}"; do
    IFS='|' read -r vname vflags <<< "$vspec"
    for seq_len in "${SEQ_LENS[@]}"; do
      for pred_len in $preds; do
        fn=logs/ablation-study-2/ssfversion_${data}_${seq_len}_${pred_len}_${vname}.log
        if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then echo "SKIP $fn"; continue; fi
        echo "[ssf-version] $data seq=$seq_len pred=$pred_len variant=$vname"
        python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
          --dataset "$data" --seq_len "$seq_len" --pred_len "$pred_len" --freq "$freq" \
          --train_ratio 0.5 --test_ratio 0.45 --gpu $GPU \
          $vflags \
          --batch_size 64 --online_learning_rate $LR --itr $ITR >> "$fn" 2>&1
      done
    done
  done
done
echo "[ssf-version] DONE -> logs/ablation-study-2/ssfversion_*"
