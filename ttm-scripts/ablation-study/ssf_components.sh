#!/usr/bin/env bash
# =============================================================================
# Ablation 3 -- PROCEED's SSF COMPONENT decomposition.
# reduced-noft (frozen PROCEED), Exchange only.
# PROCEED applies:  out = scale * ( W * (scale2 * x) ) + shift + b
#   full        : scale + scale2 + shift            [= current method]
#   scale_shift : drop input scale2 (standard SSF on the output)
#   scale_only  : output scale only
#   shift_only  : additive shift only
#   inscale_only: input scale2 only
# Purpose: which SSF component carries the win?
# =============================================================================
set -u
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/ablation-study" ]; then mkdir ./logs/ablation-study; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data=Exchange; freq=d; PREDS="96 192 336"; LR=0.000003; CDIM=64; BNECK=8; EMA=0
SEQ_LENS=(512 1024 1536)
# name | extra flags (disabled components forced to identity)
VARIANTS=(
  "full|"
  "scale_shift|--ssf_no_input_scale"
  "scale_only|--ssf_no_input_scale --ssf_no_shift"
  "shift_only|--ssf_no_input_scale --ssf_no_scale"
  "inscale_only|--ssf_no_scale --ssf_no_shift"
)
ITR=1

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r vname vflag <<< "$spec"
  for seq_len in "${SEQ_LENS[@]}"; do
    for pred_len in $PREDS; do
      fn=logs/ablation-study/ssfcomp_${data}_${seq_len}_${pred_len}_${vname}.log
      if [ -s "$fn" ] && grep -q 'mse:' "$fn" 2>/dev/null; then echo "SKIP $fn"; continue; fi
      echo "[ssf-comp] $data seq=$seq_len pred=$pred_len variant=$vname"
      python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
        --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \
        --train_ratio 0.5 --test_ratio 0.45 \
        --online_method Proceed --concept_dim $CDIM --bottleneck_dim $BNECK --ema $EMA \
        $vflag \
        --batch_size 64 --online_learning_rate $LR --itr $ITR >> "$fn" 2>&1
    done
  done
done
echo "[ssf-comp] DONE -> logs/ablation-study/ssfcomp_*"
