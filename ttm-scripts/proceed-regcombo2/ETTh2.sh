if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/proceed_regcombo2" ]; then
    mkdir ./logs/proceed_regcombo2
fi

# ---------------------------------------------------------------------------
# reg_combo2 confirmation run (frozen PROCEED + TTM), ETTm2.
#
# Best recipe found on ETTh1/ETTh2 (see documents/PROCEED_TTM_HYPERPARAM_NOTES.md):
# reg_combo2 = cdim64 + bneck8 + ema0.9, with online_lr corrected from the
# original sweep's 3e-5 down to the properly-tuned 3e-6. Beat zero-shot TTM on
# 6/8 corner cells on ETTh1/ETTh2. This script applies that same fixed recipe
# across the FULL grid (all seq_len x pred_len, matching
# ttm-scripts/proceed-frz-onl/ shape) to confirm it generalizes to a dataset
# not used for tuning.
#
# --freeze_online keeps the TTM backbone+head frozen; only the adapter adapts.
# --freq h matches the convention in proceed-frz-onl/ (TTM uses timeenc=2,
# which does not consume the freq string, so this is a no-op regardless of
# dataset sampling rate).
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
freq=h
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh2

# reg_combo2 recipe:
concept_dim=64
bottleneck_dim=8
ema=0.9
online_learning_rate=0.000003

for seq_len in 512 1024 1536
do
for pred_len in 96 192 336 720
do
    filename=logs/proceed_regcombo2/TTM'_'Proceed'_'$data'_'$seq_len'_'$pred_len'_'regcombo2'_'freezeonl.log

    echo "[regcombo2] data=$data seq=$seq_len pred=$pred_len cdim=$concept_dim bneck=$bottleneck_dim ema=$ema lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Proceed --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --ema $ema --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
