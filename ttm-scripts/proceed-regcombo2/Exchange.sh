if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/proceed_regcombo2" ]; then
    mkdir ./logs/proceed_regcombo2
fi

# ---------------------------------------------------------------------------
# reg_combo2 full-grid run (frozen PROCEED + TTM), Exchange (headroom study,
# non-stationary air-quality; headroom-study candidate).
#
# reg_combo2 = cdim64 + bneck8 + ema0.9 + online_lr 3e-6, frozen backbone.
# Compare each cell to few-shot TTM (run the probe / Naive baseline separately)
# to see if the Exchange-style headroom (up to -26%) replicates here.
# 35k rows -> full grid (all pred incl 720) fits.
# --freq h: TTM uses timeenc=2, so freq is a no-op.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
freq=d
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=Exchange

# reg_combo2 recipe:
concept_dim=64
bottleneck_dim=8
ema=0.9
online_learning_rate=0.000003

for seq_len in 512 1024 1536
do
for pred_len in 96 192 336
do
    filename=logs/proceed_regcombo2/TTM'_'Proceed'_'$data'_'$seq_len'_'$pred_len'_'regcombo2'_'freezeonl.log

    echo "[regcombo2] data=$data seq=$seq_len pred=$pred_len cdim=$concept_dim bneck=$bottleneck_dim ema=$ema lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode mix_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Proceed --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --ema $ema --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
