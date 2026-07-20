if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/reduced-noft" ]; then
    mkdir ./logs/reduced-noft
fi

# ---------------------------------------------------------------------------
# reg_combo2 full-grid run (frozen PROCEED + TTM).
#
# reg_combo2 = cdim64 + bneck8 + ema0.9 + online_lr 3e-6, frozen backbone.
# --freeze_online keeps the TTM backbone+head frozen; only the adapter adapts.
# freq is a no-op for TTM (timeenc=2 ignores it).
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
freq=h
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=BeijingAQ

# reg_combo2/reduced-noft recipe:
concept_dim=64
bottleneck_dim=8
ema=0.9
online_learning_rate=0.000003

decoder_mode=common_channel

for seq_len in 512 1024 1536
do
for pred_len in 96 192 336 720
do
    filename=logs/reduced-noft/TTM'_'Proceed'_'$data'_'$seq_len'_'$pred_len'_'reduced-noft.log

    echo "[reduced-noft] data=$data seq=$seq_len pred=$pred_len cdim=$concept_dim bneck=$bottleneck_dim ema=$ema lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode $decoder_mode \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Proceed --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --ema $ema --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
