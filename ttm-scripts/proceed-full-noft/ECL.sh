if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/full-noft" ]; then
    mkdir ./logs/full-noft
fi

itr=1
freq=h
train_ratio=0.5
test_ratio=0.45
batch_size=16
data=ECL

concept_dim=200
bottleneck_dim=32
ema=0
online_learning_rate=0.000003

decoder_mode=common_channel

for seq_len in 512 1024 1536
do
for pred_len in 96 192 336 720
do
    filename=logs/full-noft/TTM'_'Proceed'_'$data'_'$seq_len'_'$pred_len'_'full-noft.log

    echo "[full-noft] data=$data seq=$seq_len pred=$pred_len cdim=$concept_dim bneck=$bottleneck_dim ema=$ema lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode $decoder_mode \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Proceed --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done