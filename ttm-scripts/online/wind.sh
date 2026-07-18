if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

if [ ! -d "./logs/proceed" ]; then
    mkdir ./logs/proceed
fi

learning_rate=0.0001
online_learning_rate=0.0001
itr=1
freq=h
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=wind

for seq_len in 512 1024 1536
do
for pred_len in 96 192 336 720
do
    filename=logs/online/TTM'_'Online'_'$data'_'$seq_len'_'$pred_len.log

    python run.py --model TinyTimeMixer  --decoder_mode mix_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --batch_size $batch_size \
    --learning_rate $learning_rate --itr $itr >> $filename 2>&1
    
    python run.py --model TinyTimeMixer  --decoder_mode mix_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Online --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done