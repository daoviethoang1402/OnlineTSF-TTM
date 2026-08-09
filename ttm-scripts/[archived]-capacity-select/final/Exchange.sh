if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/capacity-final" ]; then mkdir ./logs/capacity-final; fi

# FINAL full grid for Exchange. Single frozen PROCEED (no MoE/guard).
# Decided on validation corners: capacity=full (cdim200/bneck32), lr=0.000003, ema=0.  (== existing proceed-full-noft/Exchange.sh; reuse run-2)
# Interior cells (seq 1024; pred 192/336) are held-out generalization checks of
# the corner-selected config. Compare per cell vs few-shot AND best-fixed
# (reduced-noft/full-noft, run-2).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data=Exchange; freq=d; cdim=200; bneck=32; lr=0.000003
for seq_len in 512 1024 1536
do
for pred_len in 96 192 336
do
    fn=logs/capacity-final/${data}_${seq_len}_${pred_len}_full_lr${lr}.log
    echo "[capfinal] data=$data seq=$seq_len pred=$pred_len cap=full lr=$lr"
    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \
    --train_ratio 0.5 --test_ratio 0.45 \
    --online_method Proceed --concept_dim $cdim --bottleneck_dim $bneck --ema 0 \
    --batch_size 64 --online_learning_rate $lr --itr 1 >> $fn 2>&1
done
done
