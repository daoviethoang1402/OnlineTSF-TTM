if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/proceed_lr_sweep" ]; then
    mkdir ./logs/proceed_lr_sweep
fi

# ---------------------------------------------------------------------------
# Q1 follow-up: online-LR sweep of the reg_combo2 recipe (frozen PROCEED + TTM),
# ETTh1.
#
# The proceed-sweep found that lowering online_lr was the single dominant lever
# (improved MSE in 8/8 cells, monotone, never hit a floor), and reg_combo2
# (cdim64 + bneck8 + ema0.9 + lr3e5) was the best recipe (6/6). This sweep fixes
# the reg_combo2 capacity/ema settings and varies ONLY online_lr downward to
# locate the optimum:
#   - lr=3e-5 reproduces reg_combo2 exactly (also recovers the two 1536_720
#     cells that OOM'd in proceed-sweep due to external GPU contention);
#   - lr in {1e-5, 3e-6, 1e-6} extends below it.
# Watch for a horizon-dependent optimum: pred=720 is expected to prefer a
# smaller LR than pred=96 (where reg_combo2 already >= zero-shot).
#
# --freeze_online keeps the TTM backbone+head frozen; only the adapter adapts.
# ---------------------------------------------------------------------------

# Mitigate the fragmentation OOM seen in proceed-sweep (harmless if memory is free).
# For a truly busy card, also pin an idle GPU, e.g.:  export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh1
freq=h

# Fixed reg_combo2 recipe (everything except online_lr):
concept_dim=64
bottleneck_dim=8
ema=0.9

# Each config: "TAG online_lr"
configs=(
    "lr3e5  0.00003"   # == reg_combo2 (top of range; recovers OOM'd 1536_720)
    "lr1e5  0.00001"
    "lr3e6  0.000003"
    "lr1e6  0.000001"
)

for seq_len in 512 1536
do
for pred_len in 96 720
do
for cfg in "${configs[@]}"
do
    read tag online_learning_rate <<< "$cfg"

    filename=logs/proceed_lr_sweep/TTM'_'Proceed'_'$data'_'$seq_len'_'$pred_len'_'regc2'_'$tag'_'freezeonl.log

    echo "[lr-sweep] seq=$seq_len pred=$pred_len tag=$tag cdim=$concept_dim bneck=$bottleneck_dim ema=$ema lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Proceed --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --ema $ema --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
done
