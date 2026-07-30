if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/proceed_sweep" ]; then
    mkdir ./logs/proceed_sweep
fi

# ---------------------------------------------------------------------------
# Q1 hyperparameter sweep: frozen (no online fine-tune) PROCEED + TinyTimeMixer
# on ETTh1.
#
# Diagnosis (see results.xlsx): frozen PROCEED starts as an exact identity
# (generator weights[-1]/bias init to zeros -> scale=1, shift=0), so it begins
# equal to zero-shot TTM and every online step moves it AWAY from that good
# point. Frozen PROCEED therefore trails Naive/zero-shot by a small margin that
# GROWS with context length. This sweep regularizes the adapter back toward its
# identity init to close that gap:
#   - lower bottleneck_dim  -> smaller rank of the drift->(scale,shift) map
#   - lower concept_dim     -> less noise amplification in the drift estimate
#   - ema > 0               -> smoother concept history -> less noisy adaptation
#   - lower online_lr       -> smaller steps away from identity (== zero-shot)
#
# Baseline == ttm-scripts/proceed-no-ft/ETTh1.sh: cdim=200 bneck=32 ema=0 lr=1e-4
# --freeze_online keeps the TTM backbone+head frozen; only the adapter adapts.
# ---------------------------------------------------------------------------

itr=1
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh2
freq=h

# Each config: "TAG concept_dim bottleneck_dim ema online_lr"
configs=(
    "baseline    200 32 0    0.0001"   # reference (proceed-no-ft/ETTh1.sh)
    "bneck16     200 16 0    0.0001"   # OFAT: rank
    "bneck8      200 8  0    0.0001"
    "bneck4      200 4  0    0.0001"
    "cdim64      64  32 0    0.0001"   # OFAT: concept dim
    "cdim32      32  32 0    0.0001"
    "ema90       200 32 0.9  0.0001"   # OFAT: concept EMA
    "ema99       200 32 0.99 0.0001"
    "lr3e5       200 32 0    0.00003"  # OFAT: online LR
    "reg_combo1  64  8  0.9  0.0001"   # combined regularized
    "reg_combo2  64  8  0.9  0.00003"  # combined regularized + low LR
)

for seq_len in 512 1536
do
for pred_len in 96 720
do
for cfg in "${configs[@]}"
do
    read tag concept_dim bottleneck_dim ema online_learning_rate <<< "$cfg"

    filename=logs/proceed_sweep/TTM'_'Proceed'_'$data'_'$seq_len'_'$pred_len'_'$tag'_'freezeonl.log

    echo "[sweep] seq=$seq_len pred=$pred_len tag=$tag cdim=$concept_dim bneck=$bottleneck_dim ema=$ema lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Proceed --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --ema $ema --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
done
