if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/proceed_ema0_sweep" ]; then
    mkdir ./logs/proceed_ema0_sweep
fi

# ---------------------------------------------------------------------------
# Step-2 diagnostic: ema=0 vs ema=0.9 at long horizon (pred=720), ETTh2.
#
# The LR sweep left the 720 cells split under the reg_combo2 recipe (ema=0.9):
#   - ETTh1 1536_720 was FLAT ~0.84 across all LRs, WORSE than zero-shot (0.777)
#     -> suspected ema=0.9 giving a stale/biased drift estimate at long horizon.
#   - ETTh2 1536_720 improved monotonically with lower LR and BEAT zero-shot
#     -> ema=0.9 was working there, so ema=0 is a RISK, not obviously a win.
# This run turns OFF ema (ema=0) and re-tests online_lr in {3e-6, 1e-6} at
# pred=720 only. The control (ema=0.9 at the same cells/LRs) already exists in
# logs/proceed_lr_sweep/, so this is a clean A/B. MUST be run on BOTH datasets:
# ETTh1 is where ema=0 should help, ETTh2 is where it could regress a winner.
#
# Recipe: reg_combo2 with ema=0  ->  cdim=64  bneck=8  ema=0.
# 3e-7 is intentionally NOT run here; add it only if 1e-6 still wins under ema=0.
# --freeze_online keeps the TTM backbone+head frozen; only the adapter adapts.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh2
freq=h

# Fixed reg_combo2 recipe with ema turned OFF:
concept_dim=64
bottleneck_dim=8
ema=0

pred_len=720

# Each config: "TAG online_lr"
configs=(
    "lr3e6  0.000003"
    "lr1e6  0.000001"
)

for seq_len in 512 1536
do
for cfg in "${configs[@]}"
do
    read tag online_learning_rate <<< "$cfg"

    filename=logs/proceed_ema0_sweep/TTM'_'Proceed'_'$data'_'$seq_len'_'$pred_len'_'regc2ema0'_'$tag'_'freezeonl.log

    echo "[ema0-sweep] seq=$seq_len pred=$pred_len tag=$tag cdim=$concept_dim bneck=$bottleneck_dim ema=$ema lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Proceed --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --ema $ema --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
