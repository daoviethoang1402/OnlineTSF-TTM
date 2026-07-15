if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/proceed_capacity_sweep" ]; then
    mkdir ./logs/proceed_capacity_sweep
fi

# ---------------------------------------------------------------------------
# Capacity diagnostic: does FULL capacity (cdim=200, bneck=32) at the new,
# properly-tuned low LR close the long-horizon (pred=720) hold-out gap? ETTh1.
#
# Context: reg_combo2 (cdim=64, bneck=8, ema=0.9) @ lr=3e-6 is the best overall
# recipe so far (beats zero-shot on 6/8 cells). But two 720 cells remain
# hold-outs where NO reduced-capacity config beats zero-shot:
#   - ETTh1 1536_720: flat ~0.846 regardless of ema/LR (worse than zero 0.777)
#   - ETTh2 512_720:  ~0.505-0.512 (worse than zero 0.475)
# ema was already shown to be inert (see documents/PROCEED_TTM_HYPERPARAM_NOTES.md),
# so this diagnostic isolates CAPACITY as the remaining variable: is the
# hold-out gap because reduced capacity (bneck=8) under-fits the long-horizon
# drift, or because adaptation fundamentally can't help there regardless of
# capacity?
#
# Interpretation:
#   - full-cap + low-LR closes/narrows the gap  -> hold-outs are capacity-
#     starved -> MoE-SSF (more adaptation capacity) is justified.
#   - full-cap + low-LR does NOT help (or is worse, as we saw at the old
#     lr=3e-5 where full-cap 0.786 still lost to zero-shot 0.777) -> those
#     cells can't be helped by more scale/shift capacity alone -> prefer a
#     horizon-aware near-identity fallback there and scope MoE to cells that
#     already show adaptation signal.
#
# Scope: pred=720 only, both seq_len, both datasets (ETTh1 here, ETTh2 in the
# sibling script) = 8 runs total. ema fixed at 0.9 (shown inert, kept for
# consistency with the winning recipe).
# --freeze_online keeps the TTM backbone+head frozen; only the adapter adapts.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh1
freq=h

# Full-capacity generator (PROCEED default), ema fixed at 0.9 (inert but kept
# consistent with the winning recipe):
concept_dim=200
bottleneck_dim=32
ema=0.9

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

    filename=logs/proceed_capacity_sweep/TTM'_'Proceed'_'$data'_'$seq_len'_'$pred_len'_'fullcap'_'$tag'_'freezeonl.log

    echo "[capacity-sweep] seq=$seq_len pred=$pred_len tag=$tag cdim=$concept_dim bneck=$bottleneck_dim ema=$ema lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method Proceed --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --ema $ema --batch_size $batch_size \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
