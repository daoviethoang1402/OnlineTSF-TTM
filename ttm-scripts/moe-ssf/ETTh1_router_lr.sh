if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/moe_ssf" ]; then
    mkdir ./logs/moe_ssf
fi

# ---------------------------------------------------------------------------
# MoE-SSF diagnostic #4: decoupled router learning rate.
#
# Finding so far: with the router sharing the tiny online LR (3e-6), its gates
# stayed NEAR-UNIFORM (~1/3 each) and it never learned to abstain -- the gains
# were from implicit ensembling, not routing. Fix: the router now lives in its
# own optimizer param group with `--router_learning_rate` (experts/adapter stay
# at --online_learning_rate=3e-6).
#
# This sweeps the router LR on the two key cells and logs usage so we can SEE
# whether a faster router (a) concentrates and (b) starts using FALLBACK on the
# 1536_720 hold-out (where zero-shot beats adaptation, so it SHOULD abstain).
#   - 512_96:   short horizon; expect concentration on e0 (bneck8), low fallback.
#   - 1536_720: hold-out; expect rising fallback / lean on e1 (bneck32).
# Decisive test of whether a learned feed-forward router is viable at all.
#
# lb_coef=0 (the balancing loss was diluting; see the lbcoef diagnostic).
# Run on an IDLE GPU (esp. 1536_720). Drop batch_size to 32 if 1536_720 OOMs.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # <-- set to an idle GPU

itr=1
freq=h
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh1

concept_dim=64
online_learning_rate=0.000003     # experts/adapter (unchanged)
ema=0.9
expert_bottleneck_dims=8,32
router_hidden_dim=0
moe_lb_coef=0                      # balancing off (was diluting)
moe_z_coef=0.001
moe_log_every=2000

# (seq_len pred_len) cells
cells=("512 96" "1536 720")

for router_learning_rate in 0.001 0.01
do
for cell in "${cells[@]}"
do
    read seq_len pred_len <<< "$cell"
    tag=$(echo "rlr$router_learning_rate" | tr -d '.')
    filename=logs/moe_ssf/TTM'_'ProceedMoE'_'$data'_'$seq_len'_'$pred_len'_'$tag.log

    echo "[moe-ssf router_lr] data=$data seq=$seq_len pred=$pred_len router_lr=$router_learning_rate online_lr=$online_learning_rate lb=$moe_lb_coef"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method ProceedMoE --concept_dim $concept_dim --ema $ema --batch_size $batch_size \
    --expert_bottleneck_dims $expert_bottleneck_dims --router_hidden_dim $router_hidden_dim \
    --moe_lb_coef $moe_lb_coef --moe_z_coef $moe_z_coef --moe_log_every $moe_log_every \
    --router_learning_rate $router_learning_rate \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
