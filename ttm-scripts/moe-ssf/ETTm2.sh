if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/moe_ssf" ]; then
    mkdir ./logs/moe_ssf
fi

# ---------------------------------------------------------------------------
# MoE-SSF (frozen PROCEED-MoE + TTM) on ETTm2.
#
# Mixture-of-SSF-experts generator (heterogeneous rank 8,32,64 + identity
# fallback) with a drift+||drift|| router, PLUS the online abstention GUARD
# (--use_guard): on each already-revealed window it compares full-adaptation vs
# identity (few-shot) error and shrinks the mixture toward identity where
# adaptation has been hurting (no leakage). This is what turns the fixed recipe's
# "wins small, loses big" into "wins where there's headroom, ties elsewhere".
#
#   * Router picks WHICH capacity to use when adapting (gradient-learned).
#   * Guard decides WHETHER to adapt at all (online, non-gradient).
#
# --freeze_online keeps the TTM backbone+head frozen; only the adapter adapts.
# Drop --use_guard for the pure-MoE ablation. guard_tau / concept_dim are the
# dev-tuned knobs (locked on ETT corners + Exchange 336 + a Weather cell) — update
# here after Phase B if they move. online_learning_rate=3e-6 is the tuned value.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
freq=15min
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTm2

# --- locked MoE-SSF config ---
concept_dim=128                 # dev-tuned; shared drift representation for router+experts
bottleneck_dim=64               # ignored when expert_bottleneck_dims is set (kept for completeness)
expert_bottleneck_dims=8,32,64  # heterogeneous-rank experts (+1 identity/fallback slot)
router_use_norm=True            # feed ||drift|| to the router (abstention signal)
router_learning_rate=0.001      # decoupled, higher router LR
moe_lb_coef=0                   # no load-balancing (lb pushes against abstention; lb=0 best in smoke)
moe_z_coef=0.001
ema=0.9                         # inert, kept for consistency
online_learning_rate=0.000003   # 3e-6, the tuned frozen value

# --- guard config ---
guard_tau=0.1                   # dev-tuned; lower = sharper abstention
guard_ema=0.95

for seq_len in 512 1024 1536
do
for pred_len in 96 192 336 720
do
    filename=logs/moe_ssf/TTM'_'ProceedMoE'_'$data'_'$seq_len'_'$pred_len'_'guard.log

    echo "[moe-ssf] data=$data seq=$seq_len pred=$pred_len experts=$expert_bottleneck_dims guard_tau=$guard_tau lr=$online_learning_rate"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method ProceedMoE --concept_dim $concept_dim --bottleneck_dim $bottleneck_dim --ema $ema --batch_size $batch_size \
    --expert_bottleneck_dims $expert_bottleneck_dims --router_use_norm $router_use_norm \
    --router_learning_rate $router_learning_rate --moe_lb_coef $moe_lb_coef --moe_z_coef $moe_z_coef \
    --use_guard --guard_tau $guard_tau --guard_ema $guard_ema \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
done
