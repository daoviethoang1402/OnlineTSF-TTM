if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/moe_ssf_devtune" ]; then
    mkdir ./logs/moe_ssf_devtune
fi

# ---------------------------------------------------------------------------
# MoE-SSF DEV-TUNING sweep (frozen PROCEED-MoE + TTM).
#
# Purpose: pick the two dev-tuned knobs (guard_tau, concept_dim) ONCE on a small
# fixed DEV SET, then freeze them into the full-roster scripts
# (ttm-scripts/moe-ssf/<Dataset>.sh) for the held-out Phase C grid. Do NOT tune on
# the full grid — that would be tuning on test.
#
# The dev set spans the regimes the guard must reconcile:
#   ETTh1 512_96    — adaptation helps a little (want a small win)
#   ETTh1 1536_720  — adaptation HURTS (+8.7%); guard must abstain -> tie
#   ETTh2 512_720   — hold-out; adaptation hurts (+6.3%); guard must abstain
#   Exchange 512_336— adaptation helps a LOT (-26%); guard must NOT abstain
#   Weather 512_336 — weak drift / ~tie; guard should abstain -> tie
#
# Selection: pick the (guard_tau, concept_dim) that best keeps the Exchange win
# AND drives the loss/null cells back to ~few-shot (parse with parse_devtune.py).
# guard_tau is the primary knob (lower = sharper abstention); concept_dim is
# secondary. Default sweep = 3 tau x 1 cdim = 15 runs; add cdims to also sweep it.
#
# NOTE: the guard triples online forwards — run this on a real GPU, not the 2GB
# laptop card. Results/baselines: results.xlsx (fewshot / reduced-noft columns).
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
train_ratio=0.5
test_ratio=0.45
batch_size=64

# --- dev cells: "dataset seq_len pred_len freq" ---
cells=(
    "ETTh1 512 96 h"
    "ETTh1 1536 720 h"
    "ETTh2 512 720 h"
    "Exchange 512 336 d"
    "Weather 512 336 10min"
)

# --- swept knobs ---
taus=(0.05 0.1 0.2)
cdims=(128)          # add "64 200" here to also sweep concept_dim

# --- locked (non-swept) MoE-SSF config, identical to the roster scripts ---
expert_bottleneck_dims=8,32,64
router_use_norm=True
router_learning_rate=0.001
moe_lb_coef=0
moe_z_coef=0.001
ema=0.9
online_learning_rate=0.000003
guard_ema=0.95

for cell in "${cells[@]}"; do
    set -- $cell
    data=$1; seq_len=$2; pred_len=$3; freq=$4
    for cdim in "${cdims[@]}"; do
    for tau in "${taus[@]}"; do
        filename=logs/moe_ssf_devtune/TTM'_'ProceedMoE'_'$data'_'$seq_len'_'$pred_len'_'tau${tau}'_'cdim${cdim}.log

        echo "[dev-tune] data=$data seq=$seq_len pred=$pred_len guard_tau=$tau concept_dim=$cdim"

        python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
        --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
        --online_method ProceedMoE --concept_dim $cdim --bottleneck_dim 64 --ema $ema --batch_size $batch_size \
        --expert_bottleneck_dims $expert_bottleneck_dims --router_use_norm $router_use_norm \
        --router_learning_rate $router_learning_rate --moe_lb_coef $moe_lb_coef --moe_z_coef $moe_z_coef \
        --use_guard --guard_tau $tau --guard_ema $guard_ema \
        --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
    done
    done
done

echo "[dev-tune] done. Parse with: python ttm-scripts/moe-ssf/parse_devtune.py"
