if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/moe_ssf" ]; then
    mkdir ./logs/moe_ssf
fi

# ---------------------------------------------------------------------------
# MoE-SSF MVP smoke test (frozen PROCEED-MoE + TTM), ETTh1.
#
# Purpose: validate the ProceedMoE path runs end-to-end and eyeball router
# behavior. Two cells span the regimes the MoE is meant to reconcile:
#   - 512_96:   short horizon, where fixed reg_combo2 already BEATS zero-shot
#               (router should mostly use the low-capacity expert, little fallback)
#   - 1536_720: long-horizon hold-out, where fixed recipes LOSE to zero-shot
#               (router should lean on the full-capacity expert and/or fallback)
#
# Two experts spanning the capacity range we characterized (reduced=8, full=32)
# plus the implicit identity/fallback slot. LR = 3e-6 (the tuned frozen value).
# ema=0.9 (inert, kept for consistency). Watch the '[ProceedMoE]' banner and the
# router usage in the printed model / logs.
#
# NOTE: run the module self-test first in the torch env:
#   python3 -m adapter.module.moe
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

itr=1
freq=h
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh1

concept_dim=64
online_learning_rate=0.000003
ema=0.9

# MoE knobs
expert_bottleneck_dims=8,32   # reduced + full expert; +1 fallback slot added automatically
router_hidden_dim=0           # linear router
moe_lb_coef=0.01
moe_z_coef=0.001

for seq_len in 512 1536
do
for pred_len in 96 720
do
    # only the two corner cells (512_96, 1536_720)
    if { [ "$seq_len" = "512" ] && [ "$pred_len" = "96" ]; } || \
       { [ "$seq_len" = "1536" ] && [ "$pred_len" = "720" ]; }; then

        filename=logs/moe_ssf/TTM'_'ProceedMoE'_'$data'_'$seq_len'_'$pred_len'_'smoke.log

        echo "[moe-ssf smoke] data=$data seq=$seq_len pred=$pred_len experts=$expert_bottleneck_dims lr=$online_learning_rate"

        python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
        --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
        --online_method ProceedMoE --concept_dim $concept_dim --ema $ema --batch_size $batch_size \
        --expert_bottleneck_dims $expert_bottleneck_dims --router_hidden_dim $router_hidden_dim \
        --moe_lb_coef $moe_lb_coef --moe_z_coef $moe_z_coef \
        --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
    fi
done
done
