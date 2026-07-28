if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/moe_ssf_diag" ]; then mkdir ./logs/moe_ssf_diag; fi

# ---------------------------------------------------------------------------
# MoE-SSF REGRESSION DIAGNOSTIC.
#
# The dev sweep showed MoE+guard REGRESSES badly on Exchange 512_336
# (1.23 vs few-shot ~1.06 vs single-PROCEED ~0.78) and loses ETTh2 512_720
# (0.558 vs few-shot 0.475). The guard's E_full>>E_id on Exchange proves the MoE
# ADAPTATION is destructive there. Two suspects: (H1) concept_dim=128 is wrong
# (winning single configs used cdim64/cdim200 — the sweep never varied it);
# (H2) the mixture itself produces a worse adapter than a single tuned one.
#
# This isolates them on the two clear losers. Each config is one run per cell.
# Configs (all via the ProceedMoE code path so the comparison is apples-to-apples):
#   A) single-expert (bd8), cdim64, NO guard   = reg_combo2 reproduction.
#      SANITY/BUG CHECK — must land near Exchange ~0.79, ETTh2 ~0.505. If it does
#      NOT, there is a bug in the ProceedMoE path, not a modeling issue.
#   B) single-expert (bd8), cdim64, +guard     = good single adapter + guard.
#      Does the guard PRESERVE the win (Exchange) and tie the loss (ETTh2)?
#   C) MoE {8,32,64}, cdim64,  +guard          = does cdim64 fix the mixture?
#   D) MoE {8,32,64}, cdim200, +guard          = does cdim200 fix the mixture?
#   E) MoE {8,32,64}, cdim128, NO guard        = pure MoE adapter quality (isolate
#      the guard): if this is ~1.2 on Exchange, the MIXTURE adapter is the problem.
#
# Interpretation:
#   - A ~ baseline & B keeps Exchange ~0.78  -> guard is fine; DROP the MoE,
#     ship single-PROCEED + guard.
#   - C or D recovers Exchange              -> keep the MoE, fix was concept_dim.
#   - E ~ 1.2 on Exchange                   -> the mixture adapter is intrinsically
#     worse; prefer single adapter.
#
# Run on a real GPU. Baselines to compare against: results.xlsx (fewshot/reduced).
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
itr=1; train_ratio=0.5; test_ratio=0.45; batch_size=64; ema=0.9
olr=0.000003; rlr=0.001

# cells: "dataset seq pred freq"
cells=( "Exchange 512 336 d" "ETTh2 512 720 h" )

run () {  # name experts cdim guardflag
    local name=$1 experts=$2 cdim=$3 guard=$4
    for cell in "${cells[@]}"; do
        set -- $cell; data=$1; seq_len=$2; pred_len=$3; freq=$4
        local fn=logs/moe_ssf_diag/${data}_${seq_len}_${pred_len}_${name}.log
        echo "[diag] $name  data=$data seq=$seq_len pred=$pred_len experts=$experts cdim=$cdim guard=$guard"
        python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
        --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \
        --train_ratio $train_ratio --test_ratio $test_ratio \
        --online_method ProceedMoE --concept_dim $cdim --bottleneck_dim 64 --ema $ema --batch_size $batch_size \
        --expert_bottleneck_dims $experts --router_use_norm True \
        --router_learning_rate $rlr --moe_lb_coef 0 --moe_z_coef 0.001 \
        $guard --guard_tau 0.05 --guard_ema 0.95 \
        --online_learning_rate $olr --itr $itr >> $fn 2>&1
    done
}

run  A_single_bd8_cdim64_noguard   8       64  ""
run  B_single_bd8_cdim64_guard     8       64  "--use_guard"
run  C_moe_cdim64_guard            8,32,64 64  "--use_guard"
run  D_moe_cdim200_guard           8,32,64 200 "--use_guard"
run  E_moe_cdim128_noguard         8,32,64 128 ""

echo "[diag] done. Compare Exchange (want ~0.78) and ETTh2 (want ~0.48-0.50) across A-E."
