if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/moe_ssf" ]; then
    mkdir ./logs/moe_ssf
fi

# ---------------------------------------------------------------------------
# MoE-SSF diagnostic #2: load-balancing coefficient sweep at ETTh1 512_96.
#
# First MoE result at this cell: MSE 0.4725 -- beats zero-shot (0.481) but LOSES
# to the dedicated single-expert reg_combo2 (0.466). Hypothesis: 512_96 has a
# SINGLE optimal capacity (low-rank bneck8), and the load-balancing loss
# (moe_lb_coef=0.01) forces the router toward a 50/50 blend, diluting the
# adaptation. Lowering / removing it should let the router CONCENTRATE on the
# bneck8 expert and recover toward 0.466.
#
# Sweep moe_lb_coef in {0, 0.001} (the 0.01 reference already exists in
# logs/moe_ssf/..._smoke.log). Router usage is logged so we can SEE whether it
# concentrates: watch the '[MoE router usage | phase=test]' line — expect the
# e0 (bneck8) mean gate to rise and FALLBACK/e1 to fall as lb_coef -> 0.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # pin an idle GPU if 0 is busy

itr=1
freq=h
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh1
seq_len=512
pred_len=96

concept_dim=64
online_learning_rate=0.000003
ema=0.9
expert_bottleneck_dims=8,32
router_hidden_dim=0
moe_z_coef=0.001
moe_log_every=2000

for moe_lb_coef in 0 0.001
do
    tag=$(echo "lb$moe_lb_coef" | tr -d '.')
    filename=logs/moe_ssf/TTM'_'ProceedMoE'_'$data'_'$seq_len'_'$pred_len'_'$tag.log

    echo "[moe-ssf lbcoef] data=$data seq=$seq_len pred=$pred_len lb_coef=$moe_lb_coef"

    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
    --online_method ProceedMoE --concept_dim $concept_dim --ema $ema --batch_size $batch_size \
    --expert_bottleneck_dims $expert_bottleneck_dims --router_hidden_dim $router_hidden_dim \
    --moe_lb_coef $moe_lb_coef --moe_z_coef $moe_z_coef --moe_log_every $moe_log_every \
    --online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
done
