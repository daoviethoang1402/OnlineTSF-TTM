if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/moe_ssf" ]; then
    mkdir ./logs/moe_ssf
fi

# ---------------------------------------------------------------------------
# MoE-SSF diagnostic #3: re-run the ETTh1 1536_720 hold-out (OOM'd in the smoke
# test due to external GPU contention -- the card had only ~1.65 GiB free).
#
# This is the cell that MATTERS: the long-horizon hold-out where no fixed recipe
# beats zero-shot (reg_combo2 = 0.845 MSE vs zero-shot 0.777). The mixture (full-
# capacity bneck32 expert) + fallback should have headroom here. Router usage is
# logged: watch whether it leans on e1 (bneck32) and/or FALLBACK.
#
# IMPORTANT: run on an IDLE GPU. The MoE is somewhat heavier than single-expert
# (K expert forwards + large seq=1536/pred=720 activations). Set CUDA_VISIBLE_DEVICES
# to a free card. If it still OOMs on a clean card, drop batch_size to 32.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # <-- set to an idle GPU

itr=1
freq=h
train_ratio=0.5
test_ratio=0.45
batch_size=64
data=ETTh1
seq_len=1536
pred_len=720

concept_dim=64
online_learning_rate=0.000003
ema=0.9
expert_bottleneck_dims=8,32
router_hidden_dim=0
moe_lb_coef=0.01
moe_z_coef=0.001
moe_log_every=2000

filename=logs/moe_ssf/TTM'_'ProceedMoE'_'$data'_'$seq_len'_'$pred_len'_'rerun.log

echo "[moe-ssf 1536_720 rerun] data=$data seq=$seq_len pred=$pred_len (idle GPU required)"

python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
--dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq --train_ratio $train_ratio --test_ratio $test_ratio \
--online_method ProceedMoE --concept_dim $concept_dim --ema $ema --batch_size $batch_size \
--expert_bottleneck_dims $expert_bottleneck_dims --router_hidden_dim $router_hidden_dim \
--moe_lb_coef $moe_lb_coef --moe_z_coef $moe_z_coef --moe_log_every $moe_log_every \
--online_learning_rate $online_learning_rate --itr $itr >> $filename 2>&1
