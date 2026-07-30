if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/probe_forget" ]; then
    mkdir ./logs/probe_forget
fi

# ---------------------------------------------------------------------------
# Phase 1 — is PROCEED-ft's advantage over naive-ft actually forgetting
# mitigation?  (User hypothesis: PROCEED-ft > naive-ft because PROCEED compensates
# for catastrophic forgetting during online fine-tuning.)
#
# Design: LR-ROBUSTNESS as the forgetting signature. Catastrophic forgetting in
# online fine-tuning is driven by aggressive updates overwriting pretrained
# knowledge, so it should scale with the online LR:
#   - naive-ft updates the whole TTM  -> higher LR overwrites the backbone ->
#     FORGETS -> error degrades sharply (can fall below few-shot).
#   - PROCEED-ft's low-rank adapter cannot overwrite the frozen backbone ->
#     structurally forgetting-resistant -> error stays ~flat as LR rises.
# The DIVERGENCE of the two degradation curves (vs the few-shot anchor) is the
# evidence. Sweep online_lr in {1e-4, 1e-3, 1e-2}.
#
# Datasets: ETTh1 (CONTROL: adaptation is known to help a bit here) and Exchange
# (PROBE: candidate high-drift). pred 96 (method behavior; one horizon suffices).
# Read with ttm-scripts/probe/analyze_probe.py (--forget), which plots MSE vs LR
# per method.
#
# LIMITATION (honest): this is an indirect but mechanism-targeted test. A DIRECT
# measure = evaluate the post-online model on the source/train distribution
# (backward transfer). The framework does not persist the online-adapted model to
# disk (only an in-memory copy during vali), so that needs a small save hook +
# an eval-on-train mode. Ask if you want it added; this LR-robustness probe is the
# runnable first cut.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

itr=1
train_ratio=0.5
test_ratio=0.45
seq_len=512
pred_len=96
learning_rate=0.0001
batch_size=64

run_it () {  # $1 tag  $2 data  $3 freq  ...rest
    local tag=$1 data=$2 freq=$3; shift 3
    local f=logs/probe_forget/${tag}_${data}_${seq_len}_${pred_len}.log
    echo "[forget] $tag data=$data pred=$pred_len"
    python run.py --model TinyTimeMixer --decoder_mode common_channel \
        --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \
        --train_ratio $train_ratio --test_ratio $test_ratio --batch_size $batch_size \
        --learning_rate $learning_rate --itr $itr "$@" >> $f 2>&1
}

# "dataset freq"
for cfg in "ETTh1 h" "Exchange d"; do
    set -- $cfg; data=$1; freq=$2

    # few-shot anchor (also the checkpoint naive-ft loads) -- run FIRST
    run_it fewshot $data $freq

    for lr in 0.0001 0.001 0.01; do
        tag=$(echo "lr$lr" | tr -d '.')
        # naive online fine-tune (whole model) -> expected to forget at high LR
        run_it naiveft_$tag  $data $freq --online_method Online --online_learning_rate $lr
        # PROCEED online fine-tune -> expected forgetting-resistant
        run_it proceedft_$tag $data $freq --online_method Proceed \
               --concept_dim 200 --bottleneck_dim 32 --online_learning_rate $lr
    done
done
