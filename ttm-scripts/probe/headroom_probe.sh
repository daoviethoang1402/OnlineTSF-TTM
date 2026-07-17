if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/probe" ]; then
    mkdir ./logs/probe
fi

# ---------------------------------------------------------------------------
# Phase 0 — behavioral headroom probe on UNTRIED datasets.
#
# Question the headroom analysis left open: on the standard benchmarks the oracle
# adapt-or-abstain ceiling over few-shot TTM is only ~0.9% (input drift stats do
# NOT predict it). Pivot rule: select datasets by whether the STRONG few-shot TTM
# baseline behaviorally DEGRADES, not by "has drift". This probe runs four
# variants and asks: is there a dataset where online adaptation beats few-shot by
# a meaningful margin (=> real headroom => a place the thesis can win)?
#
# Variants (all seq_len 512, decoder_mode common_channel for consistency):
#   1. fewshot     : TTM head fine-tuned on train split, then just tested (no
#                    online updates). No online_method. Saves the checkpoint that
#                    naive-ft loads.  == the "Naive (frozen)" baseline.
#   2. naiveft     : plain online gradient fine-tuning (--online_method Online),
#                    loads the fewshot checkpoint. Does online adaptation help at
#                    all — and does it FORGET (worse than fewshot)?
#   3. proceedfrz  : reg_combo2 (frozen backbone): cdim64 bneck8 ema0.9 lr3e-6.
#                    Memory-efficient adaptation.
#   4. proceedft   : PROCEED online fine-tune (default cdim200 bneck32 lr1e-4).
#                    The most powerful adapter — best test of "is anything winnable".
#
# Read with ttm-scripts/probe/analyze_probe.py. A dataset is a keeper if
# proceedft (or proceedfrz) beats fewshot by clearly more than the ~1% we saw on
# ETT/Weather. Candidates chosen for likely OUT-OF-CORPUS drift (financial / wind)
# where a weather+energy-pretrained TTM is more likely to fail.
#
# NOTE: fewshot MUST run before naive-ft (naive-ft loads its checkpoint). Verify
# on the first cell that naive-ft actually loads the checkpoint (the one fragile
# link) rather than silently retraining.
# Illness is excluded: 966 rows < seq_len 512, no training windows fit TTM.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

itr=1
train_ratio=0.5
test_ratio=0.45
seq_len=512
learning_rate=0.0001

run_variant () {
    # $1 tag  $2 data  $3 freq  $4 bs  $5 pred  ...rest = extra run.py args
    local tag=$1 data=$2 freq=$3 bs=$4 pred=$5; shift 5
    local f=logs/probe/${tag}_${data}_${seq_len}_${pred}.log
    echo "[probe] $tag data=$data seq=$seq_len pred=$pred"
    python run.py --model TinyTimeMixer --decoder_mode common_channel \
        --dataset $data --seq_len $seq_len --pred_len $pred --freq $freq \
        --train_ratio $train_ratio --test_ratio $test_ratio --batch_size $bs \
        --learning_rate $learning_rate --itr $itr "$@" >> $f 2>&1
}

# "dataset  freq  batch_size  preds..."
configs=(
    "Exchange d 64 96 720"
    "wind     h 64 96 720"
    # optional heavy tier (hundreds of channels -> slow / lower headroom expected):
    # "Traffic h 8 96 720"
    # "ECL     h 8 96 720"
)

for cfg in "${configs[@]}"; do
    set -- $cfg; data=$1; freq=$2; bs=$3; shift 3; preds="$@"
    for pred in $preds; do
        # 1) few-shot (also writes the checkpoint naive-ft needs) -- run FIRST
        run_variant fewshot    $data $freq $bs $pred
        # 2) naive online fine-tune (loads few-shot checkpoint)
        run_variant naiveft    $data $freq $bs $pred --online_method Online --online_learning_rate 0.0001
        # 3) PROCEED frozen (reg_combo2)
        run_variant proceedfrz $data $freq $bs $pred --online_method Proceed --freeze_online \
                    --concept_dim 64 --bottleneck_dim 8 --ema 0.9 --online_learning_rate 0.000003
        # 4) PROCEED online fine-tune (default)
        run_variant proceedft  $data $freq $bs $pred --online_method Proceed \
                    --concept_dim 200 --bottleneck_dim 32 --online_learning_rate 0.0001
    done
done
