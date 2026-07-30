if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/capacity-select" ]; then mkdir ./logs/capacity-select; fi

# ---------------------------------------------------------------------------
# Per-dataset CAPACITY selection on the 4 corners (VALIDATION ONLY).
#
# Single frozen PROCEED adapter (no MoE, no guard). For each dataset we search
# {2 capacity presets} x {2 learning rates} at the 4 grid corners
#   seq in {512, 1536} x pred in {96, MAXPRED}
# and pick, ON THE VALIDATION SET ONLY (--do_valid skips test), the per-dataset
# capacity + confirm the global lr-vs-horizon rule. Then extend the winner to all
# context/horizon cells (parse_corners.py --emit  ->  final/<Dataset>.sh).
#
#   * capacity: per dataset  (reduced {cdim64,bneck8}  vs  full {cdim200,bneck32})
#   * lr:       GLOBAL horizon rule (3e-6 short / 1e-6 long) -- we co-search both
#               lrs here only to VALIDATE the rule transfers to these datasets.
#   * ema:      0 (documented inert; removes a knob).
#
# 2 cap x 2 lr x 4 corners = 16 val-only runs / dataset. MAXPRED is per dataset
# (720 for most; 336 for Exchange/AirQuality if pred=720 is not runnable under the
# split -- adjust below if 720 IS runnable for them).
#
# FAIRNESS: report against the best FIXED config (reduced-noft / full-noft already
# in results.xlsx run-2), not just few-shot. Selection metric = "Best Valid MSE".
# Run on a real GPU.
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
itr=1; train_ratio=0.5; test_ratio=0.45; batch_size=64; ema=0

# dataset  freq   maxpred
datasets=(
    "ETTh1         h      720"
    "ETTh2         h      720"
    "ETTm1         15min  720"
    "ETTm2         15min  720"
    "Weather       10min  720"
    "Jiaolong_DSMS s      720"
    "wind          h      720"
    "Exchange      d      336"
    "Energy        10min  720"
    "AirQuality    h      336"
    "BeijingAQ     h      720"
)

# capacity presets:  name  cdim  bneck
caps=(
    "reduced 64  8"
    "full    200 32"
)

lrs=(0.000003 0.000001)     # 3e-6 (short), 1e-6 (long) -- co-searched to validate the rule

for ds in "${datasets[@]}"; do
    set -- $ds; data=$1; freq=$2; maxpred=$3
    for seq_len in 512 1536; do
    for pred_len in 96 $maxpred; do
        for cap in "${caps[@]}"; do
            set -- $cap; capname=$1; cdim=$2; bneck=$3
            for lr in "${lrs[@]}"; do
                fn=logs/capacity-select/${data}_${seq_len}_${pred_len}_${capname}_lr${lr}.log
                echo "[capsel] data=$data seq=$seq_len pred=$pred_len cap=$capname (cdim$cdim/bneck$bneck) lr=$lr"
                python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \
                --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \
                --train_ratio $train_ratio --test_ratio $test_ratio \
                --online_method Proceed --concept_dim $cdim --bottleneck_dim $bneck --ema $ema \
                --batch_size $batch_size --online_learning_rate $lr \
                --do_valid --itr $itr >> $fn 2>&1
            done
        done
    done
    done
done

echo "[capsel] done. Pick winners:  python ttm-scripts/capacity-select/parse_corners.py"
