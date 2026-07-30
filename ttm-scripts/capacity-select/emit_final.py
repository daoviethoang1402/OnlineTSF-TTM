#!/usr/bin/env python3
"""Emit the FINAL full-grid run scripts from the DECIDED per-dataset config
(chosen on the validation corners; see analysis). Single frozen PROCEED adapter
(no MoE, no guard). Writes ttm-scripts/capacity-select/final/<Dataset>.sh, each
running every context/horizon cell on TEST.

Decision (per-dataset capacity + drift-based lr, ema=0):
  * capacity: full-noft (cdim200/bneck32) for Exchange, ETTh2, Energy;
              reduced-noft (cdim64/bneck8) for the other 8.
  * lr: 3e-6 for Exchange (genuinely non-stationary -> more online adaptation);
        1e-6 for the rest.
Exchange (full, 3e-6) already exists as ttm-scripts/proceed-full-noft/Exchange.sh
and in results.xlsx run-2; its script is emitted here only for a unified folder.

Usage: python ttm-scripts/capacity-select/emit_final.py
"""
import os

FULL = (200, 32)       # cdim, bneck
REDUCED = (64, 8)

# dataset -> (capacity, online_lr)
CONFIG = {
    "Exchange":      (FULL,    "0.000003"),   # reuse existing full-noft@3e-6
    "ETTh2":         (FULL,    "0.000001"),
    "Energy":        (FULL,    "0.000001"),
    "ETTh1":         (REDUCED, "0.000001"),
    "ETTm1":         (REDUCED, "0.000001"),
    "ETTm2":         (REDUCED, "0.000001"),
    "Weather":       (REDUCED, "0.000001"),
    "Jiaolong_DSMS": (REDUCED, "0.000001"),
    "wind":          (REDUCED, "0.000001"),
    "AirQuality":    (REDUCED, "0.000001"),
    "BeijingAQ":     (REDUCED, "0.000001"),
}
# dataset -> (freq, maxpred)
META = {
    "ETTh1": ("h", 720), "ETTh2": ("h", 720), "ETTm1": ("15min", 720),
    "ETTm2": ("15min", 720), "Weather": ("10min", 720), "Jiaolong_DSMS": ("s", 720),
    "wind": ("h", 720), "Exchange": ("d", 336), "Energy": ("10min", 720),
    "AirQuality": ("h", 336), "BeijingAQ": ("h", 720),
}

OUT = "ttm-scripts/capacity-select/final"
os.makedirs(OUT, exist_ok=True)

for data, ((cdim, bneck), lr) in CONFIG.items():
    freq, maxpred = META[data]
    preds = "96 192 336 720" if maxpred == 720 else "96 192 336"
    capname = "full" if (cdim, bneck) == FULL else "reduced"
    fname = "Jiaolong.sh" if data == "Jiaolong_DSMS" else f"{data}.sh"
    reuse = "  (== existing proceed-full-noft/Exchange.sh; reuse run-2)" if data == "Exchange" else ""
    body = f"""if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/capacity-final" ]; then mkdir ./logs/capacity-final; fi

# FINAL full grid for {data}. Single frozen PROCEED (no MoE/guard).
# Decided on validation corners: capacity={capname} (cdim{cdim}/bneck{bneck}), lr={lr}, ema=0.{reuse}
# Interior cells (seq 1024; pred 192/336) are held-out generalization checks of
# the corner-selected config. Compare per cell vs few-shot AND best-fixed
# (reduced-noft/full-noft, run-2).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data={data}; freq={freq}; cdim={cdim}; bneck={bneck}; lr={lr}
for seq_len in 512 1024 1536
do
for pred_len in {preds}
do
    fn=logs/capacity-final/${{data}}_${{seq_len}}_${{pred_len}}_{capname}_lr${{lr}}.log
    echo "[capfinal] data=$data seq=$seq_len pred=$pred_len cap={capname} lr=$lr"
    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \\
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \\
    --train_ratio 0.5 --test_ratio 0.45 \\
    --online_method Proceed --concept_dim $cdim --bottleneck_dim $bneck --ema 0 \\
    --batch_size 64 --online_learning_rate $lr --itr 1 >> $fn 2>&1
done
done
"""
    with open(os.path.join(OUT, fname), "w") as f:
        f.write(body)
    print(f"emitted {OUT}/{fname:14} cap={capname:7} lr={lr}")

print("\nNOTE: Exchange reuses the existing full-noft@3e-6 result (run-2); the other "
      "10 are new 1e-6 runs. Run the non-Exchange finals, then compare on TEST.")
