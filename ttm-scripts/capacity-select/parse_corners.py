#!/usr/bin/env python3
"""Parse the corner capacity-selection logs (logs/capacity-select/), all on the
VALIDATION metric ("Best Valid MSE"), and:

  1. VALIDATE the global lr-vs-horizon rule (does 3e-6 win the short/pred=96
     corners and 1e-6 win the long/max-pred corners?).
  2. PICK the per-dataset capacity preset (reduced vs full), using the rule's lr
     at each corner, averaged over the 4 corners.
  3. With --emit, write the final full-grid run scripts
     (ttm-scripts/capacity-select/final/<Dataset>.sh) that extend the chosen
     per-dataset capacity + the global lr rule to every context/horizon cell
     (TEST, no --do_valid).

Selection uses validation only; test is never read here. Usage:
    python ttm-scripts/capacity-select/parse_corners.py [logdir] [--emit]
"""
import os
import re
import sys
import glob
from collections import defaultdict

LOGDIR = "logs/capacity-select"
EMIT = "--emit" in sys.argv
for a in sys.argv[1:]:
    if not a.startswith("-"):
        LOGDIR = a

SHORT_LR, LONG_LR = "0.000003", "0.000001"     # 3e-6 short, 1e-6 long
# dataset -> (freq, maxpred); keep in sync with corner_select.sh
DATASETS = {
    "ETTh1": ("h", 720), "ETTh2": ("h", 720), "ETTm1": ("15min", 720),
    "ETTm2": ("15min", 720), "Weather": ("10min", 720), "Jiaolong_DSMS": ("s", 720),
    "wind": ("h", 720), "Exchange": ("d", 336), "Energy": ("10min", 720),
    "AirQuality": ("h", 336), "BeijingAQ": ("h", 720),
}
CAPS = {"reduced": (64, 8), "full": (200, 32)}

FN = re.compile(r"^(?P<data>.+)_(?P<seq>\d+)_(?P<pred>\d+)_(?P<cap>reduced|full)_lr(?P<lr>[0-9.e-]+)\.log$")
VAL = re.compile(r"Best Valid MSE:\s*([0-9.]+)")


def rule_lr(pred):
    return SHORT_LR if int(pred) <= 192 else LONG_LR


# res[(data, seq, pred, cap, lr)] = val_mse
res = {}
for path in sorted(glob.glob(os.path.join(LOGDIR, "*.log"))):
    m = FN.match(os.path.basename(path))
    if not m:
        continue
    v = VAL.findall(open(path, errors="ignore").read())
    if v:
        res[(m["data"], int(m["seq"]), int(m["pred"]), m["cap"], m["lr"])] = float(v[-1])

if not res:
    print(f"No parseable 'Best Valid MSE' logs in {LOGDIR}. Run corner_select.sh first.")
    sys.exit(0)

datasets = sorted({k[0] for k in res})

# ---- 1. lr-rule validation ------------------------------------------------ #
print("\n=== lr-rule check (per corner: which lr has lower VAL MSE) ===")
print(f"{'data':14} {'seq':>5} {'pred':>5} {'cap':>8} {'lr3e-6':>9} {'lr1e-6':>9}  winner")
short_ok = short_tot = long_ok = long_tot = 0
for (data, seq, pred, cap, lr), _ in sorted(res.items()):
    if lr != SHORT_LR:
        continue
    m3 = res.get((data, seq, pred, cap, SHORT_LR))
    m1 = res.get((data, seq, pred, cap, LONG_LR))
    if m3 is None or m1 is None:
        continue
    win = "3e-6" if m3 <= m1 else "1e-6"
    if int(pred) <= 192:
        short_tot += 1; short_ok += (win == "3e-6")
    else:
        long_tot += 1; long_ok += (win == "1e-6")
    print(f"{data:14} {seq:>5} {pred:>5} {cap:>8} {m3:>9.4f} {m1:>9.4f}  {win}")
if short_tot and long_tot:
    print(f"\nrule holds: short(pred<=192) prefers 3e-6 in {short_ok}/{short_tot}; "
          f"long prefers 1e-6 in {long_ok}/{long_tot}")
    print("  (if these are near-unanimous the global rule transfers; if not, "
          "note lr is near-inert in the amortized regime and pick the simplest rule)")

# ---- 2. per-dataset capacity pick (using rule lr) ------------------------- #
print("\n=== per-dataset capacity pick (rule lr at each corner, mean VAL over corners) ===")
print(f"{'data':14} {'reduced':>9} {'full':>9}  {'pick':>8}  margin")
chosen = {}
for data in datasets:
    seqs, preds = (512, 1536), (96, DATASETS.get(data, ("h", 720))[1])
    means = {}
    for cap in ("reduced", "full"):
        vals = [res.get((data, s, p, cap, rule_lr(p))) for s in seqs for p in preds]
        vals = [v for v in vals if v is not None]
        means[cap] = sum(vals) / len(vals) if vals else None
    if means["reduced"] is None or means["full"] is None:
        print(f"{data:14}  (incomplete logs)")
        continue
    pick = "reduced" if means["reduced"] <= means["full"] else "full"
    chosen[data] = pick
    margin = abs(means["reduced"] - means["full"]) / max(means["reduced"], means["full"]) * 100
    print(f"{data:14} {means['reduced']:>9.4f} {means['full']:>9.4f}  {pick:>8}  {margin:5.1f}%")

print("\n-> chosen per-dataset capacity:")
for d, c in chosen.items():
    print(f"     {d:14} -> {c} (cdim{CAPS[c][0]}/bneck{CAPS[c][1]})")
print("   NOTE: verify the pick generalizes on the held-out INTERIOR cells after "
      "the final run; a val-only margin <~2% is a wash (default to reduced).")

# ---- 3. emit final full-grid scripts -------------------------------------- #
if EMIT and chosen:
    outdir = "ttm-scripts/capacity-select/final"
    os.makedirs(outdir, exist_ok=True)
    for data, cap in chosen.items():
        cdim, bneck = CAPS[cap]
        freq, maxpred = DATASETS.get(data, ("h", 720))
        preds = "96 192 336 720" if maxpred == 720 else "96 192 336"
        fname = ("Jiaolong.sh" if data == "Jiaolong_DSMS" else f"{data}.sh")
        body = f"""if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/capacity-final" ]; then mkdir ./logs/capacity-final; fi

# GENERATED by parse_corners.py --emit. Final full grid for {data}.
# Per-dataset capacity chosen on VALIDATION corners: {cap} (cdim{cdim}/bneck{bneck}).
# Global lr rule: 3e-6 for pred<=192, 1e-6 for pred>=336. ema=0. Single frozen
# PROCEED adapter (no MoE/guard). The INTERIOR cells (pred 192/336, seq 1024) are
# held-out generalization tests of the corner-selected config.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data={data}; freq={freq}; cdim={cdim}; bneck={bneck}
for seq_len in 512 1024 1536
do
for pred_len in {preds}
do
    if [ "$pred_len" -le 192 ]; then lr=0.000003; else lr=0.000001; fi
    fn=logs/capacity-final/${{data}}_${{seq_len}}_${{pred_len}}_{cap}.log
    echo "[capfinal] data=$data seq=$seq_len pred=$pred_len cap={cap} lr=$lr"
    python run.py --model TinyTimeMixer --freeze_online --decoder_mode common_channel \\
    --dataset $data --seq_len $seq_len --pred_len $pred_len --freq $freq \\
    --train_ratio 0.5 --test_ratio 0.45 \\
    --online_method Proceed --concept_dim $cdim --bottleneck_dim $bneck --ema 0 \\
    --batch_size 64 --online_learning_rate $lr --itr 1 >> $fn 2>&1
done
done
"""
        with open(os.path.join(outdir, fname), "w") as f:
            f.write(body)
        print(f"   emitted {outdir}/{fname}")
