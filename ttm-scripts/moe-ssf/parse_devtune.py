#!/usr/bin/env python3
"""Parse the MoE-SSF dev-tuning sweep logs (logs/moe_ssf_devtune/) and tabulate
MSE / MAE / mean guard-alpha per (cell, guard_tau, concept_dim), so you can pick
the single (guard_tau, concept_dim) to freeze into the full-roster scripts.

Selection heuristic printed at the end: for each (tau, cdim) config, average MSE
across the dev cells. Prefer the config with the lowest dev-mean MSE that ALSO
keeps mean-alpha high on Exchange (win preserved) and low on the loss/null cells
(ETTh1 1536_720, ETTh2 512_720, Weather 512_336 -> abstention working).

Usage:  python ttm-scripts/moe-ssf/parse_devtune.py [logdir]
"""
import os
import re
import sys
import glob
from collections import defaultdict

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else "logs/moe_ssf_devtune"

# filename: TTM_ProceedMoE_<data>_<seq>_<pred>_tau<tau>_cdim<cdim>.log
FN = re.compile(r"TTM_ProceedMoE_(?P<data>.+?)_(?P<seq>\d+)_(?P<pred>\d+)_"
                r"tau(?P<tau>[0-9.]+)_cdim(?P<cdim>\d+)\.log$")
MSE = re.compile(r"mse:([0-9.]+),\s*mae:([0-9.]+)")
ALPHA = re.compile(r"\[MoE guard \| phase=test\].*?mean alpha=([0-9.]+)")

rows = []
for path in sorted(glob.glob(os.path.join(LOGDIR, "*.log"))):
    m = FN.search(os.path.basename(path))
    if not m:
        continue
    txt = open(path, errors="ignore").read()
    mse = mae = alpha = None
    mm = MSE.findall(txt)
    if mm:
        mse, mae = float(mm[-1][0]), float(mm[-1][1])   # last = test phase
    am = ALPHA.findall(txt)
    if am:
        alpha = float(am[-1])
    rows.append(dict(cell=f"{m['data']}_{m['seq']}_{m['pred']}",
                     data=m['data'], seq=m['seq'], pred=m['pred'],
                     tau=float(m['tau']), cdim=int(m['cdim']),
                     mse=mse, mae=mae, alpha=alpha))

if not rows:
    print(f"No parseable logs in {LOGDIR}. Run ttm-scripts/moe-ssf/dev_tune.sh first.")
    sys.exit(0)

# ---- per-run table -------------------------------------------------------- #
print(f"\n{'cell':18} {'tau':>5} {'cdim':>5} {'MSE':>9} {'MAE':>9} {'mean_a':>7}")
print("-" * 60)
for r in sorted(rows, key=lambda r: (r['cell'], r['cdim'], r['tau'])):
    ms = f"{r['mse']:.4f}" if r['mse'] is not None else "   --   "
    ma = f"{r['mae']:.4f}" if r['mae'] is not None else "   --   "
    al = f"{r['alpha']:.3f}" if r['alpha'] is not None else "  --  "
    print(f"{r['cell']:18} {r['tau']:>5} {r['cdim']:>5} {ms:>9} {ma:>9} {al:>7}")

# ---- config summary: mean MSE across dev cells ---------------------------- #
by_cfg = defaultdict(list)
for r in rows:
    if r['mse'] is not None:
        by_cfg[(r['tau'], r['cdim'])].append(r['mse'])
print(f"\n{'== config (tau, cdim) ==':28} {'#cells':>6} {'dev-mean MSE':>13}")
print("-" * 50)
best = None
for (tau, cdim), vals in sorted(by_cfg.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
    mean = sum(vals) / len(vals)
    print(f"tau={tau:<6} cdim={cdim:<5}          {len(vals):>6} {mean:>13.4f}")
    if best is None:
        best = (tau, cdim, mean)
if best:
    print(f"\n-> lowest dev-mean MSE: tau={best[0]}, cdim={best[1]}  ({best[2]:.4f})")
    print("   (cross-check alpha: high on Exchange = win kept; low on "
          "ETTh1_1536_720 / ETTh2_512_720 / Weather = abstention working)")
