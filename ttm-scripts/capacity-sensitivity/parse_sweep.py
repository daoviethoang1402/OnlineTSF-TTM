#!/usr/bin/env python
"""
Parse the capacity-sensitivity sweep (Task 1).

Each log holds ITR completed runs, each ending in a line:  mse:<x>, mae:<y>
We average over seeds -> mean +/- std per (dataset, seq, pred, cdim, bneck).

Usage:
    python ttm-scripts/capacity-sensitivity/parse_sweep.py
    python ttm-scripts/capacity-sensitivity/parse_sweep.py --metric mae --csv out.csv
"""
import re, os, glob, argparse
from collections import defaultdict
import numpy as np

LOGDIR = "logs/capacity-sensitivity"
FN_RE = re.compile(
    r"^(?P<data>.+)_(?P<seq>\d+)_(?P<pred>\d+)_cd(?P<cdim>\d+)_bn(?P<bneck>\d+)_lr(?P<lr>[0-9.eE+-]+)\.log$"
)
MSE_RE = re.compile(r"mse:\s*([0-9.eE+-]+)\s*,\s*mae:\s*([0-9.eE+-]+)")


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(LOGDIR, "*.log"))):
        m = FN_RE.match(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            pairs = MSE_RE.findall(f.read())
        if not pairs:
            continue
        mses = np.array([float(a) for a, _ in pairs])
        maes = np.array([float(b) for _, b in pairs])
        d = m.groupdict()
        rows.append(dict(
            data=d["data"], seq=int(d["seq"]), pred=int(d["pred"]),
            cdim=int(d["cdim"]), bneck=int(d["bneck"]), lr=d["lr"],
            n=len(mses),
            mse_mean=mses.mean(), mse_std=mses.std(ddof=0),
            mae_mean=maes.mean(), mae_std=maes.std(ddof=0),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["mse", "mae"], default="mse")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = load()
    if not rows:
        print(f"No parsable logs in {LOGDIR}/ yet.")
        return
    mean_k, std_k = f"{args.metric}_mean", f"{args.metric}_std"

    # capacity columns in ascending "size" order
    caps = sorted({(r["cdim"], r["bneck"]) for r in rows})
    cap_lbl = [f"cd{c}/bn{b}" for c, b in caps]

    for data in sorted({r["data"] for r in rows}):
        print(f"\n===== {data}  ({args.metric.upper()} mean +/- std over seeds) =====")
        header = f"{'seq_pred':>10} | " + " | ".join(f"{l:>14}" for l in cap_lbl)
        print(header)
        cells = defaultdict(dict)
        for r in rows:
            if r["data"] != data:
                continue
            cells[(r["seq"], r["pred"])][(r["cdim"], r["bneck"])] = r
        for (seq, pred) in sorted(cells):
            line = f"{seq}_{pred:>4} | "
            best = min((cells[(seq, pred)][c][mean_k] for c in caps
                        if c in cells[(seq, pred)]), default=None)
            parts = []
            for c in caps:
                r = cells[(seq, pred)].get(c)
                if r is None:
                    parts.append(f"{'--':>14}")
                else:
                    star = "*" if best is not None and abs(r[mean_k]-best) < 1e-9 else " "
                    parts.append(f"{r[mean_k]:.4f}±{r[std_k]:.3f}{star}")
            print(line + " | ".join(parts))

        # per-capacity mean over all cells (the sensitivity curve)
        print(f"  --- mean over all cells (sensitivity curve) ---")
        for c, lbl in zip(caps, cap_lbl):
            vals = [r[mean_k] for r in rows if r["data"] == data and (r["cdim"], r["bneck"]) == c]
            if vals:
                print(f"    {lbl:>14}: {np.mean(vals):.4f}   (n_cells={len(vals)})")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
