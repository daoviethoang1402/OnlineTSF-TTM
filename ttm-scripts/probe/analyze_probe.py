#!/usr/bin/env python3
"""Parse Phase 0 (headroom) and Phase 1 (forgetting) probe logs.

Usage:
    python3 ttm-scripts/probe/analyze_probe.py            # Phase 0 headroom table
    python3 ttm-scripts/probe/analyze_probe.py --forget   # Phase 1 forgetting curves

Phase 0 verdict: a dataset has real HEADROOM if the best adaptation variant beats
few-shot by clearly more than the ~1% seen on ETT/Weather. Phase 1 verdict: if
naive-ft's MSE climbs with online LR (forgetting) while PROCEED-ft stays flat,
the PROCEED>=naive advantage is forgetting mitigation.
"""
import argparse, glob, os, re

def final_mse(path):
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return None
    m = re.findall(r"mse:([0-9.]+),\s*mae:([0-9.]+)", txt)
    return float(m[-1][0]) if m else None

def phase0(logdir):
    variants = ["fewshot", "naiveft", "proceedfrz", "proceedft"]
    cells = {}
    for f in glob.glob(os.path.join(logdir, "*.log")):
        b = os.path.basename(f)[:-4]
        for v in variants:
            m = re.match(rf"{v}_(.+)_(\d+)_(\d+)$", b)
            if m:
                cells.setdefault((m.group(1), int(m.group(2)), int(m.group(3))), {})[v] = final_mse(f)
                break
    print(f"{'dataset':10} {'seq':>5} {'pred':>5} | {'fewshot':>8} {'naiveft':>8} {'procFRZ':>8} {'procFT':>8} | "
          f"{'bestAdapt vs fewshot':>20}")
    for k in sorted(cells):
        d = cells[k]
        fs = d.get("fewshot")
        adapts = [d[v] for v in ("naiveft", "proceedfrz", "proceedft") if d.get(v) is not None]
        cell = f"{k[0]:10} {k[1]:>5} {k[2]:>5} | "
        cell += " ".join(f"{d.get(v):8.4f}" if d.get(v) is not None else f"{'--':>8}" for v in variants)
        if fs and adapts:
            best = min(adapts); pct = 100 * (best - fs) / fs
            verdict = "  <-- HEADROOM" if pct < -1.5 else ""
            cell += f" | {pct:+7.1f}% (best {best:.4f}){verdict}"
        print(cell)
    print("\nHEADROOM if bestAdapt beats few-shot by > ~1.5% (more than the ETT/Weather ceiling).")

def phase1(logdir):
    # naiveft_lr0001_<data>_512_96 / proceedft_lr0001_...
    pat = re.compile(r"(naiveft|proceedft)_lr([0-9]+)_(.+)_(\d+)_(\d+)$")
    data = {}   # (dataset)->{method-> {lr: mse}}, + fewshot
    fewshot = {}
    for f in glob.glob(os.path.join(logdir, "*.log")):
        b = os.path.basename(f)[:-4]
        fm = re.match(r"fewshot_(.+)_(\d+)_(\d+)$", b)
        if fm:
            fewshot[fm.group(1)] = final_mse(f); continue
        m = pat.match(b)
        if m:
            method, lrtag, ds = m.group(1), m.group(2), m.group(3)
            lr = {"00001": 1e-4, "0001": 1e-3, "001": 1e-2}.get(lrtag, lrtag)
            data.setdefault(ds, {}).setdefault(method, {})[lr] = final_mse(f)
    for ds in sorted(data):
        fs = fewshot.get(ds)
        print(f"\n=== {ds}  (few-shot MSE = {fs}) ===")
        print(f"{'online_lr':>10} | {'naiveft':>9} {'Δvs fewshot':>12} | {'proceedft':>10} {'Δvs fewshot':>12}")
        lrs = sorted({lr for meth in data[ds].values() for lr in meth})
        for lr in lrs:
            nv = data[ds].get("naiveft", {}).get(lr)
            pf = data[ds].get("proceedft", {}).get(lr)
            nvd = f"{100*(nv-fs)/fs:+.1f}%" if (nv and fs) else "--"
            pfd = f"{100*(pf-fs)/fs:+.1f}%" if (pf and fs) else "--"
            print(f"{lr:>10} | {nv if nv else '--':>9} {nvd:>12} | {pf if pf else '--':>10} {pfd:>12}")
    print("\nFORGETTING signature: naiveft MSE rises with LR (backbone overwritten) while "
          "proceedft stays ~flat (frozen backbone). Divergence => PROCEED mitigates forgetting.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget", action="store_true", help="Phase 1 forgetting curves (logs/probe_forget)")
    ap.add_argument("--logdir", default=None)
    a = ap.parse_args()
    if a.forget:
        phase1(a.logdir or "logs/probe_forget")
    else:
        phase0(a.logdir or "logs/probe")
