#!/usr/bin/env python3
"""Clean missing values that cause NaN training loss on the new headroom datasets.

- AirQuality (UCI): -200 is the missing-value sentinel -> replace with NaN.
- BeijingAQ / AirQuality: NaN gaps in the sensor columns.
Fix: time-interpolate numeric columns, then edge-fill (ffill/bfill). Writes the
files IN PLACE (originals preserved under ~/Desktop/forecasting-datasets).
Energy is already clean and is left untouched.
"""
import glob, sys
import numpy as np
import pandas as pd

FILES = glob.glob("dataset/beijing-air-quality/PRSA_*.csv") + ["dataset/air_quality.csv"]

for f in FILES:
    df = pd.read_csv(f)
    df = df.dropna(subset=["date"]).reset_index(drop=True)       # drop junk date-less rows (UCI trailing blanks)
    num = df.select_dtypes(include=[np.number]).columns
    before_nan = int(df.isna().sum().sum())
    before_sentinel = int((df[num] == -200).sum().sum())
    df[num] = df[num].replace(-200, np.nan)                       # UCI missing sentinel
    df[num] = (df[num].interpolate(method="linear", limit_direction="both")
                       .ffill().bfill())                          # fill interior + edges
    after_nan = int(df.isna().sum().sum())
    df.to_csv(f, index=False)
    print(f"{f}: NaN {before_nan}->{after_nan}, -200 sentinels replaced {before_sentinel} "
          f"[{'OK' if after_nan == 0 else 'STILL HAS NaN'}]")
