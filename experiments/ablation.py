"""Weather-input ablation: how much of the backtest skill comes from (perfect) weather?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import gridcast.features as F
from gridcast.backtest import run_backtest
from gridcast.cli import load_dataset

df = load_dataset()
WX = ["wind", "ghi", "temp", "cloud", "wind_roll6"]
FULL = list(F.FEATURES)


def run(name, frame, feats):
    F.FEATURES[:] = feats  # model.py and calibrate.py share this same list object
    r = run_backtest(frame, n_folds=4)
    print(f"{name:34s} MAE {r['models']['model']['mae']:6.1f} | skill {r['skill_vs_best_baseline']:+.1%}"
          f" | cov80 {r['coverage_80']:.1%}", flush=True)


run("A: perfect weather (as before)", df, FULL)
run("B: no weather features", df, [f for f in FULL if f not in WX])

rng = np.random.default_rng(0)
noisy = df.copy()
for c in ["wind", "ghi", "temp", "cloud"]:
    noisy[c] = noisy[c] + rng.normal(0, 0.5 * df[c].std(), len(df))
noisy["ghi"] = noisy["ghi"].clip(lower=0)
noisy["cloud"] = noisy["cloud"].clip(0, 100)
run("C: noisy weather (sd = 0.5 x std)", noisy, FULL)
