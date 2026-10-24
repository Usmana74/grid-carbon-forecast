"""Rolling-origin backtest: train only on the past, test on the following window, repeat.

No random splits: with time series they leak (neighbouring hours are near-duplicates).
The leak guard below asserts that every training *target* lies at or before the fold start.
"""
import numpy as np
import pandas as pd

from .config import HORIZONS
from .calibrate import fit_calibrated
from .features import build_frame, training_issue_times
from .model import QuantileGBM

BASELINES = {"persistence": "ci_t", "seasonal_naive_day": "snaive_day", "seasonal_naive_week": "snaive_week"}


def _metrics(pred, y):
    e = pred - y
    return {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}


def summarize(res: pd.DataFrame) -> dict:
    r = res.dropna(subset=["p10", "p50", "p90", "ci_t", "snaive_day", "snaive_week", "y"])
    models = {"model": _metrics(r["p50"], r["y"])}
    for name, col in BASELINES.items():
        models[name] = _metrics(r[col], r["y"])
    best_base = min(v["mae"] for k, v in models.items() if k != "model")
    by_h = {}
    for lo, hi in [(1, 6), (7, 12), (13, 24), (25, 48)]:
        s = r[(r["h"] >= lo) & (r["h"] <= hi)]
        by_h[f"{lo}-{hi}h"] = {
            "model_mae": float((s["p50"] - s["y"]).abs().mean()),
            "persistence_mae": float((s["ci_t"] - s["y"]).abs().mean()),
            "seasonal_naive_day_mae": float((s["snaive_day"] - s["y"]).abs().mean())}
    return {
        "n": int(len(r)), "models": models,
        "skill_vs_best_baseline": float(1 - models["model"]["mae"] / best_base),
        "skill_vs_seasonal_naive_day": float(1 - models["model"]["mae"] / models["seasonal_naive_day"]["mae"]),
        "coverage_80": float(((r["y"] >= r["p10"]) & (r["y"] <= r["p90"])).mean()),
        "mean_interval_width": float((r["p90"] - r["p10"]).mean()),
        "by_horizon": by_h,
    }


def run_backtest(df, n_folds=4, test_days=14, train_step=3, test_step=6, max_train_days=730, model_kw=None,
                 calibrate=True, calib_days=60):
    last = df["ci"].last_valid_index().floor("D")
    H = max(HORIZONS)
    parts, folds = [], []
    for k in range(n_folds):
        fold_end = last - pd.Timedelta(days=test_days * (n_folds - 1 - k))
        start = fold_end - pd.Timedelta(days=test_days)
        if calibrate:
            model = fit_calibrated(df, start, calib_days=calib_days, train_step=train_step,
                                   max_days=max_train_days, model_kw=model_kw)
            n_train, widths = model.train_rows, model.widths
        else:
            tr = build_frame(df, training_issue_times(df, cutoff=start, step=train_step, max_days=max_train_days))
            assert tr["target"].max() <= start, "LEAK: training target after fold start"
            model = QuantileGBM(**(model_kw or {})).fit(tr)
            n_train, widths = int(len(tr)), None
        test_issues = pd.date_range(start, fold_end - pd.Timedelta(hours=H), freq=f"{test_step}h")
        te = build_frame(df, test_issues)
        te = pd.concat([te, model.predict(te)], axis=1)
        te["fold"] = k
        parts.append(te)
        folds.append({"fold": k, "train_until": start.isoformat(), "test_from": start.isoformat(),
                      "test_to": fold_end.isoformat(), "train_rows": n_train, "interval_widths": widths})
    res = pd.concat(parts, ignore_index=True)
    out = summarize(res)
    out["folds"] = folds
    out["per_fold_mae"] = {int(k): float((g["p50"] - g["y"]).abs().mean()) for k, g in res.dropna(subset=["y"]).groupby("fold")}
    return out
