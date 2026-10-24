"""Split-conformal calibration of the 80% interval (conformalized quantile regression).

Problem it fixes: raw quantile models are over-confident (the backtest showed ~68% of
outcomes inside a nominal 80% band). Procedure:
  1. train a model on data up to (cutoff - calib_days),
  2. measure on the following calib_days how far outcomes fall outside [p10, p90],
  3. widen (or shrink) the band by that amount, separately per horizon bucket,
  4. apply those widths to a final model trained on ALL data up to cutoff.
Everything used for calibration lies strictly before the cutoff. Guarantees are approximate
(consecutive hours are correlated and the data is not exchangeable), so coverage is still
reported, never assumed.
"""
import numpy as np
import pandas as pd

from .config import HORIZONS
from .features import build_frame, training_issue_times
from .model import QuantileGBM

BUCKETS = [(1, 6), (7, 12), (13, 24), (25, 48)]


def _bucket(h: pd.Series) -> pd.Series:
    out = pd.Series("", index=h.index, dtype=object)
    for lo, hi in BUCKETS:
        out[(h >= lo) & (h <= hi)] = f"{lo}-{hi}"
    return out


class CalibratedModel:
    def __init__(self, model: QuantileGBM, widths: dict, train_rows: int = 0):
        self.model, self.widths, self.train_rows = model, widths, train_rows

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        p = self.model.predict(frame)
        w = _bucket(frame["h"]).map(self.widths).astype(float).to_numpy()
        p["p10"] = np.minimum(p["p10"] - w, p["p50"]).clip(lower=0)
        p["p90"] = np.maximum(p["p90"] + w, p["p50"])
        return p


def fit_calibrated(df, cutoff, calib_days=60, alpha=0.2, train_step=3, max_days=730, model_kw=None):
    kw = model_kw or {}
    cal_start = cutoff - pd.Timedelta(days=calib_days)

    tr_a = build_frame(df, training_issue_times(df, cutoff=cal_start, step=train_step, max_days=max_days))
    assert tr_a["target"].max() <= cal_start, "LEAK: stage-1 training target after calibration start"
    base = QuantileGBM(**kw).fit(tr_a)

    issues = pd.date_range(cal_start.ceil("h"), cutoff - pd.Timedelta(hours=max(HORIZONS)), freq=f"{train_step}h")
    cal = build_frame(df, issues)
    assert cal["target"].max() <= cutoff, "LEAK: calibration target after cutoff"
    cal = cal[cal["y"].notna()].reset_index(drop=True)
    pred = base.predict(cal)
    score = np.maximum(pred["p10"] - cal["y"], cal["y"] - pred["p90"])   # >0 means outside the band
    buckets = _bucket(cal["h"])
    widths = {}
    for lo, hi in BUCKETS:
        s = score[buckets == f"{lo}-{hi}"].to_numpy()
        if len(s) < 50:
            widths[f"{lo}-{hi}"] = 0.0
            continue
        level = min(1.0, np.ceil((len(s) + 1) * (1 - alpha)) / len(s))
        widths[f"{lo}-{hi}"] = float(np.quantile(s, level, method="higher"))

    tr = build_frame(df, training_issue_times(df, cutoff=cutoff, step=train_step, max_days=max_days))
    assert tr["target"].max() <= cutoff, "LEAK: final training target after cutoff"
    final = QuantileGBM(**kw).fit(tr)
    return CalibratedModel(final, widths, train_rows=int(len(tr)))
