"""Supervised table: one row per (issue time, horizon).

Everything about the target hour that is *knowable at issue time* is allowed:
calendar, weather (observed in training, forecast at inference), and carbon history
up to the issue time. Seasonal-naive lags are chosen so they never look past the
issue time.
"""
import math

import numpy as np
import pandas as pd

from .config import HORIZONS, TZ_LOCAL

FEATURES = ["h", "ci_t", "ci_mean24", "ci_mean168", "ci_trend6", "snaive_day", "snaive_week",
            "wind", "ghi", "temp", "cloud", "wind_roll6", "hour", "dow", "month"]


def build_frame(df: pd.DataFrame, issue_times, horizons=HORIZONS, with_target=True) -> pd.DataFrame:
    ci = df["ci"]
    roll24 = ci.rolling(24, min_periods=12).mean()
    roll168 = ci.rolling(168, min_periods=84).mean()
    trend6 = ci - ci.shift(6)
    wind_r6 = df["wind"].rolling(6, min_periods=1).mean()

    T = pd.DatetimeIndex(issue_times)
    base = {"ci_t": ci.reindex(T).to_numpy(), "ci_mean24": roll24.reindex(T).to_numpy(),
            "ci_mean168": roll168.reindex(T).to_numpy(), "ci_trend6": trend6.reindex(T).to_numpy()}
    out = []
    for h in horizons:
        tgt = T + pd.Timedelta(hours=h)
        lag_day = 24 * math.ceil(h / 24)       # most recent same-hour value that is already observed
        loc = tgt.tz_convert(TZ_LOCAL)
        f = pd.DataFrame({
            "issue": T, "target": tgt, "h": h, **base,
            "snaive_day": ci.reindex(tgt - pd.Timedelta(hours=lag_day)).to_numpy(),
            "snaive_week": ci.reindex(tgt - pd.Timedelta(hours=168)).to_numpy(),
            "wind": df["wind"].reindex(tgt).to_numpy(), "ghi": df["ghi"].reindex(tgt).to_numpy(),
            "temp": df["temp"].reindex(tgt).to_numpy(), "cloud": df["cloud"].reindex(tgt).to_numpy(),
            "wind_roll6": wind_r6.reindex(tgt).to_numpy(),
            "hour": loc.hour.to_numpy(), "dow": loc.day_of_week.to_numpy(), "month": loc.month.to_numpy(),
        })
        if with_target:
            f["y"] = ci.reindex(tgt).to_numpy()
        out.append(f)
    return pd.concat(out, ignore_index=True)


def training_issue_times(df, cutoff=None, step=3, max_days=730):
    """Issue times whose *targets* are all <= cutoff (no peeking past the cutoff)."""
    cutoff = cutoff if cutoff is not None else df["ci"].last_valid_index()
    first = max(df.index.min() + pd.Timedelta(hours=170), cutoff - pd.Timedelta(days=max_days))
    last_issue = cutoff - pd.Timedelta(hours=max(HORIZONS))
    if first.ceil("h") > last_issue:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(first.ceil("h"), last_issue, freq=f"{step}h")
