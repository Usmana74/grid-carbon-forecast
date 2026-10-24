import json

import numpy as np
import pandas as pd
import pytest

from gridcast.backtest import run_backtest
from gridcast.calibrate import fit_calibrated
from gridcast.carbon import intensity_from_mix
from gridcast.features import FEATURES, build_frame, training_issue_times
from gridcast.model import QuantileGBM
from gridcast.publish import cleanest_window, high_carbon_alert, make_forecast, score_all, write_forecast
from gridcast.sources import parse_generation, parse_weather
from gridcast.synthetic import synthetic_dataset


# ---------- carbon ----------
def test_intensity_weighted_and_ignores_non_generation():
    idx = pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC")
    mix = pd.DataFrame({"Fossil brown coal / lignite": [100.0, 0.0], "Wind onshore": [0.0, 100.0],
                        "Load": [999.0, 999.0], "Residual load": [5.0, 5.0]}, index=idx)
    ci = intensity_from_mix(mix)
    assert ci.iloc[0] == pytest.approx(1050) and ci.iloc[1] == pytest.approx(11)


def test_coal_derived_gas_not_matched_as_gas():
    idx = pd.date_range("2025-01-01", periods=1, freq="1h", tz="UTC")
    ci = intensity_from_mix(pd.DataFrame({"Fossil coal-derived gas": [10.0]}, index=idx))
    assert ci.iloc[0] == pytest.approx(700)


# ---------- parsers (payload shapes from the API docs) ----------
def test_parse_generation_resamples_to_hourly():
    t0 = 1735689600  # 2025-01-01 00:00 UTC
    js = {"unix_seconds": [t0 + 900 * i for i in range(8)],
          "production_types": [{"name": "Wind onshore", "data": [1, 2, 3, 4, 5, 6, 7, None]},
                               {"name": "Solar", "data": [0] * 8}]}
    df = parse_generation(js)
    assert len(df) == 2 and df["Wind onshore"].iloc[0] == pytest.approx(2.5)


def test_parse_weather_averages_locations():
    hourly = lambda v: {"hourly": {"time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                                   "wind_speed_100m": [v, v], "shortwave_radiation": [0, 0],
                                   "temperature_2m": [1, 1], "cloud_cover": [50, 50]}}
    df = parse_weather([hourly(4.0), hourly(8.0)])
    assert df["wind"].iloc[0] == pytest.approx(6.0) and str(df.index.tz) == "UTC"


# ---------- features: no leakage ----------
@pytest.fixture(scope="module")
def df():
    return synthetic_dataset(days=150, seed=1)


def test_features_ignore_future_carbon(df):
    T = df.index[2000]
    a = build_frame(df, [T])
    df2 = df.copy()
    df2.loc[df2.index > T, "ci"] = 1e6                      # corrupt everything after the issue time
    b = build_frame(df2, [T])
    cols = [c for c in FEATURES if c not in ("wind", "ghi", "temp", "cloud", "wind_roll6")]
    pd.testing.assert_frame_equal(a[cols], b[cols])         # all carbon-derived features unchanged


def test_seasonal_lags_never_look_past_issue_time():
    idx = pd.date_range("2025-01-01", periods=24 * 20, freq="1h", tz="UTC")
    d = pd.DataFrame({"ci": np.arange(len(idx), dtype=float), "wind": 1.0, "ghi": 1.0, "temp": 1.0, "cloud": 1.0}, index=idx)
    f = build_frame(d, [idx[24 * 10]])
    issue_pos = 24 * 10
    assert (f["snaive_day"] <= issue_pos).all() and (f["snaive_week"] <= issue_pos).all()


def test_training_issue_times_respect_cutoff(df):
    cutoff = df.index[3000]
    issues = training_issue_times(df, cutoff=cutoff)
    assert (issues + pd.Timedelta(hours=48)).max() <= cutoff


# ---------- model / backtest ----------
def test_backtest_smoke(df):
    res = run_backtest(df, n_folds=2, test_days=7, model_kw={"max_iter": 30})
    assert res["n"] > 500 and 0 <= res["coverage_80"] <= 1
    assert set(res["models"]) == {"model", "persistence", "seasonal_naive_day", "seasonal_naive_week"}


def test_quantiles_do_not_cross(df):
    issues = training_issue_times(df, cutoff=df.index[3000])
    tr = build_frame(df, issues)
    m = QuantileGBM(max_iter=30).fit(tr)
    p = m.predict(tr.iloc[:500])
    assert (p["p10"] <= p["p50"]).all() and (p["p50"] <= p["p90"]).all()


# ---------- publishing ----------
def test_window_and_alert():
    t = pd.date_range("2025-01-01", periods=12, freq="1h", tz="UTC")
    p50 = [300, 300, 100, 90, 80, 300, 300, 300, 300, 300, 300, 300]
    w = cleanest_window(t, p50, hours=3)
    assert w["mean"] == pytest.approx(90.0) and w["start"].startswith("2025-01-01T02")
    a = high_carbon_alert(t, p50, threshold=250, min_hours=6)
    assert a["active"] and a["hours"] == 7
    assert not high_carbon_alert(t, p50, threshold=250, min_hours=8)["active"]


def test_forecasts_are_frozen_and_scored(df, tmp_path):
    cut = df.index[3000]
    model = QuantileGBM(max_iter=30).fit(build_frame(df, training_issue_times(df, cutoff=cut)))
    fc = make_forecast(df, model, cut, "synthetic")
    assert write_forecast(fc, tmp_path) is True
    fc2 = dict(fc, p50=[0.0] * 48)
    assert write_forecast(fc2, tmp_path) is False            # no silent rewriting of history
    stored = json.loads(next((tmp_path / "forecasts").glob("*.json")).read_text())
    assert stored["p50"] == fc["p50"]
    board = score_all(df, tmp_path, "synthetic")
    assert board["summary"]["n_forecasts"] == 1 and board["summary"]["n_points"] == 48


def test_calibration_widths_and_ordering(df):
    cut = df.index[3200]
    m = fit_calibrated(df, cut, calib_days=30, model_kw={"max_iter": 30})
    assert set(m.widths) == {"1-6", "7-12", "13-24", "25-48"}
    frame = build_frame(df, [cut], with_target=False)
    p = m.predict(frame)
    assert (p["p10"] <= p["p50"]).all() and (p["p50"] <= p["p90"]).all() and (p["p10"] >= 0).all()


def test_backtest_without_calibration_still_works(df):
    res = run_backtest(df, n_folds=2, test_days=7, model_kw={"max_iter": 30}, calibrate=False)
    assert res["n"] > 500
