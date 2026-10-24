"""Command line: python -m gridcast <check|ingest|backtest|daily|score|demo>"""
import argparse
import warnings

import pandas as pd

from . import config as C
from .backtest import run_backtest
from .calibrate import fit_calibrated
from .carbon import intensity_from_mix
from .features import build_frame, training_issue_times
from .model import QuantileGBM
from .publish import make_forecast, score_all, write_forecast, write_json
from .sources import WX_COLS, fetch_generation, fetch_weather_archive, fetch_weather_recent
from .synthetic import synthetic_dataset

DATASET = C.DATA / "dataset.csv.gz"
REAL_SOURCE = "energy-charts+open-meteo"


def now_utc():
    return pd.Timestamp.now(tz="UTC")


def load_dataset(path=DATASET) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def save_dataset(df, path=DATASET):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, compression="gzip")


# ---------------------------------------------------------------- commands
def cmd_check(_):
    """Verify the live APIs return what the parsers expect (run this first)."""
    end = now_utc()
    start = end - pd.Timedelta(days=2)
    print("== Energy-Charts generation ==")
    gen = fetch_generation(start, end)
    print("columns:", list(gen.columns))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ci = intensity_from_mix(gen)
    for x in w:
        print("WARNING:", x.message)
    print(ci.dropna().tail(5))
    print("\n== Open-Meteo weather (recent + forecast) ==")
    wx = fetch_weather_recent(past_days=2, forecast_days=2)
    print(wx.tail(5))
    print("\nOK: both sources parsed. If a column list or warning looks wrong, adjust gridcast/carbon.py.")


def ingest(start=None):
    now = now_utc().floor("h")
    existing = load_dataset() if DATASET.exists() else None
    if start:
        start = pd.Timestamp(start, tz="UTC")
    elif existing is not None and existing["ci"].last_valid_index() is not None:
        start = existing["ci"].last_valid_index() - pd.Timedelta(days=3)
    else:
        start = now - pd.Timedelta(days=730)
    start = start.floor("D")
    print(f"ingesting {start.date()} -> {now.date()}")

    gen = fetch_generation(start, now)
    ci = intensity_from_mix(gen)

    cutoff = now.floor("D") - pd.Timedelta(days=7)      # archive lags ~5 days
    recent_from = max(start, cutoff - pd.Timedelta(days=1))
    past_days = int(min(92, (now - recent_from).days + 2))
    wx_recent = fetch_weather_recent(past_days=past_days, forecast_days=1)
    if start <= cutoff:
        wx = fetch_weather_archive(start, cutoff).combine_first(wx_recent)
    else:
        wx = wx_recent
    wx = wx.loc[:now]

    new = pd.concat([ci, wx[WX_COLS]], axis=1).loc[start:now]
    merged = new if existing is None else new.combine_first(existing)
    merged = merged.reindex(pd.date_range(merged.index.min(), merged.index.max(), freq="1h"))
    merged = merged.iloc[-24 * 800:]
    save_dataset(merged)
    print(f"dataset rows: {len(merged)}, last ci hour: {merged['ci'].last_valid_index()}")


def cmd_ingest(args):
    ingest(args.start)


def cmd_backtest(args):
    df = load_dataset()
    res = run_backtest(df, n_folds=args.folds)
    res["source"] = REAL_SOURCE
    write_json(res, C.SITE_DATA / "backtest.json")
    _print_backtest(res)


def _print_backtest(res):
    m = res["models"]
    print(f"\nBacktest on {res['n']} forecasts-hours (rolling origin)")
    for k, v in m.items():
        print(f"  {k:22s} MAE {v['mae']:7.2f}  RMSE {v['rmse']:7.2f}  bias {v['bias']:+7.2f}")
    print(f"  skill vs best baseline: {res['skill_vs_best_baseline']:+.1%} | "
          f"80% interval coverage: {res['coverage_80']:.1%}")


def _train(df, cutoff=None):
    cutoff = cutoff if cutoff is not None else df["ci"].last_valid_index()
    return fit_calibrated(df, cutoff)


def cmd_daily(_):
    ingest()
    df = load_dataset()
    last = df["ci"].last_valid_index()
    wx_f = fetch_weather_recent(past_days=3, forecast_days=4)
    ext = df.reindex(df.index.union(wx_f.index))
    for c in WX_COLS:
        ext[c] = ext[c].combine_first(wx_f[c])
    model = _train(df)
    fc = make_forecast(ext, model, last, REAL_SOURCE)
    if write_forecast(fc, C.SITE_DATA):
        print("issued forecast for", fc["issue_time"])
    else:
        print("forecast for", fc["issue_time"], "already issued; history is frozen")
    board = score_all(df, C.SITE_DATA, REAL_SOURCE)
    print("scoreboard:", board["summary"])


def cmd_score(_):
    df = load_dataset()
    print(score_all(df, C.SITE_DATA, REAL_SOURCE)["summary"])


def cmd_demo(args):
    """Offline end-to-end run on SYNTHETIC data -> docs/demo/ (open docs/index.html?demo)."""
    out = C.DOCS / "demo"
    full = synthetic_dataset(days=args.days + 3, seed=0)
    df = full.copy()
    df.loc[df.index[-72:], "ci"] = float("nan")          # pretend the last 3 days are the future
    last = df["ci"].last_valid_index()
    print("backtest (synthetic) ...")
    bt = run_backtest(df, n_folds=3)
    bt["source"] = "synthetic"
    write_json(bt, out / "backtest.json")
    _print_backtest(bt)

    day0 = last.floor("D") - pd.Timedelta(days=9)
    frozen = _train(df, cutoff=day0)                      # trained once, before the 'live' period
    for i in range(7):
        issue = day0 + pd.Timedelta(days=i)
        write_forecast(make_forecast(full.combine_first(df), frozen, issue, "synthetic"), out, overwrite=True)
    score_all(df, out, "synthetic")
    model = _train(df)
    write_forecast(make_forecast(full, model, last, "synthetic"), out, overwrite=True)
    print("demo written to", out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gridcast")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    s = sub.add_parser("ingest"); s.add_argument("--start", help="YYYY-MM-DD (default: incremental / 2y)"); s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("backtest"); s.add_argument("--folds", type=int, default=4); s.set_defaults(fn=cmd_backtest)
    sub.add_parser("daily").set_defaults(fn=cmd_daily)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    s = sub.add_parser("demo"); s.add_argument("--days", type=int, default=540); s.set_defaults(fn=cmd_demo)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
