"""Issue forecasts, freeze them to disk, and score them against what actually happened."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HORIZONS, MODEL_VERSION, UNIT
from .features import build_frame


def iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _lst(a):
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 1) for x in a]


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def cleanest_window(times, p50, hours=3):
    s = pd.Series(np.asarray(p50, dtype=float), index=pd.DatetimeIndex(times))
    roll = s.rolling(hours).mean().dropna()
    end = roll.idxmin()
    start = end - pd.Timedelta(hours=hours - 1)
    return {"start": iso(start), "end": iso(end + pd.Timedelta(hours=1)), "mean": round(float(roll.min()), 1), "hours": hours}


def high_carbon_alert(times, p50, threshold, min_hours=6):
    above = np.asarray(p50, dtype=float) >= threshold
    best, run_start = None, None
    for i, a in enumerate(list(above) + [False]):
        if a and run_start is None:
            run_start = i
        if not a and run_start is not None:
            if i - run_start >= min_hours and (best is None or i - run_start > best[1] - best[0]):
                best = (run_start, i)
            run_start = None
    if best is None:
        return {"active": False, "threshold": round(float(threshold), 1)}
    t = pd.DatetimeIndex(times)
    return {"active": True, "start": iso(t[best[0]]), "end": iso(t[best[1] - 1] + pd.Timedelta(hours=1)),
            "hours": int(best[1] - best[0]), "threshold": round(float(threshold), 1)}


def make_forecast(df, model, issue_time, source) -> dict:
    frame = build_frame(df, [issue_time], with_target=False)
    pred = model.predict(frame)
    times = frame["target"]
    thr = float(df["ci"].dropna().iloc[-24 * 365:].quantile(0.9))
    return {
        "issue_time": iso(issue_time), "source": source, "model_version": MODEL_VERSION, "unit": UNIT,
        "times": [iso(t) for t in times],
        "p10": _lst(pred["p10"]), "p50": _lst(pred["p50"]), "p90": _lst(pred["p90"]),
        "baseline": _lst(frame["snaive_day"]),
        "last_observed": {"time": iso(issue_time), "ci": round(float(df["ci"].loc[issue_time]), 1)},
        "cleanest_window": cleanest_window(times, pred["p50"]),
        "alert": high_carbon_alert(times, pred["p50"], thr),
        "interval_calibration": getattr(model, "widths", None),
    }


def write_forecast(fc: dict, out_dir: Path, overwrite=False) -> bool:
    """Forecast files are frozen: an already-issued forecast is never rewritten."""
    stamp = fc["issue_time"].replace("-", "").replace(":", "")[:11]  # e.g. 20261023T14
    path = Path(out_dir) / "forecasts" / f"{stamp}.json"
    if path.exists() and not overwrite:
        return False
    write_json(fc, path)
    write_json(fc, Path(out_dir) / "latest.json")
    return True


def score_all(df, out_dir: Path, source: str, min_points=12) -> dict:
    rows = []
    for f in sorted((Path(out_dir) / "forecasts").glob("*.json")):
        fc = json.loads(f.read_text(encoding="utf-8"))
        t = pd.to_datetime(fc["times"], utc=True)
        y = df["ci"].reindex(t).to_numpy()
        p10, p50, p90 = (np.array([np.nan if v is None else v for v in fc[k]], dtype=float) for k in ("p10", "p50", "p90"))
        base = np.array([np.nan if v is None else v for v in fc["baseline"]], dtype=float)
        m = ~np.isnan(y) & ~np.isnan(p50) & ~np.isnan(base)
        if m.sum() < min_points:
            continue
        rows.append({"issue_time": fc["issue_time"], "n": int(m.sum()),
                     "mae_model": float(np.abs(p50[m] - y[m]).mean()),
                     "mae_baseline": float(np.abs(base[m] - y[m]).mean()),
                     "coverage": float(((y[m] >= p10[m]) & (y[m] <= p90[m])).mean())})
    board = {"updated": iso(pd.Timestamp.now(tz="UTC")), "source": source, "summary": None, "forecasts": rows[-120:]}
    if rows:
        n = np.array([r["n"] for r in rows], dtype=float)
        mm = float(np.average([r["mae_model"] for r in rows], weights=n))
        mb = float(np.average([r["mae_baseline"] for r in rows], weights=n))
        board["summary"] = {"n_forecasts": len(rows), "n_points": int(n.sum()), "mae_model": round(mm, 2),
                            "mae_baseline": round(mb, 2), "skill": round(1 - mm / mb, 4),
                            "coverage_80": round(float(np.average([r["coverage"] for r in rows], weights=n)), 4)}
    write_json(board, Path(out_dir) / "scoreboard.json")
    return board
