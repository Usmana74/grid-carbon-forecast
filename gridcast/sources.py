"""Public data sources: Energy-Charts (Fraunhofer ISE) and Open-Meteo.

NOTE: response shapes were written from the public API docs and are covered by
offline parser tests, but you should run `python -m gridcast check` once to verify
them against the live APIs (see README).
"""
import time

import pandas as pd
import requests

from .config import WEATHER_POINTS

EC_URL = "https://api.energy-charts.info/public_power"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
WX_VARS = {"wind_speed_100m": "wind", "shortwave_radiation": "ghi",
           "temperature_2m": "temp", "cloud_cover": "cloud"}
WX_COLS = list(WX_VARS.values())


def _get(url, params, retries=5, timeout=90):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": "gridcast/0.1 (open research project)"})
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"{url} -> HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(15 * (i + 1) if r.status_code == 429 else 2 ** i)
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"request failed after {retries} tries: {last}")


# ---------------- generation (Energy-Charts) ----------------
def parse_generation(js) -> pd.DataFrame:
    idx = pd.to_datetime(js["unix_seconds"], unit="s", utc=True)
    cols = {pt["name"]: pt["data"] for pt in js["production_types"]}
    df = pd.DataFrame(cols, index=idx).apply(pd.to_numeric, errors="coerce")
    return df.resample("1h").mean()  # source is 15-min or hourly -> hourly mean MW


def fetch_generation(start, end, country="de") -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    parts, cur = [], start
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=30), end)
        js = _get(EC_URL, {"country": country, "start": cur.strftime("%Y-%m-%d"),
                           "end": nxt.strftime("%Y-%m-%d")})
        parts.append(parse_generation(js))
        cur = nxt + pd.Timedelta(days=1)
        time.sleep(0.5)
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


# ---------------- weather (Open-Meteo) ----------------
def parse_weather(js) -> pd.DataFrame:
    items = js if isinstance(js, list) else [js]
    frames = []
    for it in items:
        h = it["hourly"]
        frames.append(pd.DataFrame({v: h[k] for k, v in WX_VARS.items()},
                                   index=pd.to_datetime(h["time"], utc=True)))
    return pd.concat(frames).groupby(level=0).mean().sort_index()  # mean over locations


def _wx_params(**extra):
    p = {"latitude": ",".join(str(a) for a, _ in WEATHER_POINTS),
         "longitude": ",".join(str(b) for _, b in WEATHER_POINTS),
         "hourly": ",".join(WX_VARS), "timezone": "UTC", "wind_speed_unit": "ms"}
    p.update(extra)
    return p


def fetch_weather_archive(start, end) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    parts, cur = [], start
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=89), end)
        js = _get(OM_ARCHIVE, _wx_params(start_date=cur.strftime("%Y-%m-%d"),
                                         end_date=nxt.strftime("%Y-%m-%d")))
        parts.append(parse_weather(js))
        cur = nxt + pd.Timedelta(days=1)
        time.sleep(1.0)
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_weather_recent(past_days=3, forecast_days=4) -> pd.DataFrame:
    """Recent past + forecast in one call (archive lags ~5 days, so recent hours come from here)."""
    js = _get(OM_FORECAST, _wx_params(past_days=past_days, forecast_days=forecast_days))
    return parse_weather(js)
