"""Synthetic grid + weather generator. For tests and the offline demo ONLY.

Never publish synthetic output as real data: everything produced from this module is
tagged source="synthetic" and the dashboard shows a banner for it.
"""
import numpy as np
import pandas as pd

from .carbon import intensity_from_mix


def synthetic_dataset(days=730, seed=0, end="2025-06-30 23:00") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=days * 24, freq="1h")
    n = len(idx)
    doy, hod, dow = idx.day_of_year.to_numpy(), idx.hour.to_numpy(), idx.day_of_week.to_numpy()

    def ar1(phi, sigma):
        e, x = rng.normal(size=n), np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + sigma * e[i]
        return x

    wind = np.clip(7.5 + 2.4 * ar1(0.97, 0.25) / 1.0 + 1.5 * np.cos(2 * np.pi * (doy - 20) / 365), 0.3, 25)
    cloud = np.clip(55 + 25 * ar1(0.95, 0.35) + 10 * np.cos(2 * np.pi * (doy - 10) / 365), 0, 100)
    sun = np.clip(np.sin(np.pi * (hod - 5.5) / (11 + 4 * np.sin(2 * np.pi * (doy - 80) / 365))), 0, None)
    ghi = sun * (550 + 300 * np.sin(2 * np.pi * (doy - 80) / 365)) * (1 - 0.75 * cloud / 100)
    temp = 10 + 10 * np.sin(2 * np.pi * (doy - 110) / 365) + 3 * np.sin(2 * np.pi * (hod - 9) / 24) \
        + rng.normal(0, 0.8, n)

    wind_mw = 62000 * np.clip((wind / 12.0) ** 3, 0, 1)
    solar_mw = 72000 * np.clip(ghi, 0, None) / 1000
    load = 54000 + 9000 * np.sin(np.pi * (hod - 5) / 17).clip(0, None) - 4000 * (dow >= 5) \
        + 150 * (15 - temp).clip(0, None) + rng.normal(0, 800, n)
    biomass, hydro = np.full(n, 4200.0), np.full(n, 1800.0)
    residual = np.clip(load - wind_mw - solar_mw - biomass - hydro, 0, None)
    lignite = np.minimum(residual, 17000)
    coal = np.minimum(residual - lignite, 8000)
    gas = residual - lignite - coal

    mix = pd.DataFrame({
        "Fossil brown coal / lignite": lignite, "Fossil hard coal": coal, "Fossil gas": gas,
        "Wind onshore": wind_mw, "Solar": solar_mw, "Biomass": biomass, "Hydro Run-of-River": hydro,
        "Load": load,  # must be ignored by carbon.py
    }, index=idx)
    ci = intensity_from_mix(mix) + rng.normal(0, 6, n)
    out = pd.DataFrame({"ci": ci, "wind": wind, "ghi": ghi, "temp": temp, "cloud": cloud}, index=idx)
    out.attrs["source"] = "synthetic"
    return out
