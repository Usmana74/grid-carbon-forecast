"""Turn a generation mix (MW per source) into a carbon-intensity estimate.

IMPORTANT: this is a *production-based approximation*. It uses lifecycle emission
factors (roughly IPCC-style medians) applied to domestic generation. It ignores
cross-border imports/exports, so it will not match Electricity Maps or official
figures exactly. Factors are assumptions, editable below.
"""
import warnings

import pandas as pd

# (substring of lower-cased source name, gCO2eq/kWh). First match wins -> order matters.
EMISSION_FACTORS = [
    ("lignite", 1050), ("brown coal", 1050),
    ("hard coal", 820),
    ("coal-derived", 700),
    ("gas", 490),
    ("oil", 650),
    ("biomass", 230),
    ("waste", 400),
    ("nuclear", 12),
    ("hydro", 24),
    ("wind", 11),
    ("solar", 48),
    ("geothermal", 38),
    ("others", 500),
]
# Columns that are not generation sources.
EXCLUDE = ("consumption", "load", "residual", "share", "cross", "price", "forecast", "import", "export")


def intensity_from_mix(mix: pd.DataFrame) -> pd.Series:
    num = pd.Series(0.0, index=mix.index)
    den = pd.Series(0.0, index=mix.index)
    unmapped = []
    for col in mix.columns:
        name = str(col).lower()
        if any(k in name for k in EXCLUDE):
            continue
        factor = next((f for k, f in EMISSION_FACTORS if k in name), None)
        if factor is None:
            unmapped.append(col)
            continue
        s = mix[col].clip(lower=0).fillna(0.0)
        num = num + s * factor
        den = den + s
    if unmapped:
        warnings.warn(f"Unmapped generation types ignored: {unmapped}")
    return (num / den.where(den > 0)).rename("ci")
