from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = ROOT / "docs"
SITE_DATA = DOCS / "data"

HORIZONS = list(range(1, 49))          # forecast 1..48 hours ahead
QUANTILES = (0.1, 0.5, 0.9)            # 80% prediction interval + median
TZ_LOCAL = "Europe/Berlin"
MODEL_VERSION = "0.1.0"
UNIT = "gCO2eq/kWh (production-based approximation)"

# Representative points for wind/solar/temperature (averaged into one national signal).
WEATHER_POINTS = [
    (54.2, 9.2),    # Schleswig-Holstein (wind)
    (53.6, 7.2),    # North Sea coast
    (52.8, 9.0),    # Lower Saxony
    (52.4, 13.2),   # Brandenburg
    (51.4, 7.5),    # North Rhine-Westphalia
    (48.6, 9.0),    # Baden-Wurttemberg (solar)
    (48.5, 11.5),   # Bavaria (solar)
]
