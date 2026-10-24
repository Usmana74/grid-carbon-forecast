# Germany grid carbon-intensity forecast (scored live)

**Live dashboard:** https://usmana74.github.io/grid-carbon-forecast/

A 48-hour hourly forecast of Germany's electricity carbon intensity with an 80% prediction
interval, a "cleanest 3 hours" window, and a high-carbon alert. A scheduled job issues a new
forecast every day, freezes it in git, and later grades it against what actually happened.
Live since 2026-10-24.

Carbon-intensity trackers already exist. The point of this project is evaluation rigour and an
auditable track record, not a new idea.

## Backtest results (rolling origin, as of 2026-10-24)

4 folds of 14 days, trained only on data before each fold, 9,408 forecast-hours in total.

| Method | MAE (g/kWh) | RMSE | Bias |
|---|---|---|---|
| This model (median) | 37.1 | 47.4 | -15.5 |
| Same hour yesterday | 97.2 | 125.0 | -5.0 |
| Same hour last week | 119.1 | 150.2 | -4.9 |
| Persistence (last value) | 126.8 | 161.8 | -5.7 |

Skill vs the best baseline: +61.8%. Treat this as an UPPER BOUND (see Limitations).

By horizon (MAE): 1-6 h 36.3 | 7-12 h 36.3 | 13-24 h 37.1 | 25-48 h 37.5
(baseline "same hour yesterday": 84.2 | 84.1 | 86.1 | 109.2).

Per fold MAE: 38.0, 46.5, 32.8, 31.0. The second fold is the weakest.

80% interval coverage: 74.5% after conformal calibration (67.6% before). Closer to the 80%
target, but still under-covers.

### Why the +61.8% is an upper bound

The backtest feeds the model the OBSERVED weather for the hour being predicted. In real use only a
weather FORECAST is available, which has errors that grow with lead time. Two signs of this:
error is almost flat across horizons (a real forecaster gets worse further out), and a
weather ablation (`experiments/ablation.py`, same 4 folds, calibrated model) gave:

| Weather input | MAE | Skill vs best baseline |
|---|---|---|
| Observed (perfect) weather | 37.1 | +61.8% |
| Noisy weather (independent noise) | 43.8 | +54.9% |
| No weather features | 74.8 | +23.0% |

The noisy case adds independent random noise, whereas real forecast errors are correlated, so it
is optimistic (the noise was also applied to the training data). The honest range is roughly +23% (no weather) to +62% (perfect weather). The live
scoreboard is the number to trust.

## Live scoreboard

Every forecast is saved to `docs/data/forecasts/` and never rewritten (the code refuses to
overwrite an existing forecast, and git history shows it). Once at least 12 of a forecast's hours
have occurred, it is scored against the realized values. The dashboard shows the running MAE, the
skill versus the same-hour-yesterday baseline, and interval coverage.

## How it works

- Data: generation mix from Energy-Charts (Fraunhofer ISE); wind, sun, temperature and cloud from
  Open-Meteo, averaged over 7 points in Germany. Please check each provider's terms and
  attribution requirements before reusing the data.
- Target: production-based carbon intensity = generation mix x fixed lifecycle emission factors
  (g CO2eq/kWh): lignite 1050, hard coal 820, oil 650, gas 490, biomass 230, waste 400,
  solar 48, hydro 24, wind 11, other 500. Factors are assumptions and live in `gridcast/carbon.py`.
- Model: gradient-boosted quantile regression (p10 / p50 / p90), one global model over horizons
  1-48 h, plus split-conformal calibration of the interval per horizon bucket.
- Evaluation: rolling-origin backtest with an assertion that no training target falls after the
  fold start, and baselines that must be beaten.
- Automation: GitHub Actions runs daily, caches the dataset, commits forecasts and the
  scoreboard, and re-runs the backtest on Sundays.

## Limitations

- Observed vs forecast weather mismatch (above): the backtest is optimistic.
- The carbon number is an approximation: fixed emission factors, imports and exports are ignored,
  pumped-storage generation is treated like hydro, and the factors for "Others" and "Waste" are
  rough guesses. It will not match Electricity Maps or official figures.
- It estimates AVERAGE emissions, not the marginal emissions caused by extra demand.
- Intervals still under-cover (74.5% vs 80%).
- The alert is a "high-carbon period" (median above the 90th percentile of the last year for
  6+ hours), not a validated Dunkelflaute detector.
- One person's charging barely moves the grid. No emissions-reduction claim is made.
- Not an official or safety-critical service.

## Reproduce (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m gridcast check
python -m gridcast ingest
python -m gridcast backtest
python -m http.server -d docs 8000
```

## Roadmap

- [x] Weather ablation with the calibrated model (`experiments/ablation.py`)
- [ ] Train on archived weather forecasts instead of observed weather
- [ ] Improve interval calibration toward 80%
- [ ] Compare against published day-ahead renewable forecasts
- [ ] Automated gap, revision and daylight-saving checks on the feeds (dataaudit)

MIT licensed.
