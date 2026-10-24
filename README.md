# gridcast: Germany grid carbon-intensity forecast, scored live

A 48-hour forecast of Germany's electricity carbon intensity with an 80% prediction
interval, a "best time to charge" window, a high-carbon alert, and a **public
scoreboard that grades every forecast against what actually happened**.

The point is the track record, not a one-off notebook score: forecasts are issued
daily by CI, committed to git, and never rewritten, so accuracy is auditable.

## How it works

```
Energy-Charts (generation mix) ─┐                         ┌─ forecasts/*.json (frozen)
                                ├─ dataset ─ features ─ model ─┼─ scoreboard.json (live accuracy)
Open-Meteo (wind/sun/temp) ─────┘                         └─ backtest.json (rolling origin)
                                                                 │
                                                       docs/index.html (GitHub Pages)
```

- **Target:** hourly production-based carbon intensity (generation mix x lifecycle emission factors).
- **Model:** global gradient-boosted quantile regressors (p10/p50/p90) over horizons 1-48 h.
- **Baselines it must beat:** persistence, same hour yesterday, same hour last week.
- **Evaluation:** rolling-origin backtest (train on the past only, a leak guard asserts it) plus the live scoreboard.
- **Outputs:** median + 80% interval, cleanest 3-hour window, high-carbon alert (>= 6 h above the 90th percentile of the last year).

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                  # 11 tests, offline
python -m gridcast demo              # offline end-to-end run on SYNTHETIC data
python -m http.server -d docs 8000   # open http://localhost:8000/?demo
```

### Real data (do this once, in order)

```bash
python -m gridcast check             # 1. verifies both live APIs parse as expected
python -m gridcast ingest            # 2. backfills ~2 years (a few minutes)
python -m gridcast backtest          # 3. honest rolling-origin numbers
python -m gridcast daily             # 4. issues today's forecast + updates the scoreboard
python -m http.server -d docs 8000   #    open http://localhost:8000/
```

### Deploy

1. Push to GitHub. In **Settings > Pages**, serve from branch `main`, folder `/docs`.
2. The workflow in `.github/workflows/daily.yml` runs daily, commits `docs/data/`, and runs the backtest on Sundays.

## Status and limitations (read before quoting any number)

- **Synthetic demo numbers are meaningless.** The demo generator makes carbon a near-deterministic function of the weather inputs, so skill looks absurdly high. It exists to test the plumbing only.
- **Carbon intensity is an approximation.** Production-based, fixed emission factors (`gridcast/carbon.py`), no cross-border flows. It will not match Electricity Maps or official numbers.
- **Weather mismatch:** training uses observed/reanalysis weather; live forecasts use forecast weather. Backtest skill is therefore likely optimistic. The live scoreboard is the number to trust.
- **Interval calibration:** the 80% interval can under-cover. Coverage is reported honestly; conformal calibration is the planned fix.
- **API shapes** were written from public docs and covered by offline parser tests; `python -m gridcast check` confirms them against the live services.
- Not an official or safety-critical service.

## Roadmap

- [ ] Conformal calibration of the intervals
- [ ] Train on archived *forecast* weather instead of reanalysis
- [ ] Compare against published day-ahead renewable forecasts
- [ ] `dataaudit` integration: automated gap/revision/DST checks on the feeds
- [ ] KBA vehicle-registration angle (EV adoption vs. grid cleanliness)

## Layout

```
gridcast/   sources.py carbon.py features.py model.py backtest.py publish.py synthetic.py cli.py
tests/      offline tests incl. leakage and frozen-forecast guarantees
docs/       index.html dashboard (Pages) + data/ (forecasts, scoreboard, backtest)
```

MIT licensed.
