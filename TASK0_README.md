# Task 0 · Shared forecasting benchmark

One fixed definition of the task, the data and the evaluation, so that everyone can try different models and data processing **and every result can be compared on one leaderboard**.

```
forecast/                  shared package (import it, don't copy it)
  config.py                fixed benchmark rules + Config (per-experiment choices)
  data.py                  valid households, standard target cleaning, ground truth
  weather.py               hourly weather -> 15 min ("step" or "interpolate")
  features.py              standard features, add_lag() (enforces the day-ahead rule), split()
  evaluate.py              evaluate(), submit(), leaderboard(), plot_daily()
  baselines.py             naive reference forecasts
task0_baseline.ipynb       gradient boosting reference model
experiment_template.ipynb  copy this to run your own experiment
results/leaderboard.csv    every submission's scores (commit this)
results/predictions/       saved predictions, one .npy per submission (git-ignored)
cache/                     cleaned load data, built on first use (git-ignored)
```

## 1. The task

| | |
|---|---|
| **Target** | `kWh_received_Total`: energy drawn from the grid, per household, per 15-min interval |
| **Forecast** | All 15-min intervals of delivery day **D** (local time, Europe/Berlin) |
| **Day-ahead rule** | The forecast for day D is made at the **end of day D-1**. Features may use any data up to D-1 23:45 (load **and** weather), nothing from day D. |
| **Main quantity** | The **portfolio**: the sum over all households, which is what is procured |

`features.add_lag(df, source, cols, days)` is the only way the standard features look into the past. It refuses `days < 1`. It also blanks values that would still fall on day D: on the 25-hour day when daylight saving ends, "24 h earlier" can still be the same day.

## 2. Fixed benchmark rules (`forecast/config.py`)

These are the same for every experiment. **Don't change them**, or your scores are no longer comparable.

| Rule | Value |
|---|---|
| Valid households | > 365 days between first and last reading **and** a row in `meta_data.csv`. 379 households, of which 4 have no Total readings → **375** with data. |
| Target cleaning | Rows with missing `kWh_received_Total` are dropped; readings > 10 kWh per 15 min (40 kW, physically impossible) are dropped. **The target is never imputed.** |
| Train period | everything before **2023-03-01 00:00** (local) |
| Test period | 2023-03-01 → end of data (2024-02-28 00:45), ≈ 12 months including a full winter |
| Ground truth | `data.ground_truth()`: every cleaned reading in the test period (**12,471,135 rows**). Every submission predicts exactly these rows. |
| Split | by date, the same for all households. Never random. |

## 3. Standard features (`features.build_features(config)`)

| Group | Features | Definition |
|---|---|---|
| Load history | `kwh_lag_1d`, `kwh_lag_7d` | Same household, same 15 min, 1 / 7 days earlier |
| Weather (observed yesterday) | `temperature_1d`, `sunshine_1d`, `humidity_1d`, `wind_speed_1d` | Station weather of the same 15 min on D-1 |
| | `temperature_day_mean_1d` | Mean temperature of D-1 |
| | `sunshine_filled_1d` | 1 if sunshine came from other stations (3 stations have no sensor) |
| Calendar (local) | `quarter_hour` (0–95), `weekday` (0 = Mon), `month` | |
| Household | `is_treatment`, `after_visit` (heat pump optimised), `has_pv` (NaN if unknown) | |
| Metadata | living area, residents (numeric); building type, heat-pump type (categorical); all other `Survey_*` answers as 1 / 0 / NaN | |

Missing feature values stay NaN; nothing is filled with 0, because 0 has a meaning (0 °C, no sun, no consumption).

**Weather processing:**
- A weather timestamp marks the **end** of its hour, so each value is moved to the hour's start.
- Gaps up to 6 h are interpolated.
- Missing sunshine is filled with the mean of the other stations at the same hour, plus a flag.

### Per-experiment choices (`Config`)

| Field | Options | Default |
|---|---|---|
| `weather_resampling` | `"step"`: each 15-min interval gets its hour's value · `"interpolate"`: temperature, humidity and wind are interpolated linearly between hourly midpoints; sunshine stays per hour | `"step"` |
| `weather_max_gap_hours` | weather gaps up to this length are interpolated | 6 |
| `train_sample` | number of random training rows (`None` = all) | 3,000,000 |
| `random_state` | seed | 0 |

The config is stored with every submission.

## 4. Standard evaluation (`forecast/evaluate.py`)

`evaluate.submit(predictions, name, author, description, config, params)`:

1. **Checks the submission.** It needs one row per ground-truth row, with columns `Household_ID`, `Timestamp`, `prediction`. Missing or duplicate rows → error.
2. **Scores it at four levels:**

   | Level | Aggregation |
   |---|---|
   | `portfolio_15min` | sum over households per 15 min. **Ranking metric** |
   | `portfolio_day` | sum per local day |
   | `household_15min` | each reading |
   | `household_day` | sum per household per local day |

   Daily levels count only **complete household-days** (≥ 92 readings: 96 per day, 92 or 100 on daylight-saving days), and prediction and actual are summed over the same readings.

3. **Metrics** (error = forecast − actual):
   - **MAE**, **RMSE** in kWh.
   - **nMAE %** = MAE / mean actual. Comparable across levels, and the leaderboard number.
   - **bias %** = Σ error / Σ actual. Above 0 means over-buying, below 0 under-buying.
4. **Saves** the predictions to `results/predictions/<name>.npy` and adds (or replaces) the row `<name>` in `results/leaderboard.csv`, together with nMAE and bias for every level, the config and the model parameters.

Other helpers:
- `evaluate.evaluate(predictions)` scores without saving.
- `evaluate.leaderboard()` shows all submissions, best first.
- `evaluate.plot_daily([names])` plots daily portfolio energy.

## 5. How to collaborate

1. **Setup:** `poetry install` (or any environment with the `pyproject.toml` packages), then work from the repository root. The first run builds `cache/` (≈ 1 min).
2. **New idea:** copy `experiment_template.ipynb` to `exp_<name>_<idea>.ipynb`, then:
   - set `NAME` (unique), `AUTHOR`, `DESCRIPTION` and `Config(...)`;
   - change the data processing and/or the model;
   - run it top to bottom.
3. **Rules for a valid experiment:**
   - Predict **every** test row (`test` from `features.split`, i.e. `data.ground_truth()`). Use a fallback where your model has no input, as `baselines.naive` does.
   - Train only on data before `TEST_START`. You may clean, filter or reweight **training** rows however you like, but never drop or alter test rows.
   - Respect the day-ahead rule: build any history feature with `features.add_lag(..., days >= 1)`.
   - Don't edit the fixed rules in `config.py` or the scoring in `evaluate.py`. If they need to change, agree as a team, delete `cache/` and re-run every experiment.
4. **Share:** commit your notebook **and** `results/leaderboard.csv`. Pull before you run to avoid conflicts in the CSV.
5. **Compare predictions** (e.g. `plot_daily`) only for submissions whose `.npy` file you have. These are large (≈ 50 MB) and not in git; share them separately if needed.
6. **Reusable code** (a new feature, a new cleaning step) goes into `forecast/` as a function with an option in `Config`, so others can switch it on.

## 6. Current leaderboard (2026-10-06)

| name | model | portfolio 15 min nMAE | portfolio day nMAE | bias (15 min) |
|---|---|---|---|---|
| `gbm_step` | gradient boosting, `step` weather | **11.64 %** | **7.79 %** | +0.86 % |
| `gbm_interpolate` | gradient boosting, `interpolate` weather | 11.68 % | 7.84 % | +0.82 % |
| `naive_lag_1d` | same 15 min yesterday | 13.41 % | 7.90 % | +0.28 % |
| `linear_example` | linear regression (template example) | 13.43 % | 8.74 % | +2.23 % |
| `naive_lag_7d` | same 15 min a week ago | 20.45 % | 16.92 % | +0.60 % |

**Reading:**
- Gradient boosting beats yesterday's load clearly per 15 min (11.6 vs 13.4 %), but on **daily totals only barely** (7.8 vs 7.9 %). Yesterday's total is already a strong guess for today's, and without a weather forecast for day D the model reacts to weather changes about a day late, as the chart in `task0_baseline.ipynb` shows.
- Weather interpolation makes no difference with yesterday's weather.
- The reference model (`task0_baseline.ipynb`):

  ```python
  HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                categorical_features="from_dtype", random_state=0)
  ```

  It is trained on a random sample of 3 M training rows with all standard features.

## 7. Known limitations and ideas

1. The data has no weather forecasts, so day D's weather is unknown by design. A real system would add a weather forecast; expect a large gain.
2. Sunshine only says *how long* the sun shone, not *how strongly*. Sun elevation is not yet a feature.
3. The target is not scaled per household, so large households weigh more in training.
4. There is no validation period inside the training data. Tune on, e.g., the last winter before `TEST_START`, never on the test set.
5. The PV flag is unknown for many households, and about 1 in 10 PV-flagged households shows no PV pattern.
6. Ideas for experiments:
   - heating-degree features;
   - daily-total features (yesterday's energy, last-7-day mean);
   - per-household scaling;
   - PV / no-PV or clustered models (Level 1);
   - quantile models for uncertainty (Level 3).
