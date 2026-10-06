"""Hyperparameter tuning on a validation holdout inside the training period.

The test year (from TEST_START) is never used here: tuning on it would make the test score
optimistic and incomparable with the leaderboard.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import COMPLETE_DAY_READINGS, LOCAL_TZ, TEST_START, Config
from .evaluate import score

SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_iter": [200, 400, 800],
    "max_leaf_nodes": [31, 63, 127, 255],
    "min_samples_leaf": [20, 100, 500],
    "l2_regularization": [0.0, 0.1, 1.0],
}
FIXED = {"categorical_features": "from_dtype", "random_state": 0, "early_stopping": False}


def holdout_split(df: pd.DataFrame, config: Config, start: str = "2023-01-01",
                  sample: int | None = 1_000_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit rows: before `start` (sampled to `sample` rows). Validation rows: `start` up to TEST_START."""
    start = pd.Timestamp(start, tz=LOCAL_TZ)
    fit_rows = df[df["Timestamp"] < start]
    val_rows = df[(df["Timestamp"] >= start) & (df["Timestamp"] < TEST_START)]
    if sample is not None and len(fit_rows) > sample:
        fit_rows = fit_rows.sample(n=sample, random_state=config.random_state)
    assert val_rows["Timestamp"].max() < TEST_START, "validation must not reach the test period"
    assert fit_rows["Timestamp"].max() < start <= val_rows["Timestamp"].min()
    return fit_rows, val_rows


def score_rows(rows: pd.DataFrame, prediction) -> pd.DataFrame:
    """nMAE / R² at the four levels of evaluate.evaluate, for any set of rows with a `kwh` column."""
    m = rows[["Household_ID", "Timestamp", "kwh"]].assign(prediction=np.asarray(prediction))
    m["date"] = m["Timestamp"].dt.tz_convert(LOCAL_TZ).dt.date
    complete = m.groupby(["Household_ID", "date"])["kwh"].transform("size") >= COMPLETE_DAY_READINGS
    household_day = m[complete].groupby(["Household_ID", "date"])[["kwh", "prediction"]].sum()
    tables = {"portfolio_15min": m.groupby("Timestamp")[["kwh", "prediction"]].sum(),
              "portfolio_day": household_day.groupby("date").sum(),
              "household_15min": m, "household_day": household_day}
    return pd.DataFrame({lvl: score(t["kwh"], t["prediction"]) for lvl, t in tables.items()}).T


def random_candidates(n: int, seed: int = 0, base: dict | None = None) -> list[dict]:
    """`n` random parameter sets from SEARCH_SPACE, plus `base` (the current parameters) first."""
    rng = np.random.default_rng(seed)
    if base:
        # spell out sklearn's defaults for the searched parameters, so the results table has no gaps
        defaults = HistGradientBoostingRegressor(**base).get_params()
        base = {**{k: defaults[k] for k in [*SEARCH_SPACE, "early_stopping"]}, **base}
    candidates = [dict(base)] if base else []
    seen = set()
    while len(candidates) < n + bool(base):
        params = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        key = tuple(params.values())
        if key not in seen:
            seen.add(key)
            candidates.append({**params, **FIXED})
    return candidates


def tune(fit_rows: pd.DataFrame, val_rows: pd.DataFrame, candidates: list[dict],
         features: list[str]) -> pd.DataFrame:
    """Fit every candidate on fit_rows, score on val_rows; best (lowest portfolio_15min MAE) first.

    On one fixed validation set, ranking by MAE and by nMAE is identical (nMAE = MAE / constant).
    """
    results = []
    for i, params in enumerate(candidates):
        t = time.time()
        model = HistGradientBoostingRegressor(**params).fit(fit_rows[features], fit_rows["kwh"])
        scores = score_rows(val_rows, model.predict(val_rows[features]).clip(min=0))
        results.append({
            "candidate": i,
            "portfolio_15min_MAE": scores.loc["portfolio_15min", "MAE"],
            "portfolio_15min_RMSE": scores.loc["portfolio_15min", "RMSE"],
            "portfolio_15min_R2": scores.loc["portfolio_15min", "R2"],
            "portfolio_day_MAE": scores.loc["portfolio_day", "MAE"],
            "household_15min_R2": scores.loc["household_15min", "R2"],
            "fit_seconds": round(time.time() - t, 1),
            "params": params,
        })
        print(f"candidate {i:2d}: portfolio 15-min MAE {results[-1]['portfolio_15min_MAE']:.2f} kWh "
              f"({results[-1]['fit_seconds']} s)")
    return pd.DataFrame(results).sort_values("portfolio_15min_MAE").reset_index(drop=True)
