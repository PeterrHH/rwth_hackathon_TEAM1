"""The one standard evaluation. Every experiment is scored here, against the same ground truth."""

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data
from .config import COMPLETE_DAY_READINGS, LOCAL_TZ, RESULTS_DIR

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LEADERBOARD = RESULTS_DIR / "leaderboard.csv"
LEVELS = ["portfolio_15min", "portfolio_day", "household_15min", "household_day"]
RANK_BY = "portfolio_15min_nMAE"


def score(actual: pd.Series, pred: pd.Series) -> dict:
    """error = forecast - actual. bias > 0: forecast too high (over-buying)."""
    err = pred - actual
    return {
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "nMAE": 100 * err.abs().mean() / actual.mean(),
        "bias": 100 * err.sum() / actual.sum(),
    }


def _attach(predictions: pd.DataFrame) -> pd.DataFrame:
    """Ground truth + the submitted prediction for every benchmark row."""
    required = {"Household_ID", "Timestamp", "prediction"}
    if not required <= set(predictions.columns):
        raise ValueError(f"predictions need the columns {sorted(required)}")
    truth = data.ground_truth()
    merged = truth.merge(predictions[list(required)], on=["Household_ID", "Timestamp"],
                         how="left", validate="one_to_one")
    missing = merged["prediction"].isna().sum()
    if missing:
        raise ValueError(f"{missing:,} of {len(merged):,} benchmark rows have no prediction. "
                         "Predict every row of data.ground_truth() (fill gaps with a fallback).")
    return merged


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Scores at four levels (rows) x MAE / RMSE / nMAE % / bias % (columns).

    portfolio = sum over all households (what is bought); day = sum per local day, counting
    only complete household-days (>= COMPLETE_DAY_READINGS readings).
    """
    m = _attach(predictions)
    m["date"] = m["Timestamp"].dt.tz_convert(LOCAL_TZ).dt.date
    complete = m.groupby(["Household_ID", "date"])["kwh"].transform("size") >= COMPLETE_DAY_READINGS
    household_day = m[complete].groupby(["Household_ID", "date"])[["kwh", "prediction"]].sum()
    portfolio_15min = m.groupby("Timestamp")[["kwh", "prediction"]].sum()
    portfolio_day = household_day.groupby("date").sum()

    tables = {"portfolio_15min": portfolio_15min, "portfolio_day": portfolio_day,
              "household_15min": m, "household_day": household_day}
    return pd.DataFrame({lvl: score(t["kwh"], t["prediction"]) for lvl, t in tables.items()}).T


def submit(predictions: pd.DataFrame, name: str, author: str, description: str,
           config: dict | None = None, params: dict | None = None) -> pd.DataFrame:
    """Evaluate, save the predictions and add (or replace) the row `name` in the leaderboard."""
    scores = evaluate(predictions)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PREDICTIONS_DIR / f"{name}.npy", _attach(predictions)["prediction"].to_numpy("float32"))

    row = {"name": name, "author": author, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "description": description}
    for lvl in LEVELS:
        row[f"{lvl}_nMAE"] = round(float(scores.loc[lvl, "nMAE"]), 2)
        row[f"{lvl}_bias"] = round(float(scores.loc[lvl, "bias"]), 2)
    row["config"] = json.dumps(config or {})
    row["params"] = json.dumps(params or {}, default=str)

    board = pd.read_csv(LEADERBOARD) if LEADERBOARD.exists() else pd.DataFrame()
    if len(board):
        board = board[board["name"] != name]
    board = pd.concat([board, pd.DataFrame([row])], ignore_index=True).sort_values(RANK_BY)
    board.to_csv(LEADERBOARD, index=False)
    print(f"Submitted '{name}'. Portfolio nMAE: {scores.loc['portfolio_15min', 'nMAE']:.2f} % per 15 min, "
          f"{scores.loc['portfolio_day', 'nMAE']:.2f} % per day")
    return scores.round(3)


def leaderboard() -> pd.DataFrame:
    """All submissions, best first (by portfolio 15-min nMAE)."""
    board = pd.read_csv(LEADERBOARD).sort_values(RANK_BY)
    return board.drop(columns=["config", "params"]).set_index("name")


def load_predictions(name: str) -> pd.DataFrame:
    """Ground truth + a saved submission's predictions (only on the machine that has the .npy file)."""
    truth = data.ground_truth()
    return truth.assign(prediction=np.load(PREDICTIONS_DIR / f"{name}.npy"))


def plot_daily(names: list[str], start: str | None = None, end: str | None = None):
    """Daily portfolio energy: actual vs saved submissions."""
    truth = data.ground_truth()
    day = truth["Timestamp"].dt.tz_convert(LOCAL_TZ).dt.date
    daily = truth.groupby(day)["kwh"].sum().to_frame("actual")
    for name in names:
        daily[name] = pd.Series(np.load(PREDICTIONS_DIR / f"{name}.npy")).groupby(day.to_numpy()).sum()
    readings = truth.groupby(day).size()
    daily = daily[readings >= 0.5 * readings.median()]  # drop the partial last day of the test period
    daily.index = pd.to_datetime(daily.index)
    daily = daily.loc[start:end]

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily.index, daily["actual"], color="#0b0b0b", linewidth=1.8, label="actual")
    for name, color in zip(names, colors):
        ax.plot(daily.index, daily[name], color=color, linewidth=1.2, label=name)
    ax.set(title="Daily portfolio energy (all households, test period)", ylabel="kWh per day")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=min(len(names) + 1, 4))
    plt.show()
