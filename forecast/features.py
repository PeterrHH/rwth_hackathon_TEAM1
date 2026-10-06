"""Standard feature table that respects the day-ahead rule."""

import numpy as np
import pandas as pd

from . import data, weather
from .config import LOCAL_TZ, TEST_START, Config

WEATHER_FEATURES = ["temperature_1d", "temperature_day_mean_1d", "sunshine_1d", "humidity_1d",
                    "wind_speed_1d", "sunshine_filled_1d"]
CALENDAR_FEATURES = ["quarter_hour", "weekday", "month"]
LOAD_FEATURES = ["kwh_lag_1d", "kwh_lag_7d"]
HOUSEHOLD_FEATURES = ["is_treatment", "after_visit", "has_pv", *data.NUMERIC, *data.CATEGORICAL]


def standard_features() -> list[str]:
    return [*WEATHER_FEATURES, *CALENDAR_FEATURES, *LOAD_FEATURES, *HOUSEHOLD_FEATURES,
            *data.survey_flag_columns()]


def add_lag(df: pd.DataFrame, source: pd.DataFrame, cols: list[str], days: int,
            key: str = "Household_ID", suffix: str | None = None) -> pd.DataFrame:
    """Add `cols` from `source` measured `days` days before each row's Timestamp.

    Day-ahead rule: only data from before the target's delivery day may be used. `days` must be
    >= 1, and values that still fall on the delivery day (possible on the 25-hour day when
    daylight saving ends) are set to NaN. `df` needs a `day_start` column (see build_features).
    """
    if days < 1:
        raise ValueError("Day-ahead rule: lags must be at least 1 day")
    suffix = suffix or f"_lag_{days}d"
    lagged = source[[key, "Timestamp", *cols]].copy()
    lagged["Timestamp"] = lagged["Timestamp"] + pd.Timedelta(days=days)
    lagged = lagged.rename(columns={c: c + suffix for c in cols})
    out = df.merge(lagged, on=[key, "Timestamp"], how="left")
    too_late = out["Timestamp"] - pd.Timedelta(days=days) >= out["day_start"]
    out.loc[too_late, [c + suffix for c in cols]] = np.nan
    return out


def build_features(config: Config) -> tuple[pd.DataFrame, list[int]]:
    """All cleaned readings (train and test) with the standard features, plus the loaded household IDs.

    Extra columns: kwh (target), is_test, date (local delivery day), day_start.
    """
    load, list_household_id = data.load_clean()
    df = load.merge(data.read_households(), on="Household_ID", how="left")

    local = df["Timestamp"].dt.tz_convert(LOCAL_TZ)
    df["date"] = local.dt.date
    df["day_start"] = local.dt.normalize().dt.tz_convert("UTC")  # local midnight of the delivery day
    df["quarter_hour"] = (local.dt.hour * 4 + local.dt.minute // 15).astype("int8")
    df["weekday"] = local.dt.weekday.astype("int8")
    df["month"] = local.dt.month.astype("int8")

    # Heat-pump optimisation visit: 1 once it has happened (treatment group only)
    df["after_visit"] = (df["AffectsTimePoint"] == "after visit").astype("int8")
    df = df.drop(columns="AffectsTimePoint")

    # History of the household's own load: same 15 min, 1 and 7 days earlier
    for days in (1, 7):
        df = add_lag(df, load, ["kwh"], days)

    # Observed weather: same 15 min on the previous day (weather for day D is not known yet)
    w = weather.load_weather(config)
    df = add_lag(df, w, [c.removesuffix("_1d") for c in WEATHER_FEATURES], 1, key="Weather_ID", suffix="_1d")

    df["is_test"] = df["Timestamp"] >= TEST_START
    return df, list_household_id


def split(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train (before TEST_START, optionally sampled) and test (the benchmark rows)."""
    train, test = df[~df["is_test"]], df[df["is_test"]]
    if config.train_sample is not None and len(train) > config.train_sample:
        train = train.sample(n=config.train_sample, random_state=config.random_state)
    return train, test
