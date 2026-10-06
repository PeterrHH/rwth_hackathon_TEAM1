"""Naive reference forecasts every model should beat."""

import pandas as pd


def naive(df: pd.DataFrame, lag: str) -> pd.DataFrame:
    """Repeat the household's load from `lag` ("kwh_lag_1d" or "kwh_lag_7d") on the test rows.

    Where that value is missing: the other lag, then the household's training mean for that
    quarter-hour, then the overall training mean, so every benchmark row gets a prediction.
    """
    other = "kwh_lag_7d" if lag == "kwh_lag_1d" else "kwh_lag_1d"
    train, test = df[~df["is_test"]], df[df["is_test"]]
    profile = train.groupby(["Household_ID", "quarter_hour"])["kwh"].mean().rename("profile")
    test = test.join(profile, on=["Household_ID", "quarter_hour"])
    prediction = test[lag].fillna(test[other]).fillna(test["profile"]).fillna(train["kwh"].mean())
    return test[["Household_ID", "Timestamp"]].assign(prediction=prediction)
