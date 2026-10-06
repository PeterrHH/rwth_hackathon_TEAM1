"""Selectable model architectures.

    predictions = train_and_predict("sarimax", df, config)   # df = features.build_features(config)

Every model returns one prediction per test row (columns Household_ID, Timestamp, prediction),
ready for evaluate.submit. Extra information (chosen order, coefficients, intervals) is in
predictions.attrs["info"]. Add a model by writing a function and adding it to MODELS.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import data, features, weather
from .config import LOCAL_TZ, TEST_START, Config


def train_and_predict(name: str, df: pd.DataFrame, config: Config, **params) -> pd.DataFrame:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    predictions = MODELS[name](df, config, **params)
    predictions["prediction"] = predictions["prediction"].clip(lower=0)
    return predictions


# --- Gradient boosting (the reference setup of task0_baseline.ipynb) --------------------

def gbm(df: pd.DataFrame, config: Config, **params) -> pd.DataFrame:
    """HistGradientBoosting on the standard features, one global model for all households."""
    model_params = {"max_iter": 300, "learning_rate": 0.1, "categorical_features": "from_dtype",
                    "random_state": 0, **params}
    train, test = features.split(df, config)
    columns = features.standard_features()
    model = HistGradientBoostingRegressor(**model_params).fit(train[columns], train["kwh"])
    predictions = test[["Household_ID", "Timestamp"]].assign(prediction=model.predict(test[columns]))
    predictions.attrs["info"] = {"params": model_params}
    return predictions


# --- SARIMAX on the daily portfolio -------------------------------------------------------

HDD_BASE = 15.0  # heating degrees = max(0, 15 °C - daily mean temperature)
MIN_HOUSEHOLDS = 30  # training starts once this many households report on a day


def portfolio_daily() -> pd.DataFrame:
    """Per local day: mean consumption per reporting household (kWh/day) and number of households."""
    load = data.load_clean()
    day = load["Timestamp"].dt.tz_convert(LOCAL_TZ).dt.date
    daily = load.groupby(day).agg(kwh_mean=("kwh", "mean"), households=("Household_ID", "nunique"))
    daily["y"] = daily["kwh_mean"] * 96
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily[["y", "households"]].asfreq("D")


def portfolio_temperature(config: Config) -> pd.Series:
    """Daily mean temperature, averaged over stations weighted by their number of households."""
    w = weather.load_weather(config)
    w["date"] = w["Timestamp"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    per_station = w.groupby(["date", "Weather_ID"])["temperature"].mean().unstack()
    households = data.read_households()
    households = households[households["Household_ID"].isin(data.load_clean()["Household_ID"].unique())]
    weights = households["Weather_ID"].value_counts().reindex(per_station.columns).fillna(0)
    available = per_station.notna()
    temperature = (per_station.fillna(0) * weights).sum(axis=1) / (available * weights).sum(axis=1)
    return temperature.asfreq("D").interpolate(limit_direction="both")


def sarimax_exog(temperature: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Regressors for day D, all known at the end of D-1: yesterday's heating degrees and the weekday type."""
    hdd = (HDD_BASE - temperature).clip(lower=0)
    return pd.DataFrame({
        "hdd_1d": hdd.shift(1).reindex(index).to_numpy(),
        "is_weekend": (index.dayofweek >= 5).astype(float),
    }, index=index).bfill()


def _fit(y, exog, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order).fit(disp=False)


def sarimax(df: pd.DataFrame, config: Config, order=None, seasonal_order=None) -> pd.DataFrame:
    """SARIMAX on the daily portfolio mean per household, distributed to households and 15 min.

    1. y_D = mean kWh per reporting household on day D; exog = yesterday's heating degrees + weekend.
    2. Order chosen by AIC on the training days (unless given).
    3. One-step-ahead forecasts over the test period with parameters fixed from training:
       the forecast for D only uses observations up to D-1.
    4. Household forecast = its reading at the same 15 min yesterday x (predicted y_D / actual y_{D-1}).
    """
    daily = portfolio_daily()
    first_day = daily.index[daily["households"] >= MIN_HOUSEHOLDS][0]
    daily = daily.loc[first_day:]
    exog = sarimax_exog(portfolio_temperature(config), daily.index)
    test_start = pd.Timestamp(TEST_START.date())
    y_train, exog_train = daily.loc[:test_start - pd.Timedelta(days=1), "y"], exog.loc[:test_start - pd.Timedelta(days=1)]

    if order is None or seasonal_order is None:
        candidates = [((p, d, q), (P, 0, Q, 7)) for p, d, q, P, Q in
                      itertools.product(range(3), range(2), range(3), range(2), range(2))]
        aic = {}
        for candidate in candidates:
            try:
                aic[candidate] = _fit(y_train, exog_train, *candidate).aic
            except (ValueError, np.linalg.LinAlgError):
                continue  # order not estimable on this series
        order, seasonal_order = min(aic, key=aic.get)
    fitted = _fit(y_train, exog_train, tuple(order), tuple(seasonal_order))

    # Same parameters, full series: one-step-ahead predictions use data up to the previous day only
    full = fitted.apply(daily["y"], exog=exog, refit=False)
    forecast = full.get_prediction(start=test_start)
    y_hat = forecast.predicted_mean

    # Distribute: scale each household's reading from 1 (or 7) days earlier by the predicted change
    y = daily["y"]
    ratio_1d = (y_hat / y.shift(1).reindex(y_hat.index)).fillna(1.0)
    ratio_7d = (y_hat / y.shift(7).reindex(y_hat.index)).fillna(1.0)
    train, test = df[~df["is_test"]], df[df["is_test"]]
    dates = pd.to_datetime(test["date"])
    profile = (train.groupby(["Household_ID", "quarter_hour"])["kwh"].mean().rename("profile"))
    profile_ratio = y_hat / y_train.mean()
    test = test.join(profile, on=["Household_ID", "quarter_hour"])
    prediction = (
        (test["kwh_lag_1d"] * dates.map(ratio_1d).to_numpy())
        .fillna(test["kwh_lag_7d"] * dates.map(ratio_7d).to_numpy())
        .fillna(test["profile"] * dates.map(profile_ratio).to_numpy())
        .fillna(train["kwh"].mean())
    )
    predictions = test[["Household_ID", "Timestamp"]].assign(prediction=prediction.to_numpy())
    predictions.attrs["info"] = {
        "order": tuple(order), "seasonal_order": tuple(seasonal_order),
        "train_days": f"{first_day:%Y-%m-%d} .. {y_train.index[-1]:%Y-%m-%d}",
        "coefficients": fitted.params.round(4).to_dict(),
        "summary": fitted.summary(),
        "daily": pd.DataFrame({"actual": y.reindex(y_hat.index), "forecast": y_hat,
                               "lower_80": forecast.conf_int(alpha=0.2).iloc[:, 0],
                               "upper_80": forecast.conf_int(alpha=0.2).iloc[:, 1],
                               "lower_95": forecast.conf_int(alpha=0.05).iloc[:, 0],
                               "upper_95": forecast.conf_int(alpha=0.05).iloc[:, 1]}),
    }
    return predictions


MODELS = {"gbm": gbm, "sarimax": sarimax}
