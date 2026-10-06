"""Hourly station weather -> 15-min intervals."""

import pandas as pd

from .config import DATA_DIR, LOCAL_TZ, Config

WEATHER_COLS = {
    "Temperature_avg_hourly": "temperature",
    "Sunshine_duration_hourly": "sunshine",  # sunny fraction of the hour (0-1)
    "Humidity_avg_hourly": "humidity",
    "WindSpeed_hourly": "wind_speed",
}
SMOOTH_COLS = ["temperature", "humidity", "wind_speed"]  # interpolated in "interpolate" mode

# An hourly mean describes the middle of its hour (hh:30). The four 15-min intervals have their
# middles at hh:07.5 / 22.5 / 37.5 / 52.5, so they mix in the previous or next hour.
QUARTERS = pd.DataFrame({
    "quarter": [0, 1, 2, 3],
    "w_prev": [0.375, 0.125, 0.0, 0.0],
    "w_next": [0.0, 0.0, 0.125, 0.375],
})


def load_weather(config: Config) -> pd.DataFrame:
    """One row per station and 15-min interval.

    Columns: Weather_ID, Timestamp (UTC, interval start), temperature, sunshine, humidity,
    wind_speed, sunshine_filled, temperature_day_mean.
    """
    files = sorted((DATA_DIR / "weather_data_hourly").glob("*.csv"))
    w = pd.concat([pd.read_csv(f, sep=";") for f in files], ignore_index=True)
    w = w[["Weather_ID", "Timestamp", *WEATHER_COLS]].rename(columns=WEATHER_COLS)
    # A weather timestamp marks the END of the hour it describes -> move it to the hour's start
    w["hour_start"] = pd.to_datetime(w["Timestamp"], format="%Y-%m-%d %H:%M:%S%z") - pd.Timedelta(hours=1)
    w = w.drop(columns="Timestamp").sort_values(["Weather_ID", "hour_start"]).reset_index(drop=True)

    # Short gaps: interpolate within each station
    for col in WEATHER_COLS.values():
        w[col] = w.groupby("Weather_ID")[col].transform(
            lambda s: s.interpolate(limit=config.weather_max_gap_hours, limit_area="inside"))

    # 3 stations have no sunshine sensor: mean of the other stations at the same hour, flagged.
    # Never 0, which would mean "overcast".
    w["sunshine_filled"] = w["sunshine"].isna().astype("int8")
    w["sunshine"] = w["sunshine"].fillna(w.groupby("hour_start")["sunshine"].transform("mean"))

    local_day = w["hour_start"].dt.tz_convert(LOCAL_TZ).dt.date
    w["temperature_day_mean"] = w.groupby(["Weather_ID", local_day])["temperature"].transform("mean")

    # Hourly -> 15 min
    by_station = w.groupby("Weather_ID")
    for col in SMOOTH_COLS:
        w[f"{col}_prev"] = by_station[col].shift(1).fillna(w[col])
        w[f"{col}_next"] = by_station[col].shift(-1).fillna(w[col])
    w = w.merge(QUARTERS, how="cross")
    w["Timestamp"] = w["hour_start"] + pd.to_timedelta(15 * w["quarter"], unit="min")
    if config.weather_resampling == "interpolate":
        for col in SMOOTH_COLS:
            w[col] = ((1 - w["w_prev"] - w["w_next"]) * w[col]
                      + w["w_prev"] * w[f"{col}_prev"] + w["w_next"] * w[f"{col}_next"])

    value_cols = [*WEATHER_COLS.values(), "temperature_day_mean"]
    w[value_cols] = w[value_cols].astype("float32")
    return w[["Weather_ID", "Timestamp", *value_cols, "sunshine_filled"]].sort_values(
        ["Weather_ID", "Timestamp"]).reset_index(drop=True)
