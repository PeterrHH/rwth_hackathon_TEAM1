"""Paths, fixed benchmark rules and the per-experiment config."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
META_DIR = DATA_DIR / "smart_meter_meta_data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

LOCAL_TZ = "Europe/Berlin"  # delivery days are local days

# --- Fixed benchmark rules ---------------------------------------------------
# Every experiment is scored against the same ground truth. Changing any of
# these makes results incomparable with the leaderboard (and requires deleting cache/).
TEST_START = pd.Timestamp("2023-03-01", tz=LOCAL_TZ)  # train before, test from here (~12 months)
MIN_DAYS_OF_DATA = 365  # valid household: > 1 year between first and last reading ...
#                         ... and a row in meta_data.csv
MAX_KWH_PER_15MIN = 10.0  # readings above this (40 kW) are physically impossible and dropped
COMPLETE_DAY_READINGS = 92  # a household-day counts in daily scores with >= 92 readings (96; 92/100 on DST days)
# Day-ahead rule: the forecast for day D is made at the end of day D-1. Features may use
# any data up to D-1 23:45 (local), nothing from day D. Enforced by features.add_lag().


@dataclass
class Config:
    """Data-processing choices of one experiment. Logged with every submission."""

    weather_resampling: str = "step"  # "step": every 15 min gets its hour's value
    #                                   "interpolate": temperature, humidity, wind linear between hourly midpoints
    weather_max_gap_hours: int = 6  # weather gaps up to this length are interpolated
    train_sample: int | None = 3_000_000  # random training rows (None = all)
    random_state: int = 0

    def __post_init__(self):
        if self.weather_resampling not in ("step", "interpolate"):
            raise ValueError("weather_resampling must be 'step' or 'interpolate'")

    def to_dict(self) -> dict:
        return asdict(self)
