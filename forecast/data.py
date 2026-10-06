"""Household data, the standard target cleaning and the fixed ground truth."""

import pandas as pd

from .config import (CACHE_DIR, DATA_DIR, MAX_KWH_PER_15MIN, META_DIR, MIN_DAYS_OF_DATA,
                     TEST_START)

CATEGORICAL = ["Survey_Building_Type", "Survey_HeatPump_Installation_Type"]
NUMERIC = ["Survey_Building_LivingArea", "Survey_Building_Residents"]


def valid_household_ids() -> list[int]:
    """Households with > MIN_DAYS_OF_DATA days of readings and a metadata row."""
    overview = pd.read_csv(META_DIR / "smart_meter_data_15min_overview.csv", sep=";")
    meta = pd.read_csv(META_DIR / "meta_data.csv", sep=";")
    duration = (pd.to_datetime(overview["SMD_15min_TimeAvailable_LatestTimestamp"])
                - pd.to_datetime(overview["SMD_15min_TimeAvailable_EarliestTimestamp"]))
    long_enough = overview.loc[duration.dt.total_seconds() / 86400 > MIN_DAYS_OF_DATA, "Household_ID"]
    overview["duration_days"] = duration.dt.total_seconds() / 86400  # first to last reading, in days

    overview["coverage_ratio"] = (
        overview["SMD_15min_TimeAvailable_NumberDays"] / overview["duration_days"]
    )

    # Filter households: More than 1 year of active duration AND at least 80% data coverage
    reliable_households = overview[
        (overview["duration_days"] > MIN_DAYS_OF_DATA) & 
        (overview["coverage_ratio"] >= 0.80)
    ].copy()

    # return sorted(set(long_enough) & set(meta["Household_ID"]))
    print(f"Number of reliable households: {len(sorted(set(reliable_households['Household_ID']) & set(meta['Household_ID'])))}")
    return sorted(set(reliable_households["Household_ID"]) & set(meta["Household_ID"]))


def read_households() -> pd.DataFrame:
    """One row per household: weather station, group, PV flag and survey answers as model-ready columns."""
    households = pd.read_csv(META_DIR / "households.csv", sep=";")
    meta = pd.read_csv(META_DIR / "meta_data.csv", sep=";")
    hh = households[["Household_ID", "Group", "Weather_ID", "Installation_HasPVSystem"]].merge(
        meta, on="Household_ID", how="left")
    hh["is_treatment"] = (hh["Group"] == "treatment").astype("int8")
    hh["has_pv"] = hh["Installation_HasPVSystem"].map({True: 1.0, False: 0.0})  # unknown stays NaN
    for col in CATEGORICAL:
        hh[col] = hh[col].astype("category")
    for col in survey_flag_columns(meta):
        hh[col] = hh[col].astype("float32")  # True / False / unanswered -> 1 / 0 / NaN
    return hh.drop(columns=["Group", "Installation_HasPVSystem"])


def survey_flag_columns(meta: pd.DataFrame | None = None) -> list[str]:
    """The True/False survey columns (heat distribution, hot water, appliances)."""
    if meta is None:
        meta = pd.read_csv(META_DIR / "meta_data.csv", sep=";", nrows=1)
    return [c for c in meta.columns if c.startswith("Survey_") and c not in CATEGORICAL + NUMERIC]


def load_clean() -> tuple[pd.DataFrame, list[int]]:
    """15-min readings of the valid households after the standard target cleaning.

    Returns (load, household_ids): the readings, and the IDs of the households that actually have
    readings (valid households with no Total data at all drop out).
    Columns of load: Household_ID, Timestamp (UTC, interval start), kwh, AffectsTimePoint.
    Missing readings and readings > MAX_KWH_PER_15MIN are dropped; nothing is imputed.
    Cached in cache/load_clean.pkl (delete it if the benchmark rules change).
    """
    path = CACHE_DIR / "load_clean.pkl"
    if path.exists():
        load = pd.read_pickle(path)
        return load, sorted(load["Household_ID"].unique().tolist())
    valid_household_ids_list = valid_household_ids()
    frames = [
        # columns are read by name: their order differs between files
        pd.read_csv(DATA_DIR / "15min" / f"{hid}.csv", sep=";",
                    usecols=["Household_ID", "Timestamp", "AffectsTimePoint", "kWh_received_Total"])
        for hid in valid_household_ids_list
    ]
    load = pd.concat(frames, ignore_index=True).rename(columns={"kWh_received_Total": "kwh"})
    load = load[load["kwh"].notna() & (load["kwh"] <= MAX_KWH_PER_15MIN)]
    load = load.assign(
        Household_ID=load["Household_ID"].astype("int32"),
        Timestamp=pd.to_datetime(load["Timestamp"], format="%Y-%m-%d %H:%M:%S%z"),
        kwh=load["kwh"].astype("float32"),
        AffectsTimePoint=load["AffectsTimePoint"].astype("category"),
    ).sort_values(["Household_ID", "Timestamp"]).reset_index(drop=True)

    CACHE_DIR.mkdir(exist_ok=True)
    load.to_pickle(path)
    return load, sorted(load["Household_ID"].unique().tolist())


def ground_truth() -> pd.DataFrame:
    """The benchmark: every cleaned reading in the test period. Every experiment predicts exactly these rows."""
    load, _ = load_clean()
    return load.loc[load["Timestamp"] >= TEST_START, ["Household_ID", "Timestamp", "kwh"]].reset_index(drop=True)
