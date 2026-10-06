"""Leakage-safe day-ahead load forecasting with active data selection.

This is a compact adaptation of Aryandoust et al. (2022): candidate days are
embedded by a trained neural network, clustered, and the point furthest from
each cluster centre is labelled/added to training (the paper's ``max d_c``
variant).  Here a label is a complete 96-step household load profile.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STEPS_PER_DAY = 96
LOAD_COLUMN = "kWh_received_Total"
WEATHER_COLUMNS = (
    "Temperature_avg_hourly",
    "Humidity_avg_hourly",
    "Precipitation_total_hourly",
    "Sunshine_duration_hourly",
    "WindSpeed_hourly",
)


@dataclass(frozen=True)
class Config:
    data_dir: Path
    output_dir: Path
    max_households: int = 30
    test_days: int = 28
    initial_fraction: float = 0.10
    budget_fraction: float = 0.35
    active_iterations: int = 4
    hidden_units: int = 64
    max_iter: int = 80
    random_seed: int = 42


def _complete_daily_profiles(path: Path) -> pd.DataFrame:
    """Return complete UTC days as rows and 15-minute values as columns."""
    frame = pd.read_csv(path, sep=";", usecols=["Timestamp", LOAD_COLUMN])
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
    frame[LOAD_COLUMN] = pd.to_numeric(frame[LOAD_COLUMN], errors="coerce")
    frame = frame.dropna().drop_duplicates("Timestamp", keep="last")
    frame["date"] = frame["Timestamp"].dt.floor("D")
    frame["slot"] = frame["Timestamp"].dt.hour * 4 + frame["Timestamp"].dt.minute // 15
    daily = frame.pivot(index="date", columns="slot", values=LOAD_COLUMN)
    daily = daily.reindex(columns=range(STEPS_PER_DAY)).dropna()
    daily.columns = [f"q{i:02d}" for i in range(STEPS_PER_DAY)]
    return daily.sort_index()


def _load_weather(data_dir: Path, weather_ids: Iterable[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for weather_id in sorted(set(weather_ids)):
        path = data_dir / "weather_data_hourly" / f"{weather_id}.csv"
        if not path.exists():
            continue
        weather = pd.read_csv(path, sep=";")
        weather["Timestamp"] = pd.to_datetime(weather["Timestamp"], utc=True)
        available = [c for c in WEATHER_COLUMNS if c in weather]
        for column in available:
            weather[column] = pd.to_numeric(weather[column], errors="coerce")
        weather["date"] = weather["Timestamp"].dt.floor("D")
        aggregations = {column: "mean" for column in available}
        if "Precipitation_total_hourly" in aggregations:
            aggregations["Precipitation_total_hourly"] = "sum"
        if "Sunshine_duration_hourly" in aggregations:
            aggregations["Sunshine_duration_hourly"] = "sum"
        result[weather_id] = weather.groupby("date")[available].agg(aggregations)
    return result


def build_dataset(data_dir: Path, max_households: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build samples without using any target-day meter reading as a feature."""
    households = pd.read_csv(data_dir / "smart_meter_meta_data" / "households.csv", sep=";")
    metadata = pd.read_csv(data_dir / "smart_meter_meta_data" / "meta_data.csv", sep=";")
    households = households.merge(metadata, on="Household_ID", how="left")
    overview = pd.read_csv(
        data_dir / "smart_meter_meta_data" / "smart_meter_data_15min_overview.csv", sep=";"
    )
    count_col = next((c for c in overview if "count" in c.lower() or "point" in c.lower()), None)
    if count_col:
        overview[count_col] = pd.to_numeric(overview[count_col], errors="coerce")
        order = overview.sort_values(count_col, ascending=False)["Household_ID"]
        households = households.set_index("Household_ID").loc[
            [x for x in order if x in set(households["Household_ID"])]
        ].reset_index()
    households = households.head(max_households)
    weather_by_id = _load_weather(data_dir, households["Weather_ID"].dropna().astype(str))

    feature_rows: list[dict[str, object]] = []
    targets: list[np.ndarray] = []
    keys: list[dict[str, object]] = []
    static_cols = [
        "Installation_HasPVSystem", "Survey_Building_LivingArea", "Survey_Building_Residents",
        "Survey_HeatDistribution_System_FloorHeating", "Survey_HeatDistribution_System_Radiator",
        "Survey_DHW_Production_ByHeatPump", "Survey_Installation_HasElectricVehicle",
    ]

    for row in households.itertuples(index=False):
        household_id = int(row.Household_ID)
        meter_path = data_dir / "15min" / f"{household_id}.csv"
        if not meter_path.exists():
            continue
        daily = _complete_daily_profiles(meter_path)
        weather = weather_by_id.get(str(row.Weather_ID))
        for date in daily.index[1:]:
            previous_date = date - pd.Timedelta(days=1)
            if previous_date not in daily.index:
                continue
            values: dict[str, object] = {
                "dow_sin": np.sin(2 * np.pi * date.dayofweek / 7),
                "dow_cos": np.cos(2 * np.pi * date.dayofweek / 7),
                "year_sin": np.sin(2 * np.pi * date.dayofyear / 365.25),
                "year_cos": np.cos(2 * np.pi * date.dayofyear / 365.25),
                "is_weekend": float(date.dayofweek >= 5),
                "household_id": str(household_id),
            }
            values.update({f"lag1_{c}": v for c, v in daily.loc[previous_date].items()})
            for column in static_cols:
                values[column] = getattr(row, column, np.nan)
            if weather is not None and date in weather.index:
                values.update({f"weather_{c}": v for c, v in weather.loc[date].items()})
            feature_rows.append(values)
            targets.append(daily.loc[date].to_numpy(dtype=np.float32))
            keys.append({"household_id": household_id, "date": date})

    if not targets:
        raise ValueError("No consecutive complete meter days were found.")
    X = pd.DataFrame(feature_rows)
    X = pd.get_dummies(X, columns=["household_id"], dtype=float)
    X = X.replace({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, np.stack(targets), pd.DataFrame(keys)


def temporal_split(keys: pd.DataFrame, test_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the newest N available days per household for honest testing."""
    test = np.zeros(len(keys), dtype=bool)
    for _, indices in keys.groupby("household_id").groups.items():
        ordered = keys.loc[indices].sort_values("date").index.to_numpy()
        n_test = min(test_days, max(1, len(ordered) - 1))
        test[ordered[-n_test:]] = True
    return np.flatnonzero(~test), np.flatnonzero(test)


def _new_model(config: Config) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(config.hidden_units,), activation="relu", solver="adam",
            max_iter=config.max_iter, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=8, random_state=config.random_seed,
        )),
    ])


def hidden_embedding(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return the MLP hidden representation used by active selection."""
    transformed = model.named_steps["imputer"].transform(X)
    transformed = model.named_steps["scaler"].transform(transformed)
    mlp = model.named_steps["mlp"]
    return np.maximum(0.0, transformed @ mlp.coefs_[0] + mlp.intercepts_[0])


def select_max_distance(embedding: np.ndarray, batch_size: int, seed: int) -> np.ndarray:
    """Select one diverse, high-uncertainty point per embedding cluster."""
    if batch_size >= len(embedding):
        return np.arange(len(embedding))
    clusters = MiniBatchKMeans(
        n_clusters=batch_size, random_state=seed, n_init=3, batch_size=min(1024, len(embedding))
    ).fit(embedding)
    chosen: list[int] = []
    for cluster_id in range(batch_size):
        members = np.flatnonzero(clusters.labels_ == cluster_id)
        if not len(members):
            continue
        distances = np.linalg.norm(embedding[members] - clusters.cluster_centers_[cluster_id], axis=1)
        chosen.append(int(members[np.argmax(distances)]))
    if len(chosen) < batch_size:
        own_centres = clusters.cluster_centers_[clusters.labels_]
        distances = np.linalg.norm(embedding - own_centres, axis=1)
        distances[np.asarray(chosen, dtype=int)] = -np.inf
        fill = np.argsort(distances)[-(batch_size - len(chosen)):]
        chosen.extend(fill.tolist())
    return np.asarray(chosen, dtype=int)


def active_fit(X: pd.DataFrame, y: np.ndarray, config: Config) -> tuple[Pipeline, np.ndarray]:
    """Fit using a chronological seed and feature-embedding query batches."""
    n = len(X)
    budget = min(n, max(2, int(np.ceil(config.budget_fraction * n))))
    seed_size = min(budget, max(2, int(np.ceil(config.initial_fraction * n))))
    selected = list(range(seed_size))
    remaining = list(range(seed_size, n))
    rng = np.random.default_rng(config.random_seed)

    for iteration in range(config.active_iterations + 1):
        model = _new_model(config)
        model.fit(X.iloc[selected], y[selected])
        if len(selected) >= budget or not remaining or iteration == config.active_iterations:
            break
        remaining_budget = budget - len(selected)
        iterations_left = config.active_iterations - iteration
        batch_size = min(len(remaining), max(1, int(np.ceil(remaining_budget / iterations_left))))
        candidate = np.asarray(remaining)
        # Subsampling is a documented speed/accuracy trade-off in the paper.
        max_candidates = max(5_000, batch_size * 20)
        if len(candidate) > max_candidates:
            candidate = rng.choice(candidate, size=max_candidates, replace=False)
        local = select_max_distance(hidden_embedding(model, X.iloc[candidate]), batch_size, config.random_seed + iteration)
        picked = candidate[local].tolist()
        selected.extend(picked)
        picked_set = set(picked)
        remaining = [i for i in remaining if i not in picked_set]
    return model, np.asarray(selected, dtype=int)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    absolute = np.abs(y_true - y_pred)
    return {
        "mae_kwh_per_15min": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kwh_per_15min": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "wape": float(absolute.sum() / max(np.abs(y_true).sum(), 1e-9)),
    }


def run(config: Config) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    X, y, keys = build_dataset(config.data_dir, config.max_households)
    train_idx, test_idx = temporal_split(keys, config.test_days)
    order = np.argsort(keys.loc[train_idx, "date"].to_numpy())
    train_idx = train_idx[order]
    model, selected_local = active_fit(X.iloc[train_idx].reset_index(drop=True), y[train_idx], config)
    selected_global = train_idx[selected_local]
    prediction = np.maximum(0.0, model.predict(X.iloc[test_idx]))
    baseline = X.iloc[test_idx][[f"lag1_q{i:02d}" for i in range(STEPS_PER_DAY)]].to_numpy()

    report: dict[str, object] = {
        "config": {**asdict(config), "data_dir": str(config.data_dir), "output_dir": str(config.output_dir)},
        "samples": {"all": len(X), "train_pool": len(train_idx), "selected": len(selected_global), "test": len(test_idx)},
        "active_model": metrics(y[test_idx], prediction),
        "previous_day_baseline": metrics(y[test_idx], baseline),
    }
    joblib.dump({"model": model, "feature_columns": X.columns.tolist(), "config": report["config"]}, config.output_dir / "active_day_ahead.joblib")
    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    pd.concat([keys.loc[selected_global].reset_index(drop=True), pd.DataFrame({"selection_order": range(len(selected_global))})], axis=1).to_csv(
        config.output_dir / "selected_training_days.csv", index=False
    )
    actual_columns = [f"actual_q{i:02d}" for i in range(STEPS_PER_DAY)]
    prediction_columns = [f"prediction_q{i:02d}" for i in range(STEPS_PER_DAY)]
    pred_rows = pd.concat([
        keys.loc[test_idx].reset_index(drop=True),
        pd.DataFrame(y[test_idx], columns=actual_columns),
        pd.DataFrame(prediction, columns=prediction_columns),
    ], axis=1)
    pred_rows.to_csv(config.output_dir / "test_predictions.csv", index=False)
    return report


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "active_day_ahead")
    parser.add_argument("--max-households", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--initial-fraction", type=float, default=0.10)
    parser.add_argument("--budget-fraction", type=float, default=0.35)
    parser.add_argument("--active-iterations", type=int, default=4)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
