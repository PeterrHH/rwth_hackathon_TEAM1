import numpy as np
import pandas as pd

from models.active_day_ahead import metrics, select_max_distance, temporal_split


def test_temporal_split_keeps_latest_days_for_each_household():
    keys = pd.DataFrame({
        "household_id": [1] * 5 + [2] * 5,
        "date": list(pd.date_range("2024-01-01", periods=5, tz="UTC")) * 2,
    })
    train, test = temporal_split(keys, test_days=2)
    assert set(test) == {3, 4, 8, 9}
    assert not set(train) & set(test)


def test_active_selector_returns_unique_batch():
    rng = np.random.default_rng(3)
    chosen = select_max_distance(rng.normal(size=(50, 8)), batch_size=7, seed=4)
    assert len(chosen) == 7
    assert len(set(chosen)) == 7


def test_active_selector_handles_duplicate_embeddings():
    chosen = select_max_distance(np.zeros((20, 4)), batch_size=5, seed=4)
    assert len(chosen) == 5
    assert len(set(chosen)) == 5


def test_metrics_are_zero_for_perfect_prediction():
    y = np.array([[1.0, 2.0]])
    assert metrics(y, y) == {"mae_kwh_per_15min": 0.0, "rmse_kwh_per_15min": 0.0, "wape": 0.0}
