"""Unit tests for FR-BATCH-02 temporal splits."""

from __future__ import annotations

import pandas as pd
import pytest

from fashion_recommendation_system.models.retrieval.two_tower.split import apply_temporal_split


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t_dat": [
                "2020-03-01",
                "2020-03-31",
                "2020-04-01",
                "2020-05-15",
                "2020-05-16",
                "2020-06-30",
                "2020-07-01",
            ],
            "customer_id": ["c1"] * 7,
        }
    )


def test_temporal_split_boundaries(sample_transactions: pd.DataFrame) -> None:
    temporal = {
        "train_end": "2020-03-31",
        "val_start": "2020-04-01",
        "val_end": "2020-05-15",
        "test_start": "2020-05-16",
        "test_end": "2020-06-30",
    }
    train, val, test = apply_temporal_split(sample_transactions, temporal)
    assert len(train) == 2
    assert len(val) == 2
    assert len(test) == 2
    assert train["t_dat"].max() <= pd.Timestamp("2020-03-31")
    assert val["t_dat"].min() >= pd.Timestamp("2020-04-01")
    assert test["t_dat"].min() >= pd.Timestamp("2020-05-16")
