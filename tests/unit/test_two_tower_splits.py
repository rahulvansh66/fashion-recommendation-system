"""Unit tests for FR-BATCH-02 snap-date temporal splits."""

from __future__ import annotations

import pandas as pd
import pytest

from fashion_recommendation_system.models.retrieval.two_tower.split import (
    apply_drift_splits,
    apply_temporal_split,
)


SNAP_CONFIG = {
    "train_snaps": [
        {"snap_date": "2020-03-31", "label_start": "2020-04-01", "label_end": "2020-04-07"},
        {"snap_date": "2020-04-07", "label_start": "2020-04-08", "label_end": "2020-04-14"},
    ],
    "val_snaps": [
        {"snap_date": "2020-04-14", "label_start": "2020-04-15", "label_end": "2020-04-21"},
        {"snap_date": "2020-04-28", "label_start": "2020-04-29", "label_end": "2020-05-05"},
    ],
    "test_snaps": [
        {"snap_date": "2020-05-15", "label_start": "2020-05-16", "label_end": "2020-05-22"},
    ],
    "drift_snaps": [
        {"snap_date": "2020-05-31", "label_start": "2020-06-01", "label_end": "2020-06-07"},
        {"snap_date": "2020-06-30", "label_start": "2020-07-01", "label_end": "2020-07-07"},
    ],
}


@pytest.fixture
def anchor_features() -> pd.DataFrame:
    """Anchor rows partitioned by snap_date (current FE layout)."""
    rows = [
        ("2020-03-24", 1),
        ("2020-03-31", 1),
        ("2020-03-31", 0),
        ("2020-04-07", 1),
        ("2020-04-14", 1),
        ("2020-04-28", 1),
        ("2020-05-15", 1),
        ("2020-05-31", 1),
        ("2020-06-30", 1),
        ("2020-09-15", 1),
    ]
    return pd.DataFrame(
        {
            "snap_date": [r[0] for r in rows],
            "label": [r[1] for r in rows],
            "customer_id": ["c1"] * len(rows),
        }
    )


@pytest.fixture
def legacy_purchases() -> pd.DataFrame:
    """Legacy purchase rows keyed by t_dat."""
    dates = [
        "2020-03-31",
        "2020-04-03",
        "2020-04-10",
        "2020-04-18",
        "2020-05-01",
        "2020-05-17",
        "2020-06-04",
        "2020-07-15",
    ]
    return pd.DataFrame({"t_dat": dates, "customer_id": ["c1"] * len(dates)})


def test_anchor_train_rows_match_train_snaps(anchor_features: pd.DataFrame) -> None:
    train, _val, _test = apply_temporal_split(anchor_features, SNAP_CONFIG)
    assert len(train) == 2
    assert set(train["snap_date"].dt.strftime("%Y-%m-%d")) == {"2020-03-31", "2020-04-07"}


def test_anchor_split_excludes_label_zero(anchor_features: pd.DataFrame) -> None:
    train, _val, _test = apply_temporal_split(anchor_features, SNAP_CONFIG)
    assert (train["label"] == 1).all()


def test_anchor_val_and_test_snap_dates(anchor_features: pd.DataFrame) -> None:
    _train, val, test = apply_temporal_split(anchor_features, SNAP_CONFIG)
    assert set(val["snap_date"].dt.strftime("%Y-%m-%d")) == {"2020-04-14", "2020-04-28"}
    assert len(test) == 1
    assert test["snap_date"].dt.strftime("%Y-%m-%d").iloc[0] == "2020-05-15"


def test_anchor_snaps_outside_config_are_excluded(anchor_features: pd.DataFrame) -> None:
    train, val, test = apply_temporal_split(anchor_features, SNAP_CONFIG)
    selected = pd.concat([train, val, test])
    assert "2020-03-24" not in selected["snap_date"].dt.strftime("%Y-%m-%d").tolist()
    assert "2020-09-15" not in selected["snap_date"].dt.strftime("%Y-%m-%d").tolist()


def test_legacy_train_rows_come_from_label_windows(legacy_purchases: pd.DataFrame) -> None:
    train, _val, _test = apply_temporal_split(legacy_purchases, SNAP_CONFIG)
    assert len(train) == 2
    dates = pd.to_datetime(train["t_dat"]).dt.normalize().tolist()
    assert pd.Timestamp("2020-04-03") in dates
    assert pd.Timestamp("2020-04-10") in dates


def test_legacy_no_leakage_between_roles(legacy_purchases: pd.DataFrame) -> None:
    train, val, test = apply_temporal_split(legacy_purchases, SNAP_CONFIG)
    train_dates = set(pd.to_datetime(train["t_dat"]).dt.normalize())
    val_dates = set(pd.to_datetime(val["t_dat"]).dt.normalize())
    test_dates = set(pd.to_datetime(test["t_dat"]).dt.normalize())
    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)


def test_anchor_drift_splits(anchor_features: pd.DataFrame) -> None:
    drifts = apply_drift_splits(anchor_features, SNAP_CONFIG)
    assert len(drifts) == 2
    assert drifts[0][0] == "2020-05-31"
    assert len(drifts[0][1]) == 1


def test_legacy_drift_splits(legacy_purchases: pd.DataFrame) -> None:
    drifts = apply_drift_splits(legacy_purchases, SNAP_CONFIG)
    assert len(drifts) == 2
    assert drifts[0][0] == "2020-05-31"
    assert len(drifts[0][1]) == 1
