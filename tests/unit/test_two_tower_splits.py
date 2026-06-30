"""Unit tests for FR-BATCH-02 snap-date temporal splits."""

from __future__ import annotations

import pandas as pd
import pytest

from fashion_recommendation_system.models.retrieval.two_tower.split import (
    apply_drift_splits,
    apply_temporal_split,
)


# Dates that fall inside each snap's label window
# Train snap 1 (Mar 31): label window Apr 1–7
# Train snap 2 (Apr 7): label window Apr 8–14
# Val snap 1 (Apr 14): label window Apr 15–21
# Val snap 2 (Apr 28): label window Apr 29–May 5
# Test snap (May 15): label window May 16–22
# Drift snap 1 (May 31): label window Jun 1–7
SAMPLE_DATES = [
    "2020-03-31",  # before any label window -> excluded from all roles
    "2020-04-03",  # train snap 1 label window
    "2020-04-10",  # train snap 2 label window
    "2020-04-18",  # val snap 1 label window
    "2020-05-01",  # val snap 2 label window
    "2020-05-17",  # test snap label window
    "2020-06-04",  # drift snap 1 label window
    "2020-07-15",  # outside all label windows
]

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
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t_dat": SAMPLE_DATES,
            "customer_id": ["c1"] * len(SAMPLE_DATES),
        }
    )


def test_train_rows_come_from_label_windows(sample_transactions: pd.DataFrame) -> None:
    """Train DataFrame contains only rows from train snap label windows."""
    train, _val, _test = apply_temporal_split(sample_transactions, SNAP_CONFIG)
    # Rows: 2020-04-03 (snap1 window), 2020-04-10 (snap2 window)
    assert len(train) == 2
    dates = pd.to_datetime(train["t_dat"]).dt.normalize().tolist()
    assert pd.Timestamp("2020-04-03") in dates
    assert pd.Timestamp("2020-04-10") in dates


def test_val_rows_come_from_val_label_windows(sample_transactions: pd.DataFrame) -> None:
    """Val DataFrame contains rows from both val snap label windows."""
    _train, val, _test = apply_temporal_split(sample_transactions, SNAP_CONFIG)
    assert len(val) == 2
    dates = pd.to_datetime(val["t_dat"]).dt.normalize().tolist()
    assert pd.Timestamp("2020-04-18") in dates
    assert pd.Timestamp("2020-05-01") in dates


def test_test_rows_come_from_test_label_window(sample_transactions: pd.DataFrame) -> None:
    """Test DataFrame contains only rows from the test snap label window."""
    _train, _val, test = apply_temporal_split(sample_transactions, SNAP_CONFIG)
    assert len(test) == 1
    assert pd.to_datetime(test["t_dat"]).dt.normalize().iloc[0] == pd.Timestamp("2020-05-17")


def test_no_leakage_between_roles(sample_transactions: pd.DataFrame) -> None:
    """Train, val, and test DataFrames have no overlapping rows."""
    train, val, test = apply_temporal_split(sample_transactions, SNAP_CONFIG)
    train_dates = set(pd.to_datetime(train["t_dat"]).dt.normalize())
    val_dates = set(pd.to_datetime(val["t_dat"]).dt.normalize())
    test_dates = set(pd.to_datetime(test["t_dat"]).dt.normalize())
    assert train_dates.isdisjoint(val_dates), "train and val windows overlap"
    assert train_dates.isdisjoint(test_dates), "train and test windows overlap"
    assert val_dates.isdisjoint(test_dates), "val and test windows overlap"


def test_rows_outside_windows_are_excluded(sample_transactions: pd.DataFrame) -> None:
    """Rows whose t_dat falls outside any label window are not in any split."""
    train, val, test = apply_temporal_split(sample_transactions, SNAP_CONFIG)
    all_selected = pd.concat([train, val, test])
    all_dates = set(pd.to_datetime(all_selected["t_dat"]).dt.normalize())
    # 2020-03-31 and 2020-07-15 are outside all label windows
    assert pd.Timestamp("2020-03-31") not in all_dates
    assert pd.Timestamp("2020-07-15") not in all_dates


def test_drift_splits_return_correct_count(sample_transactions: pd.DataFrame) -> None:
    """apply_drift_splits returns one entry per drift snap."""
    drifts = apply_drift_splits(sample_transactions, SNAP_CONFIG)
    assert len(drifts) == 2  # 2 drift snaps in SNAP_CONFIG


def test_drift_split_first_snap_correct_date(sample_transactions: pd.DataFrame) -> None:
    """First drift snap contains the row in its label window."""
    drifts = apply_drift_splits(sample_transactions, SNAP_CONFIG)
    snap_date, drift_df = drifts[0]
    assert snap_date == "2020-05-31"
    # 2020-06-04 is in [Jun 1, Jun 7]
    assert len(drift_df) == 1
    assert pd.to_datetime(drift_df["t_dat"]).dt.normalize().iloc[0] == pd.Timestamp("2020-06-04")
