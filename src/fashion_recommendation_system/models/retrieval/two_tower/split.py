"""Pandas-only data loading and temporal splits (no TensorFlow dependency).

Implements FR-BATCH-02 snap-date temporal splits for two data layouts:

1. **Anchor features** (current FE output): Hive-partitioned ``features/`` table
   with ``snap_date`` and optional ``label``.  Rows are assigned to train/val/test
   by matching ``snap_date`` to ``temporal_split`` snap keys.  When ``label`` is
   present, only ``label == 1`` purchase positives are kept (implicit retrieval
   pairs).

2. **Legacy purchase rows**: transaction table with ``t_dat``; rows are selected
   when ``t_dat`` falls inside a snap's forward label window.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

QUERY_FEATURES = ["customer_id", "age", "txn_month_sin", "txn_month_cos"]
CANDIDATE_FEATURES = ["article_id", "item_category", "index_group_name"]
ALL_FEATURES = QUERY_FEATURES + CANDIDATE_FEATURES


def _rows_for_snap_dates(df: pd.DataFrame, snaps: List[dict]) -> pd.DataFrame:
    """Return anchor rows whose ``snap_date`` matches any snap in ``snaps``.

    When a ``label`` column exists, keep only purchase positives (``label == 1``).
    """
    if not snaps:
        return df.iloc[0:0].copy()
    snap_dates = {pd.Timestamp(s["snap_date"]).normalize() for s in snaps}
    mask = pd.to_datetime(df["snap_date"]).dt.normalize().isin(snap_dates)
    subset = df[mask].copy()
    if "label" in subset.columns:
        subset = subset[subset["label"] == 1].copy()
    return subset


def _rows_in_label_windows(
    df: pd.DataFrame,
    snaps: List[dict],
) -> pd.DataFrame:
    """Return rows whose ``t_dat`` falls inside any snap's label window."""
    if not snaps:
        return df.iloc[0:0].copy()
    combined_mask = pd.Series(False, index=df.index)
    for snap in snaps:
        lo = pd.Timestamp(snap["label_start"])
        hi = pd.Timestamp(snap["label_end"])
        combined_mask |= (df["t_dat"] >= lo) & (df["t_dat"] <= hi)
    return df[combined_mask].copy()


def apply_temporal_split(
    df: pd.DataFrame,
    temporal: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split feature rows into train / val / test by FR-BATCH-02 snap-date scheme.

    Args:
        df: Feature DataFrame with ``snap_date`` (anchor layout) or ``t_dat``
            (legacy purchase rows).  Must contain all columns in ``ALL_FEATURES``.
        temporal: Dict loaded from ``configs/models/two_tower.yaml``
                  ``temporal_split`` section.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    work = df.copy()
    if "snap_date" in work.columns:
        work["snap_date"] = pd.to_datetime(work["snap_date"]).dt.normalize()
        train_df = _rows_for_snap_dates(work, temporal.get("train_snaps", []))
        val_df = _rows_for_snap_dates(work, temporal.get("val_snaps", []))
        test_df = _rows_for_snap_dates(work, temporal.get("test_snaps", []))
        return train_df, val_df, test_df

    if "t_dat" in work.columns:
        work["t_dat"] = pd.to_datetime(work["t_dat"]).dt.normalize()
        train_df = _rows_in_label_windows(work, temporal.get("train_snaps", []))
        val_df = _rows_in_label_windows(work, temporal.get("val_snaps", []))
        test_df = _rows_in_label_windows(work, temporal.get("test_snaps", []))
        return train_df, val_df, test_df

    raise ValueError("Temporal split requires a 'snap_date' or 't_dat' column")


def apply_drift_splits(
    df: pd.DataFrame,
    temporal: dict,
) -> list[tuple[str, pd.DataFrame]]:
    """Return one DataFrame per drift snap (for monitoring / decay plotting)."""
    work = df.copy()
    if "snap_date" in work.columns:
        work["snap_date"] = pd.to_datetime(work["snap_date"]).dt.normalize()
        result = []
        for snap in temporal.get("drift_snaps", []):
            snap_ts = pd.Timestamp(snap["snap_date"]).normalize()
            drift_df = work[work["snap_date"] == snap_ts].copy()
            if "label" in drift_df.columns:
                drift_df = drift_df[drift_df["label"] == 1].copy()
            result.append((snap["snap_date"], drift_df))
        return result

    if "t_dat" not in work.columns:
        raise ValueError("Drift split requires a 'snap_date' or 't_dat' column")

    work["t_dat"] = pd.to_datetime(work["t_dat"]).dt.normalize()
    result = []
    for snap in temporal.get("drift_snaps", []):
        lo = pd.Timestamp(snap["label_start"])
        hi = pd.Timestamp(snap["label_end"])
        mask = (work["t_dat"] >= lo) & (work["t_dat"] <= hi)
        result.append((snap["snap_date"], work[mask].copy()))
    return result


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transaction feature Parquet (file or Hive-partitioned directory."""
    df = pd.read_parquet(path)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def build_vocabularies(train_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build train-only vocabularies for embedding and one-hot layers."""
    return {
        "user_ids": train_df["customer_id"].astype(str).unique().tolist(),
        "item_ids": train_df["article_id"].astype(str).unique().tolist(),
        "item_categories": train_df["item_category"].astype(str).unique().tolist(),
        "index_groups": train_df["index_group_name"].astype(str).unique().tolist(),
    }
