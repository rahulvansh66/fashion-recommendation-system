"""Pandas-only data loading and temporal splits (no TensorFlow dependency).

Implements the FR-BATCH-02 snap-date + forward label window scheme:
  - Each snap defines a feature cutoff (t_dat <= snap_date) and a 7-day label
    window (snap_date + 1 .. snap_date + 7).
  - Train rows come from purchases in any train snap's label window (stacked).
  - Val rows come from purchases in any val snap's label window (stacked).
  - Test rows come from purchases in the test snap's label window.
  - Drift snaps are not returned by apply_temporal_split; use apply_drift_splits
    separately for monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

QUERY_FEATURES = ["customer_id", "age", "txn_month_sin", "txn_month_cos"]
CANDIDATE_FEATURES = ["article_id", "item_category", "index_group_name"]
ALL_FEATURES = QUERY_FEATURES + CANDIDATE_FEATURES


def _rows_in_label_windows(
    df: pd.DataFrame,
    snaps: List[dict],
) -> pd.DataFrame:
    """Return all rows whose ``t_dat`` falls inside any snap's label window.

    Args:
        df: Transaction DataFrame with a normalised ``t_dat`` column.
        snaps: List of dicts with keys ``label_start`` and ``label_end``
               (ISO date strings).

    Returns:
        Subset of ``df`` matching any label window, preserving order.
    """
    if not snaps:
        return df.iloc[0:0].copy()  # empty DataFrame with same schema
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
    """Split transactions into train / val / test by FR-BATCH-02 snap-date scheme.

    Each role is defined by one or more (snap_date, label_start, label_end)
    triples in ``temporal``.  A row belongs to a role when its ``t_dat``
    falls inside any snap's label window for that role.  Train rows from all
    train snaps are stacked into a single DataFrame; val and test follow the
    same logic.

    Args:
        df: Transaction DataFrame.  Must contain ``t_dat`` and all columns in
            ``ALL_FEATURES``.
        temporal: Dict loaded from ``configs/models/two_tower.yaml``
                  ``temporal_split`` section.  Expected keys:
                  ``train_snaps``, ``val_snaps``, ``test_snaps``.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    work = df.copy()
    work["t_dat"] = pd.to_datetime(work["t_dat"]).dt.normalize()

    train_df = _rows_in_label_windows(work, temporal.get("train_snaps", []))
    val_df = _rows_in_label_windows(work, temporal.get("val_snaps", []))
    test_df = _rows_in_label_windows(work, temporal.get("test_snaps", []))
    return train_df, val_df, test_df


def apply_drift_splits(
    df: pd.DataFrame,
    temporal: dict,
) -> list[tuple[str, pd.DataFrame]]:
    """Return one DataFrame per drift snap (for monitoring / decay plotting).

    Args:
        df: Transaction DataFrame with a normalised ``t_dat`` column.
        temporal: Dict with ``drift_snaps`` list.

    Returns:
        List of ``(snap_date_str, drift_df)`` tuples in chronological order.
    """
    work = df.copy()
    work["t_dat"] = pd.to_datetime(work["t_dat"]).dt.normalize()

    result = []
    for snap in temporal.get("drift_snaps", []):
        lo = pd.Timestamp(snap["label_start"])
        hi = pd.Timestamp(snap["label_end"])
        mask = (work["t_dat"] >= lo) & (work["t_dat"] <= hi)
        result.append((snap["snap_date"], work[mask].copy()))
    return result


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transaction feature Parquet (file or directory).

    Args:
        path: Path to a single Parquet file or a directory of Parquet files.

    Returns:
        DataFrame with all required feature columns present.

    Raises:
        ValueError: If any required column is missing.
    """
    df = pd.read_parquet(path)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def build_vocabularies(train_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build train-only vocabularies for embedding and one-hot layers.

    Vocabularies are derived from the stacked train split only so that
    val/test IDs do not leak into the embedding lookup tables.

    Args:
        train_df: Stacked train DataFrame (rows from all train snaps).

    Returns:
        Dict with keys ``user_ids``, ``item_ids``, ``item_categories``,
        ``index_groups`` — each a list of unique string values.
    """
    return {
        "user_ids": train_df["customer_id"].astype(str).unique().tolist(),
        "item_ids": train_df["article_id"].astype(str).unique().tolist(),
        "item_categories": train_df["item_category"].astype(str).unique().tolist(),
        "index_groups": train_df["index_group_name"].astype(str).unique().tolist(),
    }
