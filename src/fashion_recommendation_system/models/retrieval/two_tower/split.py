"""Pandas-only data loading and temporal splits (no TensorFlow dependency)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

QUERY_FEATURES = ["customer_id", "age", "txn_month_sin", "txn_month_cos"]
CANDIDATE_FEATURES = ["article_id", "item_category", "index_group_name"]
ALL_FEATURES = QUERY_FEATURES + CANDIDATE_FEATURES


def apply_temporal_split(
    df: pd.DataFrame,
    temporal: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split transactions by FR-BATCH-02 date windows on ``t_dat``."""
    work = df.copy()
    work["t_dat"] = pd.to_datetime(work["t_dat"]).dt.normalize()

    train_end = pd.Timestamp(temporal["train_end"])
    val_start = pd.Timestamp(temporal["val_start"])
    val_end = pd.Timestamp(temporal["val_end"])
    test_start = pd.Timestamp(temporal["test_start"])
    test_end = pd.Timestamp(temporal["test_end"])

    train_df = work[work["t_dat"] <= train_end]
    val_df = work[(work["t_dat"] >= val_start) & (work["t_dat"] <= val_end)]
    test_df = work[(work["t_dat"] >= test_start) & (work["t_dat"] <= test_end)]
    return train_df, val_df, test_df


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transaction feature Parquet (file or directory)."""
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
