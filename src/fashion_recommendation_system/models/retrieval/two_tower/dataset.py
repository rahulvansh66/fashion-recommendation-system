"""PyTorch dataset builders for two-tower training."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fashion_recommendation_system.models.retrieval.two_tower.split import (
    ALL_FEATURES,
    CANDIDATE_FEATURES,
    QUERY_FEATURES,
    build_vocabularies,
)

__all__ = [
    "QUERY_FEATURES",
    "CANDIDATE_FEATURES",
    "ALL_FEATURES",
    "build_vocabularies",
    "TwoTowerDataset",
    "collate_raw_batch",
    "build_dataloaders",
    "get_unique_items_df",
]


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and cast model feature columns."""
    out = df[ALL_FEATURES].copy()
    for col in ("customer_id", "article_id", "item_category", "index_group_name"):
        out[col] = out[col].astype(str)
    out["age"] = out["age"].astype("float32")
    out["txn_month_sin"] = out["txn_month_sin"].astype("float32")
    out["txn_month_cos"] = out["txn_month_cos"].astype("float32")
    return out


class TwoTowerDataset(Dataset):
    """One purchase row = one (user, item) positive pair."""

    def __init__(self, df: pd.DataFrame) -> None:
        frame = _select_features(df).reset_index(drop=True)
        self._rows: list[dict[str, Any]] = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]


def collate_raw_batch(batch: list[dict[str, Any]]) -> dict[str, list]:
    """Stack list-of-dict samples into batched lists for preprocessing."""
    keys = batch[0].keys()
    return {key: [row[key] for row in batch] for key in keys}


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build shuffled train and batched val DataLoaders."""
    train_loader = DataLoader(
        TwoTowerDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_raw_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        TwoTowerDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_raw_batch,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def get_unique_items_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated candidate rows for evaluation corpus embedding."""
    return _select_features(train_df[CANDIDATE_FEATURES].drop_duplicates("article_id")).reset_index(
        drop=True
    )
