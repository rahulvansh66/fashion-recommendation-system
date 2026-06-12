"""TensorFlow dataset builders for two-tower training."""

from __future__ import annotations

import pandas as pd
import tensorflow as tf

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
    "build_tf_datasets",
    "get_unique_items_dataset",
    "df_to_dataset",
]


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[ALL_FEATURES].copy()
    for col in ("customer_id", "article_id", "item_category", "index_group_name"):
        out[col] = out[col].astype(str)
    out["age"] = out["age"].astype("float32")
    out["txn_month_sin"] = out["txn_month_sin"].astype("float32")
    out["txn_month_cos"] = out["txn_month_cos"].astype("float32")
    return out


def df_to_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    """Convert a feature DataFrame to a columnar ``tf.data.Dataset``."""
    frame = _select_features(df)
    return tf.data.Dataset.from_tensor_slices({col: frame[col].values for col in ALL_FEATURES})


def build_tf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build batched train (shuffled) and val datasets."""
    train_ds = (
        df_to_dataset(train_df)
        .batch(batch_size)
        .cache()
        .shuffle(batch_size * 10)
    )
    val_ds = df_to_dataset(val_df).batch(batch_size).cache()
    return train_ds, val_ds


def get_unique_items_dataset(train_df: pd.DataFrame) -> tf.data.Dataset:
    """Deduplicated candidate rows for FactorizedTopK evaluation corpus."""
    items = _select_features(train_df[CANDIDATE_FEATURES].drop_duplicates("article_id"))
    return tf.data.Dataset.from_tensor_slices(
        {col: items[col].values for col in CANDIDATE_FEATURES}
    )
