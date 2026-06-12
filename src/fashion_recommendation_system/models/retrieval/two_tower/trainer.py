"""Training loop for the two-tower retrieval model."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from fashion_recommendation_system.models.retrieval.two_tower.dataset import (
    build_tf_datasets,
    get_unique_items_dataset,
)
from fashion_recommendation_system.models.retrieval.two_tower.model import TwoTowerModel
from fashion_recommendation_system.models.retrieval.two_tower.popularity import (
    build_label_probs_table,
)
from fashion_recommendation_system.models.retrieval.two_tower.towers import ItemTower, QueryTower


def build_model(
    train_df,
    *,
    embedding_dim: int,
    batch_size: int,
    vocabs: dict[str, list[str]] | None = None,
) -> TwoTowerModel:
    """Construct towers, popularity table, and FactorizedTopK task."""
    if vocabs is None:
        from fashion_recommendation_system.models.retrieval.two_tower.split import (
            build_vocabularies,
        )

        vocabs = build_vocabularies(train_df)

    query_model = QueryTower(vocabs["user_ids"], embedding_dim)
    item_model = ItemTower(
        vocabs["item_ids"],
        vocabs["item_categories"],
        vocabs["index_groups"],
        embedding_dim,
    )
    label_probs_table = build_label_probs_table(train_df, item_model.article_lookup)
    item_ds = get_unique_items_dataset(train_df)
    return TwoTowerModel(
        query_model=query_model,
        item_model=item_model,
        item_ds=item_ds,
        batch_size=batch_size,
        label_probs_table=label_probs_table,
    )


def _warm_start(query_model: QueryTower, train_ds: tf.data.Dataset) -> None:
    """Adapt age normalization and build lookup tables."""
    query_model.normalized_age.adapt(train_ds.map(lambda x: x["age"]))
    for batch in train_ds.take(1):
        query_model(batch)
        break


def train_model(
    train_df,
    val_df,
    *,
    embedding_dim: int = 16,
    batch_size: int = 2048,
    epochs: int = 10,
    learning_rate: float = 0.01,
    weight_decay: float = 0.001,
) -> tuple[TwoTowerModel, dict[str, Any]]:
    """Train the two-tower model and return history dict."""
    train_ds, val_ds = build_tf_datasets(train_df, val_df, batch_size)
    model = build_model(train_df, embedding_dim=embedding_dim, batch_size=batch_size)
    _warm_start(model.query_model, train_ds)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(optimizer=optimizer)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )
    return model, history.history


def recall_at_100(metrics: dict[str, float]) -> float:
    """Extract recall@100 from FactorizedTopK metric names."""
    for key in ("top_100_categorical_accuracy", "factorized_top_k/top_100_categorical_accuracy"):
        if key in metrics:
            return metrics[key]
    # Fallback: first metric containing top_100
    for key, value in metrics.items():
        if "top_100" in key:
            return value
    return 0.0
