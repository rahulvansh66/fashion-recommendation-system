"""Unit tests for log-q popularity table."""

from __future__ import annotations

import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from fashion_recommendation_system.models.retrieval.two_tower.popularity import (
    build_label_probs_table,
)
from fashion_recommendation_system.models.retrieval.two_tower.towers import ItemTower


def test_label_probs_sum_to_one_on_train_items() -> None:
    train_df = pd.DataFrame(
        {
            "article_id": ["a1", "a1", "a2"],
            "item_category": ["cat1", "cat1", "cat2"],
            "index_group_name": ["g1", "g1", "g2"],
        }
    )
    item_model = ItemTower(
        item_ids=["a1", "a2"],
        item_categories=["cat1", "cat2"],
        index_groups=["g1", "g2"],
        emb_dim=4,
    )
    table = build_label_probs_table(train_df, item_model.article_lookup)
    keys = tf.constant(["a1", "a2"], dtype=tf.string)
    indices = item_model.article_lookup(keys)
    probs = table.lookup(indices).numpy()
    assert probs[0] == pytest.approx(2 / 3)
    assert probs[1] == pytest.approx(1 / 3)
