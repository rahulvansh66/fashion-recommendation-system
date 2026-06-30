"""Unit tests for two-tower preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from fashion_recommendation_system.models.retrieval.two_tower.preprocess import (
    AgeNormalizer,
    PreprocessState,
    Vocabulary,
    build_preprocess_state,
)


def test_vocabulary_unknown_maps_to_zero() -> None:
    vocab = Vocabulary(["a1", "a2"])
    assert vocab.encode("a1") == 1
    assert vocab.encode("missing") == 0


def test_age_normalizer_zscore() -> None:
    norm = AgeNormalizer.from_series(pd.Series([10.0, 20.0, 30.0]))
    assert norm.normalize(20.0) == pytest.approx(0.0)


def test_build_preprocess_state_from_train() -> None:
    train_df = pd.DataFrame(
        {
            "customer_id": ["c1"],
            "age": [25.0],
            "article_id": ["a1"],
            "item_category": ["cat1"],
            "index_group_name": ["g1"],
        }
    )
    vocabs = {
        "user_ids": ["c1"],
        "item_ids": ["a1"],
        "item_categories": ["cat1"],
        "index_groups": ["g1"],
    }
    state = build_preprocess_state(train_df, vocabs)
    assert isinstance(state, PreprocessState)
    assert state.user_vocab.encode("c1") == 1
