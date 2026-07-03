"""Unit tests for log-q article probability map."""

from __future__ import annotations

import pandas as pd
import pytest

from fashion_recommendation_system.models.retrieval.two_tower.loss import build_article_prob_map
from fashion_recommendation_system.models.retrieval.two_tower.preprocess import Vocabulary


def test_article_prob_map_sums_to_one() -> None:
    train_df = pd.DataFrame({"article_id": ["a1", "a1", "a2"]})
    vocab = Vocabulary(["a1", "a2"])
    prob_map = build_article_prob_map(train_df, vocab)
    assert prob_map[vocab.encode("a1")] == pytest.approx(2 / 3)
    assert prob_map[vocab.encode("a2")] == pytest.approx(1 / 3)
    assert sum(prob_map.values()) == pytest.approx(1.0)
