"""Unit tests for recall@K evaluation."""

from __future__ import annotations

import torch

from fashion_recommendation_system.models.retrieval.two_tower.evaluate import recall_at_k_from_scores


def test_recall_at_k_perfect_diagonal() -> None:
    scores = torch.eye(5)
    labels = torch.arange(5)
    assert recall_at_k_from_scores(scores, labels, k=1) == 1.0
