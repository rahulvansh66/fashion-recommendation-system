"""Unit tests for popularity-corrected in-batch loss."""

from __future__ import annotations

import torch

from fashion_recommendation_system.models.retrieval.two_tower.loss import popularity_corrected_loss


def test_aligned_loss_lower_than_shuffled() -> None:
    """Correct pairings should yield lower loss than shuffled item embeddings."""
    batch_size, dim = 8, 16
    torch.manual_seed(0)
    user_emb = torch.randn(batch_size, dim)
    item_emb = user_emb.clone()
    article_indices = torch.arange(1, batch_size + 1)
    prob_map = {i: 1.0 / batch_size for i in range(1, batch_size + 1)}

    aligned_loss = popularity_corrected_loss(user_emb, item_emb, article_indices, prob_map)
    shuffled_loss = popularity_corrected_loss(
        user_emb, item_emb[torch.randperm(batch_size)], article_indices, prob_map
    )
    assert aligned_loss.item() < shuffled_loss.item()
