"""In-batch contrastive loss with log-q popularity correction."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

from fashion_recommendation_system.models.retrieval.two_tower.preprocess import Vocabulary


def build_article_prob_map(
    train_df: pd.DataFrame,
    item_vocab: Vocabulary,
) -> dict[int, float]:
    """Build P(article_id) keyed by embedding index from train counts."""
    counts = train_df.groupby("article_id").size()
    total = len(train_df)
    prob_map: dict[int, float] = {}
    for article_id, count in counts.items():
        prob_map[item_vocab.encode(str(article_id))] = float(count) / float(total)
    return prob_map


def popularity_corrected_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    article_indices: torch.Tensor,
    prob_map: dict[int, float],
    default_prob: float = 1e-8,
) -> torch.Tensor:
    """In-batch softmax cross-entropy with log-q debiasing (training only).

    Args:
        user_emb: Query embeddings, shape (batch, dim).
        item_emb: Item embeddings, shape (batch, dim).
        article_indices: Article embedding indices for each row, shape (batch,).
        prob_map: P(article) keyed by embedding index.
        default_prob: Fallback probability for missing indices.

    Returns:
        Scalar mean loss.
    """
    logits = user_emb @ item_emb.T
    col_probs = torch.tensor(
        [prob_map.get(int(idx), default_prob) for idx in article_indices.tolist()],
        device=logits.device,
        dtype=logits.dtype,
    )
    corrected_logits = logits - torch.log(col_probs).unsqueeze(0)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(corrected_logits, labels)
