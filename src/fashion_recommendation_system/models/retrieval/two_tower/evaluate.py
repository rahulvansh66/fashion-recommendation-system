"""Recall@K evaluation for two-tower retrieval."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fashion_recommendation_system.models.retrieval.two_tower.dataset import collate_raw_batch
from fashion_recommendation_system.models.retrieval.two_tower.model import ItemTower, QueryTower
from fashion_recommendation_system.models.retrieval.two_tower.preprocess import (
    PreprocessState,
    encode_batch,
)


def recall_at_k_from_scores(
    scores: torch.Tensor,
    true_indices: torch.Tensor,
    k: int,
) -> float:
    """Compute recall@K from a score matrix and ground-truth corpus indices.

    Args:
        scores: Query-corpus dot products, shape (num_queries, num_corpus).
        true_indices: Index of true item in corpus for each query, shape (num_queries,).
        k: Top-K cutoff.

    Returns:
        Fraction of queries where the true item appears in top-K.
    """
    if scores.numel() == 0:
        return 0.0
    k = min(k, scores.size(1))
    top_k = torch.topk(scores, k=k, dim=1).indices
    hits = (top_k == true_indices.unsqueeze(1)).any(dim=1)
    return float(hits.float().mean().item())


@torch.no_grad()
def embed_candidate_corpus(
    item_tower: ItemTower,
    items_df: pd.DataFrame,
    state: PreprocessState,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Embed deduplicated train articles; return embeddings and article_id -> row index."""
    item_tower.eval()
    article_ids = items_df["article_id"].astype(str).tolist()
    article_to_row = {article_id: idx for idx, article_id in enumerate(article_ids)}

    raw_batch = collate_raw_batch(items_df.to_dict(orient="records"))
    encoded = encode_batch(state, raw_batch)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    embeddings = item_tower(encoded)
    return embeddings, article_to_row


@torch.no_grad()
def evaluate_epoch(
    query_tower: QueryTower,
    item_tower: ItemTower,
    val_loader: DataLoader,
    corpus_embeddings: torch.Tensor,
    article_to_row: dict[str, int],
    state: PreprocessState,
    device: torch.device,
    k: int = 100,
) -> dict[str, float]:
    """Run validation and return recall metrics at several K values."""
    query_tower.eval()
    item_tower.eval()

    all_scores: list[torch.Tensor] = []
    all_labels: list[int] = []

    for raw_batch in val_loader:
        encoded = encode_batch(state, raw_batch)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        query_emb = query_tower(encoded)
        scores = query_emb @ corpus_embeddings.T
        all_scores.append(scores.cpu())

        for article_id in raw_batch["article_id"]:
            all_labels.append(article_to_row[str(article_id)])

    if not all_scores:
        return {"recall_at_100": 0.0}

    scores_tensor = torch.cat(all_scores, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    metrics: dict[str, float] = {}
    for cutoff in (1, 5, 10, 50, 100):
        metrics[f"recall_at_{cutoff}"] = recall_at_k_from_scores(
            scores_tensor, labels_tensor, k=cutoff
        )
    return metrics


def recall_at_100(metrics: dict[str, Any]) -> float:
    """Extract headline recall@100 from evaluation metrics dict."""
    for key in ("recall_at_100", "val_recall_at_100", "top_100_categorical_accuracy"):
        if key in metrics:
            return float(metrics[key])
    for key, value in metrics.items():
        if "100" in key and "recall" in key:
            return float(value)
    return 0.0
