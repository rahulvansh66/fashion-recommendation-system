"""Checkpoint save/load for two-tower PyTorch models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from fashion_recommendation_system.models.retrieval.two_tower.model import ItemTower, QueryTower
from fashion_recommendation_system.models.retrieval.two_tower.preprocess import PreprocessState


def save_artifacts(
    query_tower: QueryTower,
    item_tower: ItemTower,
    state: PreprocessState,
    out_dir: Path,
    metrics: dict[str, Any],
) -> None:
    """Persist tower weights, preprocessing state, and metrics JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(query_tower.state_dict(), out_dir / "query_tower.pt")
    torch.save(item_tower.state_dict(), out_dir / "candidate_tower.pt")
    (out_dir / "preprocess_state.json").write_text(state.to_json(), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_artifacts(
    model_dir: Path,
    device: torch.device | None = None,
) -> tuple[QueryTower, ItemTower, PreprocessState]:
    """Load towers and preprocessing state from a checkpoint directory."""
    if device is None:
        device = torch.device("cpu")

    state = PreprocessState.from_json((model_dir / "preprocess_state.json").read_text(encoding="utf-8"))
    query_tower = QueryTower(state.user_vocab.num_embeddings - 1, _embedding_dim_from_state_dict(
        model_dir / "query_tower.pt"
    ))
    item_tower = ItemTower(
        num_items=state.item_vocab.num_embeddings - 1,
        num_categories=state.category_vocab.size,
        num_index_groups=state.index_group_vocab.size,
        emb_dim=_embedding_dim_from_state_dict(model_dir / "candidate_tower.pt"),
    )
    query_tower.load_state_dict(torch.load(model_dir / "query_tower.pt", map_location=device))
    item_tower.load_state_dict(torch.load(model_dir / "candidate_tower.pt", map_location=device))
    query_tower.to(device)
    item_tower.to(device)
    return query_tower, item_tower, state


def _embedding_dim_from_state_dict(path: Path) -> int:
    """Infer embedding dimension from the first layer weight shape."""
    state_dict = torch.load(path, map_location="cpu")
    for key, tensor in state_dict.items():
        if key.endswith("weight") and tensor.ndim == 2:
            return int(tensor.shape[1])
    raise ValueError(f"Could not infer embedding dim from {path}")
