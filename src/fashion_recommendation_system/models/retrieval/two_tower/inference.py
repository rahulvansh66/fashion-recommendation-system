"""SageMaker inference handler stub for two-tower query encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fashion_recommendation_system.models.retrieval.two_tower.export import load_artifacts
from fashion_recommendation_system.models.retrieval.two_tower.preprocess import encode_batch


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load query tower and preprocessing state for SageMaker inference."""
    query_tower, _, state = load_artifacts(Path(model_dir))
    query_tower.eval()
    return {"query_tower": query_tower, "state": state}


def predict_fn(input_data: dict[str, Any], model: dict[str, Any]) -> np.ndarray:
    """Encode a single query request into a 16-d embedding vector."""
    query_tower = model["query_tower"]
    state = model["state"]
    raw_batch = {key: [input_data[key]] for key in input_data}
    encoded = encode_batch(state, raw_batch)
    with torch.no_grad():
        embedding = query_tower(encoded)
    return embedding.numpy()
