"""Vocabulary maps and feature encoding for two-tower PyTorch training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch


@dataclass
class Vocabulary:
    """String-to-index map; index 0 is reserved for unknown tokens."""

    values: list[str]

    def __post_init__(self) -> None:
        # Index 0 = unknown; known tokens start at 1.
        self._to_idx: dict[str, int] = {value: idx + 1 for idx, value in enumerate(self.values)}

    @property
    def size(self) -> int:
        """Number of known tokens (excluding unknown slot)."""
        return len(self.values)

    @property
    def num_embeddings(self) -> int:
        """Embedding table size including unknown slot."""
        return len(self.values) + 1

    def encode(self, value: str) -> int:
        """Map a string to an integer index."""
        return self._to_idx.get(str(value), 0)

    def encode_series(self, series: pd.Series) -> torch.Tensor:
        """Encode a pandas Series to a long tensor."""
        return torch.tensor([self.encode(v) for v in series.astype(str)], dtype=torch.long)

    def to_dict(self) -> dict[str, Any]:
        return {"values": self.values}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        return cls(values=list(data["values"]))


@dataclass
class AgeNormalizer:
    """Z-score normalizer for customer age (train statistics only)."""

    mean: float
    std: float

    @classmethod
    def from_series(cls, ages: pd.Series) -> AgeNormalizer:
        mean = float(ages.mean())
        std = float(ages.std())
        if std == 0.0:
            std = 1.0
        return cls(mean=mean, std=std)

    def normalize(self, age: float) -> float:
        return (age - self.mean) / self.std

    def normalize_tensor(self, ages: torch.Tensor) -> torch.Tensor:
        return (ages - self.mean) / self.std

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> AgeNormalizer:
        return cls(mean=float(data["mean"]), std=float(data["std"]))


@dataclass
class PreprocessState:
    """Serializable preprocessing state for training and inference."""

    user_vocab: Vocabulary
    item_vocab: Vocabulary
    category_vocab: Vocabulary
    index_group_vocab: Vocabulary
    age_normalizer: AgeNormalizer

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_vocab": self.user_vocab.to_dict(),
            "item_vocab": self.item_vocab.to_dict(),
            "category_vocab": self.category_vocab.to_dict(),
            "index_group_vocab": self.index_group_vocab.to_dict(),
            "age_normalizer": self.age_normalizer.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessState:
        return cls(
            user_vocab=Vocabulary.from_dict(data["user_vocab"]),
            item_vocab=Vocabulary.from_dict(data["item_vocab"]),
            category_vocab=Vocabulary.from_dict(data["category_vocab"]),
            index_group_vocab=Vocabulary.from_dict(data["index_group_vocab"]),
            age_normalizer=AgeNormalizer.from_dict(data["age_normalizer"]),
        )

    @classmethod
    def from_json(cls, raw: str) -> PreprocessState:
        return cls.from_dict(json.loads(raw))


def build_preprocess_state(
    train_df: pd.DataFrame,
    vocabs: dict[str, list[str]],
) -> PreprocessState:
    """Build preprocessing state from train split vocabularies and age stats."""
    return PreprocessState(
        user_vocab=Vocabulary(vocabs["user_ids"]),
        item_vocab=Vocabulary(vocabs["item_ids"]),
        category_vocab=Vocabulary(vocabs["item_categories"]),
        index_group_vocab=Vocabulary(vocabs["index_groups"]),
        age_normalizer=AgeNormalizer.from_series(train_df["age"].astype(float)),
    )


def encode_batch(state: PreprocessState, batch: dict[str, list]) -> dict[str, torch.Tensor]:
    """Encode a collated raw batch (lists) into model input tensors."""
    ages = torch.tensor([float(v) for v in batch["age"]], dtype=torch.float32)
    return {
        "customer_idx": torch.tensor(
            [state.user_vocab.encode(v) for v in batch["customer_id"]], dtype=torch.long
        ),
        "age": state.age_normalizer.normalize_tensor(ages),
        "txn_month_sin": torch.tensor(
            [float(v) for v in batch["txn_month_sin"]], dtype=torch.float32
        ),
        "txn_month_cos": torch.tensor(
            [float(v) for v in batch["txn_month_cos"]], dtype=torch.float32
        ),
        "article_idx": torch.tensor(
            [state.item_vocab.encode(v) for v in batch["article_id"]], dtype=torch.long
        ),
        "category_idx": torch.tensor(
            [state.category_vocab.encode(v) for v in batch["item_category"]], dtype=torch.long
        ),
        "index_group_idx": torch.tensor(
            [state.index_group_vocab.encode(v) for v in batch["index_group_name"]],
            dtype=torch.long,
        ),
    }
