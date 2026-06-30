"""PyTorch query and candidate towers for two-tower retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTower(nn.Module):
    """Encode user identity and temporal context into a shared embedding space."""

    def __init__(self, num_users: int, emb_dim: int) -> None:
        super().__init__()
        # +1 for unknown customer at index 0.
        self.customer_embedding = nn.Embedding(num_users + 1, emb_dim)
        self.fnn = nn.Sequential(
            nn.Linear(emb_dim + 3, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return query embeddings of shape (batch, emb_dim)."""
        customer_vec = self.customer_embedding(batch["customer_idx"])
        features = torch.stack(
            [
                customer_vec,
                batch["age"].unsqueeze(1),
                batch["txn_month_sin"].unsqueeze(1),
                batch["txn_month_cos"].unsqueeze(1),
            ],
            dim=1,
        )
        concatenated = features.reshape(features.size(0), -1)
        return self.fnn(concatenated)


class ItemTower(nn.Module):
    """Encode item identity and category metadata into the shared space."""

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        num_index_groups: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.num_index_groups = num_index_groups
        # +1 for unknown article at index 0.
        self.article_embedding = nn.Embedding(num_items + 1, emb_dim)
        input_dim = emb_dim + num_categories + num_index_groups
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return item embeddings of shape (batch, emb_dim)."""
        article_vec = self.article_embedding(batch["article_idx"])
        category_one_hot = F.one_hot(batch["category_idx"], self.num_categories).float()
        index_one_hot = F.one_hot(batch["index_group_idx"], self.num_index_groups).float()
        concatenated = torch.cat([article_vec, category_one_hot, index_one_hot], dim=1)
        return self.fnn(concatenated)
