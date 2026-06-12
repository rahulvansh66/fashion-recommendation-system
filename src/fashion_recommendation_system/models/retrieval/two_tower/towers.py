"""Query and candidate tower Keras models."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Normalization, StringLookup


class QueryTower(tf.keras.Model):
    """Encode user + temporal context into a shared embedding space."""

    def __init__(self, user_ids: list[str], emb_dim: int) -> None:
        super().__init__()
        self.user_embedding = tf.keras.Sequential(
            [
                StringLookup(vocabulary=user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(user_ids) + 1, emb_dim),
            ]
        )
        self.normalized_age = Normalization(axis=None)
        self.fnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(emb_dim, activation="relu"),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        concatenated = tf.concat(
            [
                self.user_embedding(inputs["customer_id"]),
                tf.reshape(self.normalized_age(inputs["age"]), (-1, 1)),
                tf.reshape(inputs["txn_month_sin"], (-1, 1)),
                tf.reshape(inputs["txn_month_cos"], (-1, 1)),
            ],
            axis=1,
        )
        return self.fnn(concatenated, training=training)


class ItemTower(tf.keras.Model):
    """Encode item identity and category metadata into the shared space."""

    def __init__(
        self,
        item_ids: list[str],
        item_categories: list[str],
        index_groups: list[str],
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.item_categories = item_categories
        self.index_groups = index_groups

        self.item_embedding = tf.keras.Sequential(
            [
                StringLookup(vocabulary=item_ids, mask_token=None),
                tf.keras.layers.Embedding(len(item_ids) + 1, emb_dim),
            ]
        )
        self.item_category_tokenizer = StringLookup(
            vocabulary=item_categories, mask_token=None
        )
        self.index_group_tokenizer = StringLookup(
            vocabulary=index_groups, mask_token=None
        )
        self.fnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(emb_dim, activation="relu"),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    @property
    def article_lookup(self) -> StringLookup:
        """StringLookup layer for article_id (used for popularity table keys)."""
        return self.item_embedding.layers[0]

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        category_one_hot = tf.one_hot(
            self.item_category_tokenizer(inputs["item_category"]),
            len(self.item_categories),
        )
        index_one_hot = tf.one_hot(
            self.index_group_tokenizer(inputs["index_group_name"]),
            len(self.index_groups),
        )
        concatenated = tf.concat(
            [
                self.item_embedding(inputs["article_id"]),
                category_one_hot,
                index_one_hot,
            ],
            axis=1,
        )
        return self.fnn(concatenated, training=training)
