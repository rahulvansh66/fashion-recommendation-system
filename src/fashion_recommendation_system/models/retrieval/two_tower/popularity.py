"""Log-q popularity correction table for in-batch negative debiasing."""

from __future__ import annotations

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import StringLookup


def build_label_probs_table(
    train_df: pd.DataFrame,
    article_lookup: StringLookup,
) -> tf.lookup.StaticHashTable:
    """Build P(article_id) lookup keyed by StringLookup integer indices.

    Frequencies are computed on the train split only. Used during training
    to subtract log P(article) from in-batch softmax logits (column-wise).
    """
    counts = train_df.groupby("article_id")["article_id"].count()
    total = float(len(train_df))

    keys = tf.constant(counts.index.astype(str).tolist(), dtype=tf.string)
    keys = article_lookup(keys)
    values = tf.constant(
        [count / total for count in counts.values],
        dtype=tf.float32,
    )
    # Avoid log(0) if an unknown article appears in a batch.
    default_prob = tf.constant(1.0 / total, dtype=tf.float32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=default_prob,
    )
