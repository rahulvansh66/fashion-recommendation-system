"""Two-tower retrieval model with log-q popularity correction."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_recommenders as tfrs

from fashion_recommendation_system.models.retrieval.two_tower.towers import (
    ItemTower,
    QueryTower,
)


class TwoTowerModel(tf.keras.Model):
    """Dual-encoder model with in-batch contrastive loss and log-q debiasing."""

    def __init__(
        self,
        query_model: QueryTower,
        item_model: ItemTower,
        item_ds: tf.data.Dataset,
        batch_size: int,
        label_probs_table: tf.lookup.StaticHashTable,
    ) -> None:
        super().__init__()
        self.query_model = query_model
        self.item_model = item_model
        self.label_probs_table = label_probs_table
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_ds.batch(batch_size).map(self.item_model)
            )
        )

    def _compute_logits(
        self,
        batch: dict[str, tf.Tensor],
        training: bool,
    ) -> tf.Tensor:
        user_embeddings = self.query_model(batch, training=training)
        item_embeddings = self.item_model(batch, training=training)
        return tf.matmul(user_embeddings, item_embeddings, transpose_b=True)

    def _popularity_corrected_loss(
        self,
        batch: dict[str, tf.Tensor],
        logits: tf.Tensor,
    ) -> tf.Tensor:
        """In-batch softmax CE with log-q correction (training only)."""
        article_indices = self.item_model.article_lookup(batch["article_id"])
        label_probs = self.label_probs_table.lookup(article_indices)
        # Column j corresponds to item j in the batch; subtract log P(item_j).
        corrected_logits = logits - tf.math.log(label_probs)[tf.newaxis, :]
        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)
        per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=corrected_logits,
        )
        return tf.reduce_mean(per_example)

    def train_step(self, batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            logits = self._compute_logits(batch, training=True)
            loss = self._popularity_corrected_loss(batch, logits)
            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": loss,
            "regularization_loss": regularization_loss,
            "total_loss": total_loss,
        }

    def test_step(self, batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        user_embeddings = self.query_model(batch, training=False)
        item_embeddings = self.item_model(batch, training=False)
        loss = self.task(user_embeddings, item_embeddings, compute_metrics=True)
        regularization_loss = sum(self.losses)
        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss
        return metrics

    def evaluate_dataset(self, dataset: tf.data.Dataset) -> dict[str, float]:
        """Run validation/test and return scalar metrics."""
        for metric in self.metrics:
            metric.reset_states()
        for batch in dataset:
            self.test_step(batch)
        return {metric.name: float(metric.result().numpy()) for metric in self.metrics}
