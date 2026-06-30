#!/usr/bin/env python3
"""SageMaker entrypoint for two-tower retrieval training and evaluation.

Usage (local):
    python src/fashion_recommendation_system/models/retrieval/two_tower/train.py \\
        --train-uri s3/experiments/two_tower/run_id/train.parquet \\
        --val-uri s3/experiments/two_tower/run_id/val.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch

# Repo root: six levels up from this file (two_tower -> retrieval -> models -> package -> src -> root).
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from fashion_recommendation_system.common.two_tower_config import load_two_tower_config
from fashion_recommendation_system.models.retrieval.two_tower.dataset import (
    build_dataloaders,
    get_unique_items_df,
)
from fashion_recommendation_system.models.retrieval.two_tower.evaluate import (
    embed_candidate_corpus,
    evaluate_epoch,
    recall_at_100,
)
from fashion_recommendation_system.models.retrieval.two_tower.export import load_artifacts, save_artifacts
from fashion_recommendation_system.models.retrieval.two_tower.loss import (
    build_article_prob_map,
    popularity_corrected_loss,
)
from fashion_recommendation_system.models.retrieval.two_tower.model import ItemTower, QueryTower
from fashion_recommendation_system.models.retrieval.two_tower.preprocess import (
    build_preprocess_state,
    encode_batch,
)
from fashion_recommendation_system.models.retrieval.two_tower.split import (
    build_vocabularies,
    load_transactions,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments; hyperparameters mirror the former TF pipeline."""
    defaults = load_two_tower_config(_REPO_ROOT)
    parser = argparse.ArgumentParser(description="Two-tower retrieval training (PyTorch)")
    parser.add_argument("--train-uri", required=True, help="Train Parquet path (file or dir)")
    parser.add_argument("--val-uri", required=True, help="Val Parquet path")
    parser.add_argument("--test-uri", default="", help="Test Parquet path (optional)")
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="train: fit on train; eval: load checkpoint and score test",
    )
    parser.add_argument("--embedding-dim", type=int, default=defaults["embedding_dim"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))
    parser.add_argument("--mlflow-run-id", default="", help="Optional parent MLflow run id")
    parser.add_argument("--trial-number", type=int, default=-1)
    return parser.parse_args()


def _load_split(uri: str) -> pd.DataFrame:
    """Load a staged split Parquet."""
    path = Path(uri)
    if path.is_dir() and not list(path.glob("*.parquet")):
        raise FileNotFoundError(f"No parquet data at {uri}")
    return load_transactions(uri)


def _setup_mlflow() -> None:
    """Configure MLflow tracking from environment variables."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = os.environ.get("MLFLOW_EXPERIMENT", "fashion-reco-dev")
    mlflow.set_experiment(experiment)


def _log_tags() -> None:
    mlflow.set_tags(
        {
            "git_sha": os.environ.get("GIT_SHA", "unknown"),
            "feature_snapshot": os.environ.get("FEATURE_SNAPSHOT", "unknown"),
            "feature_cutoff": "2020-03-31",
            "model": "two_tower",
            "framework": "pytorch",
            "data_env": os.environ.get("DATA_ENV", "dev"),
        }
    )


def _build_towers(state, embedding_dim: int) -> tuple[QueryTower, ItemTower]:
    """Construct query and item towers from preprocessing state."""
    query_tower = QueryTower(state.user_vocab.size, embedding_dim)
    item_tower = ItemTower(
        num_items=state.item_vocab.size,
        num_categories=state.category_vocab.size,
        num_index_groups=state.index_group_vocab.size,
        emb_dim=embedding_dim,
    )
    return query_tower, item_tower


def _train_loop(
    query_tower: QueryTower,
    item_tower: ItemTower,
    train_loader,
    val_loader,
    corpus_embeddings: torch.Tensor,
    article_to_row: dict[str, int],
    state,
    prob_map: dict[int, float],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[list[float], dict[str, float]]:
    """Run training epochs; return per-epoch train loss and final val metrics."""
    params = list(query_tower.parameters()) + list(item_tower.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    train_losses: list[float] = []
    final_val_metrics: dict[str, float] = {}

    for _epoch in range(epochs):
        query_tower.train()
        item_tower.train()
        epoch_loss = 0.0
        num_batches = 0

        for raw_batch in train_loader:
            encoded = encode_batch(state, raw_batch)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            user_emb = query_tower(encoded)
            item_emb = item_tower(encoded)
            loss = popularity_corrected_loss(
                user_emb, item_emb, encoded["article_idx"], prob_map
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1

        train_losses.append(epoch_loss / max(num_batches, 1))
        final_val_metrics = evaluate_epoch(
            query_tower,
            item_tower,
            val_loader,
            corpus_embeddings,
            article_to_row,
            state,
            device,
        )

    return train_losses, final_val_metrics


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = _load_split(args.train_uri)
    val_df = _load_split(args.val_uri)

    _setup_mlflow()
    run_name = f"trial_{args.trial_number}" if args.trial_number >= 0 else "two_tower_train"

    with mlflow.start_run(run_name=run_name, nested=bool(args.mlflow_run_id)):
        if args.mlflow_run_id:
            mlflow.set_tag("mlflow.parentRunId", args.mlflow_run_id)
        _log_tags()
        mlflow.log_params(
            {
                "embedding_dim": args.embedding_dim,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "mode": args.mode,
            }
        )

        vocabs = build_vocabularies(train_df)
        state = build_preprocess_state(train_df, vocabs)
        train_loader, val_loader = build_dataloaders(train_df, val_df, args.batch_size)
        items_df = get_unique_items_df(train_df)

        if args.mode == "train":
            query_tower, item_tower = _build_towers(state, args.embedding_dim)
            query_tower.to(device)
            item_tower.to(device)
            prob_map = build_article_prob_map(train_df, state.item_vocab)
            corpus_embeddings, article_to_row = embed_candidate_corpus(
                item_tower, items_df, state, device
            )

            train_losses, val_metrics = _train_loop(
                query_tower,
                item_tower,
                train_loader,
                val_loader,
                corpus_embeddings,
                article_to_row,
                state,
                prob_map,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=device,
            )

            for epoch, train_loss in enumerate(train_losses):
                mlflow.log_metric("train_loss", train_loss, step=epoch)

            val_recall = recall_at_100(val_metrics)
            mlflow.log_metric("val_recall_at_100", val_recall)
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)

            metrics: dict[str, Any] = {
                "val_recall_at_100": val_recall,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            if args.test_uri:
                test_df = _load_split(args.test_uri)
                _, test_loader = build_dataloaders(train_df, test_df, args.batch_size)
                test_metrics = evaluate_epoch(
                    query_tower,
                    item_tower,
                    test_loader,
                    corpus_embeddings,
                    article_to_row,
                    state,
                    device,
                )
                test_recall = recall_at_100(test_metrics)
                mlflow.log_metric("test_recall_at_100", test_recall)
                metrics["test_recall_at_100"] = test_recall
        else:
            model_dir = Path(args.model_dir)
            query_tower, item_tower, state = load_artifacts(model_dir, device)
            corpus_embeddings, article_to_row = embed_candidate_corpus(
                item_tower, items_df, state, device
            )
            test_df = _load_split(args.test_uri)
            _, test_loader = build_dataloaders(train_df, test_df, args.batch_size)
            test_metrics = evaluate_epoch(
                query_tower,
                item_tower,
                test_loader,
                corpus_embeddings,
                article_to_row,
                state,
                device,
            )
            test_recall = recall_at_100(test_metrics)
            mlflow.log_metric("test_recall_at_100", test_recall)
            metrics = {"test_recall_at_100": test_recall, **{f"test_{k}": v for k, v in test_metrics.items()}}

        out_dir = Path(args.model_dir)
        save_artifacts(query_tower, item_tower, state, out_dir, metrics)
        mlflow.log_artifacts(str(out_dir))

        result_path = Path(args.output_data_dir) / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
