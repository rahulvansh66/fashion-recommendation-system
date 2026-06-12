#!/usr/bin/env python3
"""SageMaker entrypoint for two-tower retrieval training and evaluation.

Usage (local):
    python pipelines/training/two_tower/train.py \\
        --train-uri s3/dataset/sample_2000_users/features/transactions \\
        --val-uri ... --mode train

On SageMaker, hyperparameters are passed as CLI flags by the Estimator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from fashion_recommendation_system.common.two_tower_config import load_two_tower_config
from fashion_recommendation_system.models.retrieval.two_tower.dataset import (
    build_tf_datasets,
)
from fashion_recommendation_system.models.retrieval.two_tower.split import load_transactions
from fashion_recommendation_system.models.retrieval.two_tower.trainer import (
    build_model,
    recall_at_100,
    train_model,
)


def _parse_args() -> argparse.Namespace:
    defaults = load_two_tower_config(_REPO_ROOT)
    parser = argparse.ArgumentParser(description="Two-tower retrieval training")
    parser.add_argument("--train-uri", required=True, help="Train Parquet path (file or dir)")
    parser.add_argument("--val-uri", required=True, help="Val Parquet path")
    parser.add_argument("--test-uri", default="", help="Test Parquet path (eval mode)")
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="train: fit on train, eval on val; eval: load weights and score test",
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
    """Load a staged split Parquet; if uri is features dir, apply temporal split."""
    path = Path(uri)
    if path.is_dir() and not (path / "_SUCCESS").exists() and not list(path.glob("*.parquet")):
        raise FileNotFoundError(f"No parquet data at {uri}")
    df = load_transactions(uri)
    return df


def _setup_mlflow() -> None:
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
            "data_env": os.environ.get("DATA_ENV", "dev"),
        }
    )


def _save_artifacts(model, model_dir: Path, metrics: dict) -> None:
    """Persist towers and metrics JSON for SageMaker / MLflow."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model.query_model.save(model_dir / "query_model")
    model.item_model.save(model_dir / "candidate_model")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
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

        if args.mode == "train":
            model, history = train_model(
                train_df,
                val_df,
                embedding_dim=args.embedding_dim,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            for epoch, train_loss in enumerate(history.get("loss", [])):
                mlflow.log_metric("train_loss", train_loss, step=epoch)
            for epoch, val_loss in enumerate(history.get("val_loss", [])):
                mlflow.log_metric("val_loss", val_loss, step=epoch)

            _, val_ds = build_tf_datasets(train_df, val_df, args.batch_size)
            val_metrics = model.evaluate_dataset(val_ds)
            val_recall = recall_at_100(val_metrics)
            mlflow.log_metric("val_recall_at_100", val_recall)
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)

            metrics = {"val_recall_at_100": val_recall, **{f"val_{k}": v for k, v in val_metrics.items()}}

            if args.test_uri:
                test_df = _load_split(args.test_uri)
                _, test_ds = build_tf_datasets(train_df, test_df, args.batch_size)
                test_metrics = model.evaluate_dataset(test_ds)
                test_recall = recall_at_100(test_metrics)
                mlflow.log_metric("test_recall_at_100", test_recall)
                metrics["test_recall_at_100"] = test_recall
        else:
            # Eval-only: train briefly is not needed; rebuild and load if checkpoint exists
            model = build_model(
                train_df,
                embedding_dim=args.embedding_dim,
                batch_size=args.batch_size,
            )
            model_dir = Path(args.model_dir)
            query_path = model_dir / "query_model"
            if query_path.exists():
                model.query_model = tf.keras.models.load_model(query_path)
                model.item_model = tf.keras.models.load_model(model_dir / "candidate_model")
            test_df = _load_split(args.test_uri)
            _, test_ds = build_tf_datasets(train_df, test_df, args.batch_size)
            test_metrics = model.evaluate_dataset(test_ds)
            val_recall = recall_at_100(test_metrics)
            mlflow.log_metric("test_recall_at_100", val_recall)
            metrics = {"test_recall_at_100": val_recall, **{f"test_{k}": v for k, v in test_metrics.items()}}

        out_dir = Path(args.model_dir)
        _save_artifacts(model, out_dir, metrics)
        mlflow.log_artifacts(str(out_dir))

        result_path = Path(args.output_data_dir) / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
