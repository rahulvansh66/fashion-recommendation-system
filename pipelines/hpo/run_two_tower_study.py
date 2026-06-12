#!/usr/bin/env python3
"""Optuna study orchestrator — launches one SageMaker Training Job per trial."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import mlflow
import optuna
from optuna.integration import MLflowCallback

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fashion_recommendation_system.common.two_tower_config import (
    load_two_tower_config,
    load_two_tower_search_space,
)
from pipelines.sagemaker.launch_training_job import launch_training_job


def _parse_args() -> argparse.Namespace:
    defaults = load_two_tower_config(_REPO_ROOT)
    optuna_cfg = defaults.get("optuna", {})
    parser = argparse.ArgumentParser(description="Run Optuna two-tower HPO study")
    parser.add_argument("--train-uri", required=True)
    parser.add_argument("--val-uri", required=True)
    parser.add_argument("--study-name", default=optuna_cfg.get("study_name", "two_tower_hpo_v1"))
    parser.add_argument("--n-trials", type=int, default=optuna_cfg.get("n_trials", 3))
    parser.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN", ""))
    return parser.parse_args()


def _suggest_params(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    """Map YAML search space to Optuna suggestions."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def _fetch_result_json(output_s3_uri: str | None) -> dict[str, Any]:
    """Read result.json from the training job output channel."""
    if not output_s3_uri:
        return {}
    import json
    import tempfile

    import boto3

    if output_s3_uri.startswith("s3://"):
        parts = output_s3_uri[5:].split("/", 1)
        bucket, key_prefix = parts[0], parts[1] if len(parts) > 1 else ""
    else:
        return {}

    s3 = boto3.client("s3")
    key = f"{key_prefix.rstrip('/')}/result.json"
    try:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            s3.download_file(bucket, key, tmp.name)
            return json.loads(Path(tmp.name).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _wait_for_job(job_name: str, region: str) -> dict[str, Any]:
    """Poll SageMaker until the training job completes."""
    sm = boto3.client("sagemaker", region_name=region)
    while True:
        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(30)

    if status != "Completed":
        raise RuntimeError(f"Training job {job_name} ended with status {status}")

    output_path = desc.get("OutputDataConfig", {}).get("S3OutputPath")
    return {"job_name": job_name, "output_path": output_path, "status": status}


def main() -> None:
    args = _parse_args()
    if not args.role:
        raise ValueError("SAGEMAKER_ROLE_ARN must be set")

    defaults = load_two_tower_config(_REPO_ROOT)
    space = load_two_tower_search_space(_REPO_ROOT)
    storage = os.environ["OPTUNA_STORAGE_URI"]
    region = os.environ.get("AWS_REGION", "us-east-1")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "fashion-reco-dev"))

    callback = MLflowCallback(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        metric_name="val_recall_at_100",
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction=defaults.get("optuna", {}).get("direction", "maximize"),
    )

    # Enqueue baseline trial with guide defaults
    study.enqueue_trial(
        {
            "learning_rate": defaults["learning_rate"],
            "embedding_dim": defaults["embedding_dim"],
            "batch_size": defaults["batch_size"],
            "weight_decay": defaults["weight_decay"],
            "epochs": defaults["epochs"],
        }
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, space)
        hp = {
            "embedding-dim": params["embedding_dim"],
            "batch-size": params["batch_size"],
            "epochs": params["epochs"],
            "learning-rate": params["learning_rate"],
            "weight-decay": params["weight_decay"],
        }
        job_name, model_artifacts = launch_training_job(
            role=args.role,
            train_uri=args.train_uri,
            val_uri=args.val_uri,
            mode="train",
            trial_number=trial.number,
            hyperparameters=hp,
            wait=True,
        )
        job_info = _wait_for_job(job_name, region)
        result = _fetch_result_json(job_info.get("output_path"))
        val_recall = float(result.get("val_recall_at_100", 0.0))
        trial.set_user_attr("val_recall_at_100", val_recall)
        trial.set_user_attr("training_job_name", job_name)
        return val_recall

    with mlflow.start_run(run_name=args.study_name) as parent:
        mlflow.set_tags(
            {
                "git_sha": os.environ.get("GIT_SHA", "unknown"),
                "feature_snapshot": os.environ.get("FEATURE_SNAPSHOT", "unknown"),
                "feature_cutoff": "2020-03-31",
                "model": "two_tower",
                "data_env": os.environ.get("DATA_ENV", "dev"),
            }
        )
        mlflow.log_params({"n_trials": args.n_trials, "study_name": args.study_name})
        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[callback],
        )
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_recall_at_100", study.best_value)

    summary = {
        "best_params": study.best_params,
        "best_value": study.best_value,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
