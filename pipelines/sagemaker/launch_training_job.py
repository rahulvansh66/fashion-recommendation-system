"""Launch SageMaker TensorFlow Training Jobs for two-tower retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sagemaker.tensorflow import TensorFlow


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def launch_training_job(
    *,
    role: str,
    train_uri: str,
    val_uri: str,
    test_uri: str = "",
    mode: str = "train",
    trial_number: int = -1,
    mlflow_run_id: str = "",
    hyperparameters: dict[str, Any] | None = None,
    instance_type: str = "ml.m5.large",
    use_spot: bool = True,
    volume_size: int = 30,
    wait: bool = True,
) -> tuple[str, str | None]:
    """Launch a Training Job; return (job_name, s3_output_path)."""
    root = _repo_root()
    region = os.environ.get("AWS_REGION", "us-east-1")

    hp = {
        "train-uri": train_uri,
        "val-uri": val_uri,
        "test-uri": test_uri,
        "mode": mode,
        "trial-number": str(trial_number),
        "mlflow-run-id": mlflow_run_id,
    }
    if hyperparameters:
        for key, value in hyperparameters.items():
            hp[key if "-" in key else key.replace("_", "-")] = str(value)

    estimator = TensorFlow(
        entry_point="train.py",
        source_dir=str(root / "pipelines" / "training" / "two_tower"),
        dependencies=[str(root / "src"), str(root / "configs")],
        role=role,
        instance_type=instance_type,
        volume_size=volume_size,
        max_run=3600,
        hyperparameters=hp,
        framework_version="2.15",
        py_version="py311",
        base_job_name="two-tower-retrieval",
        environment={
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
            "MLFLOW_EXPERIMENT": os.environ.get("MLFLOW_EXPERIMENT", "fashion-reco-dev"),
            "GIT_SHA": os.environ.get("GIT_SHA", "unknown"),
            "FEATURE_SNAPSHOT": os.environ.get("FEATURE_SNAPSHOT", "unknown"),
            "DATA_ENV": os.environ.get("DATA_ENV", "dev"),
        },
        use_spot_instances=use_spot,
        max_wait=7200 if use_spot else None,
    )

    estimator.fit(wait=wait)
    job_name = estimator.latest_training_job.name
    output_path = None
    if estimator.latest_training_job.describe().get("ModelArtifacts"):
        output_path = estimator.latest_training_job.describe()["ModelArtifacts"]["S3ModelArtifacts"]
    return job_name, output_path
