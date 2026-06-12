#!/usr/bin/env python3
"""Launch SageMaker Processing job that runs the Optuna HPO orchestrator."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from fashion_recommendation_system.common.two_tower_config import load_two_tower_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch HPO Processing job")
    parser.add_argument("--train-uri", required=True)
    parser.add_argument("--val-uri", required=True)
    parser.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN", ""))
    args = parser.parse_args()

    if not args.role:
        raise ValueError("SAGEMAKER_ROLE_ARN must be set")

    defaults = load_two_tower_config(_REPO_ROOT)
    sm_cfg = defaults.get("sagemaker", {})
    region = os.environ.get("AWS_REGION", "us-east-1")
    session = __import__("sagemaker").Session(boto_session=__import__("boto3").Session(region_name=region))

    processor = ScriptProcessor(
        role=args.role,
        image_uri=session.image_uris.retrieve(
            framework="tensorflow",
            region=region,
            version="2.15",
            py_version="py311",
            instance_type=sm_cfg.get("instance_type", "ml.m5.large"),
        ),
        command=["python3"],
        instance_type=sm_cfg.get("instance_type", "ml.m5.large"),
        instance_count=1,
        volume_size_in_gb=sm_cfg.get("volume_size_gb", 30),
        env={
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
            "MLFLOW_EXPERIMENT": os.environ.get("MLFLOW_EXPERIMENT", "fashion-reco-dev"),
            "OPTUNA_STORAGE_URI": os.environ.get("OPTUNA_STORAGE_URI", ""),
            "AWS_REGION": region,
            "SAGEMAKER_ROLE_ARN": args.role,
            "GIT_SHA": os.environ.get("GIT_SHA", "unknown"),
            "FEATURE_SNAPSHOT": os.environ.get("FEATURE_SNAPSHOT", "unknown"),
            "DATA_ENV": os.environ.get("DATA_ENV", "dev"),
        },
    )

    processor.run(
        code=str(_REPO_ROOT / "pipelines" / "hpo" / "run_two_tower_study.py"),
        inputs=[
            ProcessingInput(
                source=str(_REPO_ROOT / "src"),
                destination="/opt/ml/processing/input/src",
            ),
            ProcessingInput(
                source=str(_REPO_ROOT / "configs"),
                destination="/opt/ml/processing/input/configs",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{os.environ.get('S3_BUCKET', 'fashion-reco-dev')}/experiments/optuna/",
            ),
        ],
        arguments=[
            "--train-uri",
            args.train_uri,
            "--val-uri",
            args.val_uri,
        ],
        wait=False,
    )


if __name__ == "__main__":
    main()
