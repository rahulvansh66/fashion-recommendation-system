"""Infrastructure env vars for notebooks (no src/ imports)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class MLInfraConfig:
    """AWS MLflow, Optuna, and SageMaker settings from environment."""

    s3_bucket: str
    aws_region: str
    mlflow_tracking_uri: str
    mlflow_experiment: str
    optuna_storage_uri: str
    sagemaker_role_arn: str
    mlflow_tracking_server_name: str
    git_sha: str
    data_env: str

    @classmethod
    def from_env(cls) -> "MLInfraConfig":
        """Load config from process environment (typically ``.env.local``)."""
        return cls(
            s3_bucket=os.getenv("S3_BUCKET", "fashion-reco-dev"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT", "fashion-reco-dev"),
            optuna_storage_uri=os.getenv("OPTUNA_STORAGE_URI", ""),
            sagemaker_role_arn=os.getenv("SAGEMAKER_ROLE_ARN", ""),
            mlflow_tracking_server_name=os.getenv(
                "MLFLOW_TRACKING_SERVER_NAME", "fashion-reco-mlflow-dev"
            ),
            git_sha=os.getenv("GIT_SHA", "unknown"),
            data_env=os.getenv("DATA_ENV", "dev"),
        )

    def validate_for_aws(self) -> None:
        """Raise if required AWS experiment-tracking vars are missing."""
        missing = [
            name
            for name, value in [
                ("MLFLOW_TRACKING_URI", self.mlflow_tracking_uri),
                ("OPTUNA_STORAGE_URI", self.optuna_storage_uri),
                ("SAGEMAKER_ROLE_ARN", self.sagemaker_role_arn),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required env vars: {', '.join(missing)}")
