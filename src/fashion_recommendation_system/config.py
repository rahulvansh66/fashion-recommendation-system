"""Infrastructure configuration — all os.getenv() calls live here only.

Model and feature parameters belong in configs/**/*.yaml, not in this module.
See docs/system-design/project-structure.md §4.

On import, loads ``.env.local`` from the repository root when present (does not
override variables already set in the process environment).
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_env_local() -> None:
    """Load repository ``.env.local`` into ``os.environ`` if the file exists."""
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".env.local"
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


_load_env_local()

# --- AWS auth (profile name only — keys live in ~/.aws/credentials or SSO) ---
AWS_PROFILE = os.getenv("AWS_PROFILE", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# --- Object storage ---
S3_BUCKET = os.getenv("S3_BUCKET", "fashion-reco-dev")
STORAGE_MODE = os.getenv("STORAGE_MODE", "local").lower()  # local | aws
LOCAL_S3_ROOT = os.getenv("FE_LOCAL_S3_ROOT", "../s3")
LOCAL_DATASET_ROOT = os.getenv("FE_LOCAL_DATASET_ROOT", "../dataset")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "")

# --- Cache ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# --- ML inference endpoints (local defaults for dev) ---
TWO_TOWER_ENDPOINT = os.getenv("TWO_TOWER_ENDPOINT", "http://localhost:8080")
XGBOOST_ENDPOINT = os.getenv("XGBOOST_ENDPOINT", "http://localhost:8081")
FAISS_LAMBDA_ARN = os.getenv("FAISS_LAMBDA_ARN", "")

# --- Experiment tracking & HPO ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fashion-reco-dev")
OPTUNA_STORAGE_URI = os.getenv("OPTUNA_STORAGE_URI", "")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
MLFLOW_TRACKING_SERVER_NAME = os.getenv(
    "MLFLOW_TRACKING_SERVER_NAME", "fashion-reco-mlflow-dev"
)

# --- Glue runtime detection ---
IS_GLUE = os.getenv("AWS_EXECUTION_ENV") is not None
