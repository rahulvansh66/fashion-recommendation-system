"""Helpers for two_tower_retrieval_experiments.ipynb — no src/ imports."""

from __future__ import annotations

import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from utils.config_loader import find_repo_root

REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "txn_month_sin",
    "txn_month_cos",
    "article_id",
    "item_category",
    "index_group_name",
    "snap_date",
]


def load_two_tower_yaml(repo_root: Path | None = None) -> dict[str, Any]:
    """Load configs/models/two_tower.yaml."""
    root = repo_root or find_repo_root()
    path = root / "configs" / "models" / "two_tower.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_search_space_yaml(repo_root: Path | None = None) -> dict[str, Any]:
    """Load configs/hpo/two_tower_search_space.yaml."""
    root = repo_root or find_repo_root()
    path = root / "configs" / "hpo" / "two_tower_search_space.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def resolve_features_path(repo_root: Path, fe_cfg: dict[str, Any]) -> Path:
    """Return the Hive-partitioned features directory for the active dataset."""
    return (
        repo_root
        / fe_cfg["local_s3_root"]
        / "dataset"
        / fe_cfg["dataset_name"]
        / "features"
    )


def verify_schema(df: pd.DataFrame) -> None:
    """Ensure the anchor feature frame has required model columns."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns (extend FE first): {missing}")


def apply_temporal_split(
    df: pd.DataFrame,
    temporal: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """FR-BATCH-02 snap-date temporal split on anchor feature rows.

    Rows are assigned to train / val / test when ``snap_date`` matches a snap
    key for that role in ``temporal``.  When ``label`` is present, only purchase
    positives (``label == 1``) are kept for retrieval training.

    Legacy tables with ``t_dat`` (and no ``snap_date``) fall back to label-window
    selection on purchase dates.
    """
    work = df.copy()
    if "snap_date" in work.columns:
        work["snap_date"] = pd.to_datetime(work["snap_date"]).dt.normalize()

        def _collect(snaps: list) -> pd.DataFrame:
            if not snaps:
                return work.iloc[0:0].copy()
            snap_dates = {pd.Timestamp(s["snap_date"]).normalize() for s in snaps}
            mask = work["snap_date"].isin(snap_dates)
            subset = work[mask].copy()
            if "label" in subset.columns:
                subset = subset[subset["label"] == 1].copy()
            return subset

        train_df = _collect(temporal.get("train_snaps", []))
        val_df = _collect(temporal.get("val_snaps", []))
        test_df = _collect(temporal.get("test_snaps", []))
        return train_df, val_df, test_df

    if "t_dat" in work.columns:
        work["t_dat"] = pd.to_datetime(work["t_dat"]).dt.normalize()

        def _collect_legacy(snaps: list) -> pd.DataFrame:
            if not snaps:
                return work.iloc[0:0].copy()
            mask = pd.Series(False, index=work.index)
            for snap in snaps:
                lo = pd.Timestamp(snap["label_start"])
                hi = pd.Timestamp(snap["label_end"])
                mask |= (work["t_dat"] >= lo) & (work["t_dat"] <= hi)
            return work[mask].copy()

        train_df = _collect_legacy(temporal.get("train_snaps", []))
        val_df = _collect_legacy(temporal.get("val_snaps", []))
        test_df = _collect_legacy(temporal.get("test_snaps", []))
        return train_df, val_df, test_df

    raise ValueError("Temporal split requires a 'snap_date' or 't_dat' column")


def new_run_id() -> str:
    """Generate a unique experiment run id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def stage_splits_local(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    local_s3_root: Path,
    run_id: str,
) -> dict[str, str]:
    """Write train/val/test Parquet under local s3 mirror."""
    base = local_s3_root / "experiments" / "two_tower" / run_id
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": base / "train.parquet",
        "val": base / "val.parquet",
        "test": base / "test.parquet",
    }
    train_df.to_parquet(paths["train"], index=False)
    val_df.to_parquet(paths["val"], index=False)
    test_df.to_parquet(paths["test"], index=False)
    return {k: str(v) for k, v in paths.items()}


def stage_splits_s3(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    bucket: str,
    run_id: str,
    region: str = "us-east-1",
) -> dict[str, str]:
    """Upload split Parquet to S3 experiment prefix."""
    import boto3

    prefix = f"experiments/two_tower/{run_id}"
    s3 = boto3.client("s3", region_name=region)
    uris: dict[str, str] = {}

    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        key = f"{prefix}/{name}.parquet"
        local = Path(f"/tmp/{name}.parquet")
        frame.to_parquet(local, index=False)
        s3.upload_file(str(local), bucket, key)
        uris[name] = f"s3://{bucket}/{key}"
    return uris


def mlflow_server_status(server_name: str, region: str) -> str:
    """Return MLflow tracking server status via AWS CLI."""
    result = subprocess.run(
        [
            "aws",
            "sagemaker",
            "describe-mlflow-tracking-server",
            "--tracking-server-name",
            server_name,
            "--region",
            region,
            "--query",
            "TrackingServerStatus",
            "--output",
            "text",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or result.stderr.strip()


def export_best_params(best_params: dict[str, Any], repo_root: Path | None = None) -> Path:
    """Write study.best_params into configs/models/two_tower.yaml."""
    root = repo_root or find_repo_root()
    config = load_two_tower_yaml(root)
    for key, value in best_params.items():
        if key in config:
            config[key] = value
    path = root / "configs" / "models" / "two_tower.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
