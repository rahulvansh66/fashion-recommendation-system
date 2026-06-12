"""Load configs/*.yaml for notebooks — does not import from src/ (per project-structure.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Locate repository root by finding configs/ directory.

    Parameters
    ----------
    start : Path | None
        Starting directory for upward search. Defaults to cwd.

    Returns
    -------
    Path
        Repository root directory.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").is_dir():
            return candidate
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _repo_relative_path(repo_root: Path, path_str: str) -> str:
    """Normalize local roots written as notebooks-relative (``../dataset``) or repo-relative.

    Notebook helpers join ``resolve_repo_root() / local_*_root``, so returned paths
    must be relative to the repository root (e.g. ``dataset``, ``s3``).
    """
    path = Path(path_str)
    if path_str.startswith("../"):
        resolved = (repo_root / "notebooks" / path).resolve()
    else:
        resolved = (repo_root / path).resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def load_feature_engineering_config(
    repo_root: Path | None = None,
    environment: str = "local_dev",
) -> dict[str, Any]:
    """Merge feature-engineering YAML into a flat dict for notebook use.

    Notebooks read YAML only. Infrastructure env overrides are applied when
    running ``pipelines/run_feature_pipeline.py`` (uses src/config.py).

    Parameters
    ----------
    repo_root : Path | None
        Repository root. Auto-detected when None.
    environment : str
        ``local_dev`` or ``aws`` key under s3_paths.yaml → environments.

    Returns
    -------
    dict
        Merged configuration dict (CONFIG).
    """
    root = repo_root or find_repo_root()
    configs = root / "configs"

    preprocessing = _load_yaml(configs / "data" / "preprocessing.yaml")
    s3_paths = _load_yaml(configs / "data" / "s3_paths.yaml")
    item_features = _load_yaml(configs / "features" / "item_features.yaml")
    user_features = _load_yaml(configs / "features" / "user_features.yaml")
    cross_features = _load_yaml(configs / "features" / "cross_features.yaml")
    feature_sets = _load_yaml(configs / "features" / "feature_sets.yaml")

    temporal = preprocessing.get("temporal_split", {})
    train_end = temporal.get("train_end", "2020-03-24")
    feature_cutoff = temporal.get("feature_cutoff") or train_end

    env_cfg = s3_paths.get("environments", {}).get(environment, {})
    datasets = s3_paths.get("datasets", {})

    return {
        "repo_root": str(root),
        "environment": environment,
        "storage_mode": env_cfg.get("storage_mode", "local"),
        "dataset_name": datasets.get("active", "sample_2000_users"),
        "stage_datasets": datasets.get("stage", ["dummy", "sample_2000_users"]),
        "local_dataset_root": _repo_relative_path(
            root, env_cfg.get("local_dataset_root", "../dataset")
        ),
        "local_s3_root": _repo_relative_path(
            root, env_cfg.get("local_s3_root", "../s3")
        ),
        "s3_bucket": env_cfg.get("s3_bucket", "fashion-reco-dev"),
        "aws_region": "us-east-1",
        "localstack_endpoint": "",
        "train_end": train_end,
        "cutoff_date": temporal.get("cutoff_date", "2020-03-31"),
        "test_end": temporal.get("test_end", "2020-04-07"),
        "feature_cutoff": feature_cutoff,
        "label_window_days": temporal.get("label_window_days", 7),
        "category_col": item_features.get("columns", {}).get("category", "garment_group_name"),
        "color_col": item_features.get("columns", {}).get("color", "colour_group_name"),
        "decay_half_life_days": user_features.get("decay", {}).get("half_life_days", 180),
        "item_lookbacks_days": item_features.get("lookbacks_days", {}),
        "user_pref_top_n": user_features.get("preference", {}).get("top_n", 3),
        "user_pref_lookback_days": user_features.get("preference", {}).get("lookback_days", 365),
        "feature_sets": feature_sets.get("enabled", {}),
        "prefixes": env_cfg.get("prefixes", {}),
        "runtime_mode": preprocessing.get("runtime", {}).get("mode", "local"),
        "cross_features": cross_features,
        "user_features": user_features,
        "item_features": item_features,
    }


def is_glue_runtime(runtime_mode: str) -> bool:
    """Return True when YAML runtime mode is set to glue.

    Parameters
    ----------
    runtime_mode : str
        Value from ``configs/data/preprocessing.yaml`` → ``runtime.mode``.

    Returns
    -------
    bool
        True for Glue notebook / job runs.
    """
    return runtime_mode.lower() == "glue"
