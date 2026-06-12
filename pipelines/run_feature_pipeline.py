#!/usr/bin/env python3
"""Entry point for the offline feature-engineering pipeline.

Loads configs/*.yaml for feature parameters and src/config.py for infrastructure.
Feature implementation will move from notebooks into src/features/ over time.

Usage:
    python pipelines/run_feature_pipeline.py
    python pipelines/run_feature_pipeline.py --environment aws
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when running as a script without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from fashion_recommendation_system.common.yaml_config import load_feature_engineering_config


def main() -> None:
    """Load merged config and print summary (pipeline steps wired in follow-up PRs)."""
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument(
        "--environment",
        default="local_dev",
        choices=["local_dev", "aws"],
        help="Config environment key in configs/data/s3_paths.yaml",
    )
    args = parser.parse_args()

    config = load_feature_engineering_config(
        repo_root=_REPO_ROOT,
        environment=args.environment,
    )

    print("Feature pipeline config loaded:")
    print(json.dumps({k: v for k, v in config.items() if k not in ("cross_features", "user_features", "item_features")}, indent=2))
    print("\nNext: wire Spark feature builders from src/fashion_recommendation_system/features/")


if __name__ == "__main__":
    main()
