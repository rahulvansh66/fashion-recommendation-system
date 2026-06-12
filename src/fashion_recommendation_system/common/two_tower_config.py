"""Load two-tower model and HPO YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_two_tower_config(repo_root: Path | None = None) -> dict[str, Any]:
    """Load frozen defaults from configs/models/two_tower.yaml."""
    root = repo_root or _find_repo_root()
    path = root / "configs" / "models" / "two_tower.yaml"
    return _load_yaml(path)


def load_two_tower_search_space(repo_root: Path | None = None) -> dict[str, Any]:
    """Load Optuna search space from configs/hpo/two_tower_search_space.yaml."""
    root = repo_root or _find_repo_root()
    path = root / "configs" / "hpo" / "two_tower_search_space.yaml"
    return _load_yaml(path)


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").is_dir():
            return candidate
    return Path.cwd()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
