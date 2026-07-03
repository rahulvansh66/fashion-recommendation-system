"""Helpers for 06_ranking_model_training.ipynb — no src/ imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from utils.config_loader import find_repo_root
from utils.two_tower_training_helpers import mlflow_server_status, new_run_id

# Re-export shared infra helpers used by the ranking notebook.
__all__ = [
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "RANKING_FEATURE_COLUMNS",
    "apply_temporal_split_ranking",
    "build_feature_schema",
    "build_xgb_classifier",
    "export_ranking_best_params",
    "extract_gain_importance",
    "hit_rate_at_k",
    "load_ranking_search_space_yaml",
    "load_ranking_yaml",
    "mlflow_server_status",
    "new_run_id",
    "prepare_feature_matrix",
    "sample_optuna_params",
    "save_model_artifacts",
    "stage_splits_local_ranking",
    "stage_splits_s3_ranking",
    "val_aucpr",
    "verify_ranking_schema",
    "zero_importance_features",
]

CATEGORICAL_FEATURES: list[str] = [
    "product_type_name",
    "item_category",
    "item_color",
    "department_name",
    "section_name",
    "index_group_name",
    "graphical_appearance_name",
    "product_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "index_name",
    "user_category_pref_1y_rank1",
    "user_category_pref_1y_rank2",
    "user_category_pref_1y_rank3",
    "user_color_pref_1y_rank1",
    "user_color_pref_1y_rank2",
]

NUMERIC_FEATURES: list[str] = [
    "user_item_repurchase",
    "user_item_decayed_repurchase",
    "user_item_decayed_interaction_ratio",
    "user_item_repurchase_365d",
    "user_item_repurchase_90d",
    "user_item_sales_channel_2_count",
    "user_item_repurchase_30d",
    "user_item_days_since_last_purchase",
    "user_item_price_decayed_zscore",
    "user_purchases_in_candidate_category_1y",
    "user_days_since_last_purchase_in_category",
    "user_category_match_rank1",
    "user_category_match_rank2",
    "user_category_match_rank3",
    "item_pop_7d",
    "item_pop_30d",
    "item_pop_180d",
    "item_pop_same_month_last_year",
    "item_category_pop_30d",
    "item_category_pop_180d",
    "days_since_first_sold",
    "item_recent_to_last_180d_ratio",
    "item_category_recent_to_lifetime_ratio",
    "item_seasonality_strength",
    "item_avg_price",
    "item_days_since_last_sold",
    "item_sales_channel_2_count",
    "item_sales_channel_2_share",
    "candidate_price",
    "age",
    "user_days_since_last_purchase",
    "user_purchase_count_30d",
    "user_purchase_count_180d",
    "user_decayed_price_avg",
    "user_decayed_price_std",
    "txn_month_sin",
    "txn_month_cos",
]

RANKING_FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

HPO_PARAM_KEYS = {
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
}


def load_ranking_yaml(repo_root: Path | None = None) -> dict[str, Any]:
    """Load ``configs/models/ranking.yaml``.

    Parameters
    ----------
    repo_root:
        Repository root. Auto-detected when ``None``.

    Returns
    -------
    dict
        Parsed ranking model configuration.
    """
    root = repo_root or find_repo_root()
    path = root / "configs" / "models" / "ranking.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_ranking_search_space_yaml(repo_root: Path | None = None) -> dict[str, Any]:
    """Load ``configs/hpo/ranking_search_space.yaml``.

    Parameters
    ----------
    repo_root:
        Repository root. Auto-detected when ``None``.

    Returns
    -------
    dict
        Optuna search-space specification keyed by hyperparameter name.
    """
    root = repo_root or find_repo_root()
    path = root / "configs" / "hpo" / "ranking_search_space.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def verify_ranking_schema(df: pd.DataFrame) -> None:
    """Ensure the anchor feature frame has ranking model columns.

    Parameters
    ----------
    df:
        Feature table loaded from ``dataset/{name}/features/``.

    Raises
    ------
    ValueError
        When required identifier, target, or model feature columns are missing.
    """
    required = ["customer_id", "article_id", "snap_date", "label", *RANKING_FEATURE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns (extend FE first): {missing}")


def apply_temporal_split_ranking(
    df: pd.DataFrame,
    temporal: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """FR-BATCH-02 snap-date temporal split for ranking (positives + negatives).

    Unlike retrieval training, **all** labeled rows matching each snap are kept.

    Parameters
    ----------
    df:
        Full feature table with ``snap_date`` and ``label``.
    temporal:
        ``temporal_split`` block from ``configs/models/ranking.yaml``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)`` excluding drift snaps.
    """
    work = df.copy()
    if "snap_date" not in work.columns:
        raise ValueError("Temporal split requires a 'snap_date' column")

    work["snap_date"] = pd.to_datetime(work["snap_date"]).dt.normalize()

    def _collect(snaps: list[dict[str, Any]] | None) -> pd.DataFrame:
        if not snaps:
            return work.iloc[0:0].copy()
        snap_dates = {pd.Timestamp(s["snap_date"]).normalize() for s in snaps}
        return work[work["snap_date"].isin(snap_dates)].copy()

    train_df = _collect(temporal.get("train_snaps", []))
    val_df = _collect(temporal.get("val_snaps", []))
    test_df = _collect(temporal.get("test_snaps", []))
    return train_df, val_df, test_df


def stage_splits_local_ranking(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    local_s3_root: Path,
    run_id: str,
) -> dict[str, str]:
    """Write train/val/test Parquet under the local ``s3/`` mirror.

    Parameters
    ----------
    train_df, val_df, test_df:
        Temporal split frames.
    local_s3_root:
        Local mirror root (e.g. ``repo/s3``).
    run_id:
        Unique experiment identifier.

    Returns
    -------
    dict[str, str]
        Mapping of split name to absolute file path.
    """
    base = local_s3_root / "experiments" / "ranking" / run_id
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


def stage_splits_s3_ranking(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    bucket: str,
    run_id: str,
    region: str = "us-east-1",
) -> dict[str, str]:
    """Upload split Parquet to ``s3://{bucket}/experiments/ranking/{run_id}/``.

    Parameters
    ----------
    train_df, val_df, test_df:
        Temporal split frames.
    bucket:
        Target S3 bucket name.
    run_id:
        Unique experiment identifier.
    region:
        AWS region for the S3 client.

    Returns
    -------
    dict[str, str]
        Mapping of split name to ``s3://`` URI.
    """
    import boto3

    prefix = f"experiments/ranking/{run_id}"
    s3 = boto3.client("s3", region_name=region)
    uris: dict[str, str] = {}

    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        key = f"{prefix}/{name}.parquet"
        local = Path(f"/tmp/ranking_{name}_{run_id}.parquet")
        frame.to_parquet(local, index=False)
        s3.upload_file(str(local), bucket, key)
        uris[name] = f"s3://{bucket}/{key}"
    return uris


def build_feature_schema(
    cat_cols: list[str],
    num_cols: list[str],
) -> dict[str, Any]:
    """Build a JSON-serializable feature schema for model handoff.

    Parameters
    ----------
    cat_cols:
        Categorical feature column names (``pd.Categorical`` at train time).
    num_cols:
        Numeric feature column names.

    Returns
    -------
    dict
        Schema payload saved as ``feature_schema.json``.
    """
    return {
        "model": "xgboost_ranker",
        "categorical_features": list(cat_cols),
        "numeric_features": list(num_cols),
        "feature_columns": list(cat_cols) + list(num_cols),
        "n_features": len(cat_cols) + len(num_cols),
    }


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Select model features and ensure categorical columns use ``pd.Categorical``.

    Parameters
    ----------
    df:
        Input frame containing ``feature_cols``.
    feature_cols:
        Ordered list of model input columns.
    cat_cols:
        Subset of ``feature_cols`` that must be categorical.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for ``XGBClassifier.fit(enable_categorical=True)``.
    """
    x = df[feature_cols].copy()
    for col in cat_cols:
        if col in x.columns and not isinstance(x[col].dtype, pd.CategoricalDtype):
            x[col] = x[col].astype("category")
    return x


def build_xgb_classifier(
    ranking_cfg: dict[str, Any],
    *,
    trial_params: dict[str, Any] | None = None,
) -> Any:
    """Instantiate ``XGBClassifier`` with guide defaults and optional trial overrides.

    Parameters
    ----------
    ranking_cfg:
        Loaded ``configs/models/ranking.yaml``.
    trial_params:
        Optuna-sampled hyperparameters merged on top of defaults.

    Returns
    -------
    XGBClassifier
        Configured classifier using native categorical support.
    """
    from xgboost import XGBClassifier

    params: dict[str, Any] = {
        "n_estimators": ranking_cfg["n_estimators"],
        "learning_rate": ranking_cfg["learning_rate"],
        "max_depth": ranking_cfg["max_depth"],
        "scale_pos_weight": ranking_cfg["scale_pos_weight"],
        "subsample": ranking_cfg["subsample"],
        "colsample_bytree": ranking_cfg["colsample_bytree"],
        "min_child_weight": ranking_cfg["min_child_weight"],
        "reg_lambda": ranking_cfg["reg_lambda"],
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    if trial_params:
        params.update({k: v for k, v in trial_params.items() if k in HPO_PARAM_KEYS})
    return XGBClassifier(
        **params,
        early_stopping_rounds=ranking_cfg["early_stopping_rounds"],
    )


def sample_optuna_params(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    """Sample one hyperparameter set from a YAML-defined Optuna search space.

    Parameters
    ----------
    trial:
        Active Optuna ``Trial`` instance.
    search_space:
        Parsed ``configs/hpo/ranking_search_space.yaml``.

    Returns
    -------
    dict
        Sampled hyperparameters for ``build_xgb_classifier``.
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec.get("type")
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported search space type for {name}: {kind}")
    return params


def val_aucpr(model: Any, x_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Compute validation AUC-PR (average precision) for a fitted classifier.

    Parameters
    ----------
    model:
        Fitted ``XGBClassifier``.
    x_val:
        Validation feature matrix.
    y_val:
        Validation labels.

    Returns
    -------
    float
        Area under the precision-recall curve.
    """
    scores = model.predict_proba(x_val)[:, 1]
    return float(average_precision_score(y_val, scores))


def extract_gain_importance(
    model: Any,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return gain-based feature importance aligned to ``feature_cols`` order.

    Parameters
    ----------
    model:
        Fitted ``XGBClassifier``.
    feature_cols:
        Model input column names in training order.

    Returns
    -------
    pd.DataFrame
        Columns ``feature`` and ``gain``, sorted descending by gain.
    """
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    mapped: dict[str, float] = {}
    for key, value in raw.items():
        if key.startswith("f") and key[1:].isdigit():
            mapped[feature_cols[int(key[1:])]] = float(value)
        else:
            mapped[key] = float(value)
    rows = [{"feature": col, "gain": mapped.get(col, 0.0)} for col in feature_cols]
    out = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    return out


def zero_importance_features(importance_df: pd.DataFrame) -> list[str]:
    """List features with exactly zero gain-based importance.

    Parameters
    ----------
    importance_df:
        Output of ``extract_gain_importance``.

    Returns
    -------
    list[str]
        Feature names to drop before HPO.
    """
    return importance_df.loc[importance_df["gain"] <= 0.0, "feature"].tolist()


def hit_rate_at_k(
    model: Any,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    cat_cols: list[str],
    k: int = 15,
) -> float:
    """Oracle-candidate hit_rate@k on labeled test pairs grouped by user.

    For each user with at least one positive in the test split, rank that user's
    candidate pairs by predicted purchase probability and check whether any
    positive appears in the top-*k* positions.

    Parameters
    ----------
    model:
        Fitted ``XGBClassifier``.
    test_df:
        Test split with ``customer_id``, ``label``, and feature columns.
    feature_cols:
        Model input columns.
    cat_cols:
        Categorical subset of ``feature_cols``.
    k:
        Cutoff for the hit metric (default 15).

    Returns
    -------
    float
        Fraction of users with at least one hit in top-*k*.
    """
    if test_df.empty:
        return float("nan")

    x_test = prepare_feature_matrix(test_df, feature_cols, cat_cols)
    scores = model.predict_proba(x_test)[:, 1]
    scored = test_df[["customer_id", "label"]].copy()
    scored["score"] = scores

    users_with_pos = scored.groupby("customer_id")["label"].max()
    eligible_users = users_with_pos[users_with_pos >= 1].index
    if len(eligible_users) == 0:
        return float("nan")

    hits = 0
    for customer_id in eligible_users:
        user_rows = scored[scored["customer_id"] == customer_id].nlargest(k, "score")
        if (user_rows["label"] >= 1).any():
            hits += 1
    return hits / len(eligible_users)


def export_ranking_best_params(
    best_params: dict[str, Any],
    repo_root: Path | None = None,
) -> Path:
    """Merge HPO best params into ``configs/models/ranking.yaml``.

    Parameters
    ----------
    best_params:
        ``study.best_params`` from Optuna.
    repo_root:
        Repository root. Auto-detected when ``None``.

    Returns
    -------
    Path
        Path to the updated YAML file.
    """
    root = repo_root or find_repo_root()
    config = load_ranking_yaml(root)
    for key, value in best_params.items():
        if key in config:
            config[key] = value
    path = root / "configs" / "models" / "ranking.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def save_model_artifacts(
    model: Any,
    feature_schema: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Persist ``xgboost_ranker.json`` and ``feature_schema.json``.

    Parameters
    ----------
    model:
        Fitted ``XGBClassifier``.
    feature_schema:
        Feature schema dict from ``build_feature_schema``.
    output_dir:
        Destination directory (created if missing).

    Returns
    -------
    dict[str, str]
        Mapping of artifact name to absolute path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_ranker.json"
    schema_path = output_dir / "feature_schema.json"
    model.save_model(str(model_path))
    schema_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    return {"model": str(model_path), "feature_schema": str(schema_path)}
