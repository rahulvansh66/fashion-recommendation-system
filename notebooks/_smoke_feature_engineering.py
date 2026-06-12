"""Smoke test for point-in-time feature engineering."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

_hadoop_home = REPO / ".hadoop-win"
if sys.platform == "win32" and (_hadoop_home / "bin" / "winutils.exe").exists():
    os.environ["HADOOP_HOME"] = str(_hadoop_home.resolve())
    _bin = str((_hadoop_home / "bin").resolve())
    if _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

from config_loader import load_feature_engineering_config
from feature_engineering_core import build_enriched_transactions
from pyspark.sql import SparkSession


def main() -> None:
    config = load_feature_engineering_config(repo_root=REPO, environment="local_dev")
    dataset = "dummy"
    local_s3 = REPO / config["local_s3_root"] / "dataset" / dataset

    spark = (
        SparkSession.builder.appName("smoke-feature-engineering")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    articles = spark.read.parquet(str(local_s3 / "articles"))
    customers = spark.read.parquet(str(local_s3 / "customers"))
    transactions = spark.read.parquet(str(local_s3 / "transactions"))

    enriched = build_enriched_transactions(
        transactions,
        articles,
        customers,
        config["category_col"],
        config["color_col"],
        config["decay_half_life_days"],
        color_pref_top_n=2,
        category_pref_top_n=3,
    )

    cols = enriched.columns
    required = [
        "item_recent_to_last_180d_ratio",
        "item_seasonality_strength",
        "user_color_pref_1y_rank1",
        "user_color_pref_1y_rank2",
        "txn_month_sin",
        "txn_month_cos",
    ]
    forbidden = ["feature_cutoff", "user_color_pref_1y_rank3", "item_recent_to_lifetime_ratio"]

    missing = [c for c in required if c not in cols]
    present_forbidden = [c for c in forbidden if c in cols]
    if missing:
        raise AssertionError(f"Missing columns: {missing}")
    if present_forbidden:
        raise AssertionError(f"Forbidden columns present: {present_forbidden}")

    row_count = enriched.count()
    if row_count != transactions.count():
        raise AssertionError(
            f"Row count mismatch: enriched={row_count}, transactions={transactions.count()}"
        )

    sample = enriched.orderBy("t_dat").limit(3).collect()
    print(f"OK: {row_count} enriched transaction rows")
    for row in sample:
        print(
            f"  t_dat={row.t_dat} item_pop_7d={row.item_pop_7d} "
            f"seasonality={row.item_seasonality_strength:.3f} "
            f"txn_month_sin={row.txn_month_sin:.3f}"
        )

    spark.stop()


if __name__ == "__main__":
    main()
