import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

# Add the 'notebooks' directory to the Python path
_nb_dir = Path(__file__).resolve().parent.parent
if str(_nb_dir) not in sys.path:
    sys.path.insert(0, str(_nb_dir))

from utils.config_loader import is_glue_runtime, load_feature_engineering_config
from utils.spark_session import create_spark_session
from utils.feature_engineering_core import build_features

CONFIG = load_feature_engineering_config(environment="local_dev")
STORAGE_MODE = CONFIG["storage_mode"]
IS_GLUE = is_glue_runtime(CONFIG["runtime_mode"])

spark = create_spark_session("feature-engineering", is_glue=IS_GLUE, pin_python=True)

def resolve_repo_root() -> Path:
    for candidate in (Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent):
        if (candidate / "requirements-notebooks.txt").exists():
            return candidate
    return Path.cwd()

def storage_uri(relative_path: str) -> str:
    relative_path = relative_path.strip("/").replace("\\", "/")
    if STORAGE_MODE == "aws":
        return f"s3://{CONFIG['s3_bucket']}/{relative_path}"
    local_root = (resolve_repo_root() / CONFIG["local_s3_root"]).resolve()
    return str((local_root / relative_path).resolve())

def write_parquet_dataset(df: DataFrame, relative_path: str, partition_cols: list[str] | None = None) -> str:
    dest = storage_uri(relative_path)
    if STORAGE_MODE == "local":
        dest_path = Path(dest)
        if dest_path.exists():
            shutil.rmtree(dest_path)
    writer = df.write.mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(dest)
    return dest

def read_staged_table(spark: SparkSession, dataset_name: str, table: str) -> DataFrame:
    path = storage_uri(f"dataset/{dataset_name}/{table}")
    return spark.read.parquet(path)

def run_feature_engineering_for_dataset(dataset_name: str):
    print(f"==================================================")
    print(f"Running feature engineering for dataset: {dataset_name}")
    print(f"==================================================")
    
    articles_df = read_staged_table(spark, dataset_name, "articles")
    customers_df = read_staged_table(spark, dataset_name, "customers")
    transactions_df = read_staged_table(spark, dataset_name, "transactions")

    print(f"articles: {articles_df.count():,}")
    print(f"customers: {customers_df.count():,}")
    print(f"transactions: {transactions_df.count():,}")

    snap_date = CONFIG["cutoff_date"]
    label_end = CONFIG["test_end"]

    # 1. Positives: Purchases in the label window
    pos_df = transactions_df.filter(
        (F.col("t_dat") > snap_date) & (F.col("t_dat") <= label_end)
    ).select("customer_id", "article_id").distinct()
    pos_df = pos_df.withColumn("label", F.lit(1)).withColumn("snap_date", F.to_date(F.lit(snap_date)))

    # 2. Negatives: 5 random articles per positive
    window_spec = Window.partitionBy("customer_id").orderBy(F.rand(seed=42))
    neg_df = pos_df.select("customer_id").crossJoin(
        transactions_df.select("article_id").distinct().sample(fraction=0.1, seed=42)
    ).withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") <= 5).drop("rank")
    neg_df = neg_df.withColumn("label", F.lit(0)).withColumn("snap_date", F.to_date(F.lit(snap_date)))

    # Combine into anchors
    anchors_df = pos_df.unionByName(neg_df)

    enriched_txn_df = build_features(
        anchors_df,
        transactions_df,
        articles_df,
        customers_df,
        CONFIG["category_col"],
        CONFIG["color_col"],
        CONFIG["decay_half_life_days"],
        color_pref_top_n=2,
        category_pref_top_n=CONFIG["user_pref_top_n"],
    )

    print(f"anchors (pos + neg): {anchors_df.count():,}")
    print(f"enriched features: {enriched_txn_df.count():,}")
    print(f"total columns: {len(enriched_txn_df.columns)}")

    print("\n-- Item popularity + demand ratios --")
    enriched_txn_df.select(
        "customer_id", "article_id", "snap_date", "label",
        "item_pop_7d", "item_pop_30d", "item_pop_180d",
        "item_recent_to_last_180d_ratio",
        "item_seasonality_strength",
    ).show(5, truncate=False)

    print("\n-- New: item price / recency / channel --")
    enriched_txn_df.select(
        "customer_id", "article_id", "snap_date",
        "item_avg_price",
        "item_days_since_last_sold",
        "item_sales_channel_2_count",
        "item_sales_channel_2_share",
    ).show(5, truncate=False)

    print("\n-- New user-item windowed cross features --")
    enriched_txn_df.select(
        "customer_id", "article_id", "snap_date",
        "user_item_repurchase",
        "user_item_repurchase_30d",
        "user_item_repurchase_90d",
        "user_item_repurchase_365d",
        "user_item_days_since_last_purchase",
        "user_item_sales_channel_2_count",
    ).show(5, truncate=False)

    print("\n-- New user-category cross features --")
    enriched_txn_df.select(
        "customer_id", "article_id", "snap_date",
        "item_category",
        "user_category_pref_1y_rank1",
        "user_purchases_in_candidate_category_1y",
        "user_days_since_last_purchase_in_category",
        "user_category_match_rank1",
        "user_category_match_rank2",
        "user_category_match_rank3",
    ).show(5, truncate=False)

    features_base = f"dataset/{dataset_name}/features"
    feature_outputs = {
        "ranking_dataset": write_parquet_dataset(
            enriched_txn_df,
            f"{features_base}/ranking_dataset",
            partition_cols=["snap_date"],
        ),
    }

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage_mode": CONFIG["storage_mode"],
        "dataset": dataset_name,
        "transaction_rows": enriched_txn_df.count(),
        "feature_columns": enriched_txn_df.columns,
        "feature_outputs": feature_outputs,
    }
    print(json.dumps(summary, indent=2))
    print("\n\n")

if __name__ == "__main__":
    run_feature_engineering_for_dataset("dummy")
    run_feature_engineering_for_dataset("sample_2000_users")
