"""Point-in-time feature engineering helpers for feature_engineering.ipynb."""

from __future__ import annotations

import math

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def prepare_transaction_base(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    customers_df: DataFrame,
    category_col: str,
    color_col: str,
) -> DataFrame:
    """Join item and user dimension attributes onto transactions."""
    article_attrs = articles_df.select(
        "article_id",
        F.col(category_col).alias("item_category"),
        F.col(color_col).alias("item_color"),
    )

    return (
        transactions_df.join(article_attrs, on="article_id", how="left")
        .join(customers_df, on="customer_id", how="left")
        .withColumn("txn_id", F.monotonically_increasing_id())
        .withColumn(
            "txn_seq",
            F.row_number().over(
                Window.partitionBy("customer_id", "t_dat").orderBy("txn_id")
            ),
        )
    )


def _is_prior_event(h_t_dat, h_txn_id, a_t_dat, a_txn_id):
    return (F.col(h_t_dat) < F.col(a_t_dat)) | (
        (F.col(h_t_dat) == F.col(a_t_dat)) & (F.col(h_txn_id) < F.col(a_txn_id))
    )


def _in_last_n_days(h_t_dat, a_t_dat, h_txn_id, a_txn_id, days: int):
    days_ago = F.datediff(F.col(a_t_dat), F.col(h_t_dat))
    return _is_prior_event(h_t_dat, h_txn_id, a_t_dat, a_txn_id) & (days_ago <= days)


def _in_same_7d_last_year(h_t_dat, a_t_dat, h_txn_id, a_txn_id):
    ly_end = F.date_sub(F.add_months(F.col(a_t_dat), -12), 0)
    ly_start = F.date_sub(ly_end, 7)
    return (
        _is_prior_event(h_t_dat, h_txn_id, a_t_dat, a_txn_id)
        & (F.col(h_t_dat) >= ly_start)
        & (F.col(h_t_dat) < ly_end)
    )


def _decay_weight(h_t_dat, a_t_dat, h_txn_id, a_txn_id, half_life_days: int):
    days_ago = F.datediff(F.col(a_t_dat), F.col(h_t_dat))
    prior = _is_prior_event(h_t_dat, h_txn_id, a_t_dat, a_txn_id)
    return F.when(
        prior,
        F.pow(F.lit(2.0), -days_ago / F.lit(float(half_life_days))),
    ).otherwise(F.lit(0.0))


def _history_frame(base_df: DataFrame) -> DataFrame:
    return base_df.select(
        F.col("txn_id").alias("h_txn_id"),
        F.col("customer_id").alias("h_customer_id"),
        F.col("article_id").alias("h_article_id"),
        F.col("t_dat").alias("h_t_dat"),
        F.col("price").alias("h_price"),
        F.col("item_category").alias("h_item_category"),
        F.col("item_color").alias("h_item_color"),
    )


def _aggregate_item_user_cross_features(
    anchors: DataFrame,
    hist: DataFrame,
    half_life_days: int,
) -> DataFrame:
    item_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.article_id") == F.col("h.h_article_id"))
        & _is_prior_event("h.h_t_dat", "h.h_txn_id", "a.t_dat", "a.txn_id"),
        "left",
    )

    item_agg = item_join.groupBy(F.col("a.txn_id").alias("txn_id")).agg(
        F.first("a.t_dat").alias("t_dat"),
        F.first("a.customer_id").alias("customer_id"),
        F.first("a.article_id").alias("article_id"),
        F.first("a.price").alias("price"),
        F.first("a.item_category").alias("item_category"),
        F.first("a.item_color").alias("item_color"),
        F.first("a.txn_seq").alias("txn_seq"),
        F.sum(
            F.when(_in_last_n_days("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", 7), 1).otherwise(0)
        ).alias("item_pop_7d"),
        F.sum(
            F.when(_in_last_n_days("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", 30), 1).otherwise(0)
        ).alias("item_pop_30d"),
        F.sum(
            F.when(_in_last_n_days("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", 180), 1).otherwise(0)
        ).alias("item_pop_180d"),
        F.sum(F.when(_in_same_7d_last_year("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id"), 1).otherwise(0)).alias(
            "item_pop_same_7d_last_year"
        ),
        F.min("h.h_t_dat").alias("first_sold_date"),
    )

    cat_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.item_category") == F.col("h.h_item_category"))
        & _is_prior_event("h.h_t_dat", "h.h_txn_id", "a.t_dat", "a.txn_id"),
        "left",
    )
    cat_agg = cat_join.groupBy(F.col("a.txn_id").alias("txn_id")).agg(
        F.sum(
            F.when(_in_last_n_days("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", 30), 1).otherwise(0)
        ).alias("item_category_pop_30d"),
        F.sum(
            F.when(_in_last_n_days("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", 180), 1).otherwise(0)
        ).alias("item_category_pop_180d"),
    )

    user_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.customer_id") == F.col("h.h_customer_id"))
        & _is_prior_event("h.h_t_dat", "h.h_txn_id", "a.t_dat", "a.txn_id"),
        "left",
    )

    decay_w = _decay_weight("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", half_life_days)

    user_counts = user_join.groupBy(F.col("a.txn_id").alias("txn_id")).agg(
        F.max("h.h_t_dat").alias("last_purchase_date"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", 30), 1).otherwise(0)).alias(
            "user_purchase_count_30d"
        ),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", 180), 1).otherwise(0)).alias(
            "user_purchase_count_180d"
        ),
    )
    user_weighted = user_join.withColumn("decay_w", decay_w)
    user_avg = user_weighted.groupBy(F.col("a.txn_id").alias("txn_id")).agg(
        (F.sum(F.col("h.h_price") * F.col("decay_w")) / F.sum("decay_w")).alias(
            "user_decayed_price_avg"
        )
    )
    user_std = (
        user_weighted.join(user_avg, on="txn_id", how="left")
        .groupBy(F.col("a.txn_id").alias("txn_id"))
        .agg(
            F.first("user_decayed_price_avg").alias("user_decayed_price_avg"),
            F.sqrt(
                F.sum(
                    F.col("decay_w")
                    * F.pow(F.col("h.h_price") - F.col("user_decayed_price_avg"), 2)
                )
                / F.sum("decay_w")
            ).alias("user_decayed_price_std_raw"),
        )
        .withColumn(
            "user_decayed_price_std",
            F.when(F.col("user_decayed_price_std_raw") < 1e-6, F.lit(1e-6)).otherwise(
                F.col("user_decayed_price_std_raw")
            ),
        )
        .drop("user_decayed_price_std_raw")
    )
    user_agg = user_counts.join(user_std, on="txn_id", how="left")

    user_item_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.customer_id") == F.col("h.h_customer_id"))
        & (F.col("a.article_id") == F.col("h.h_article_id"))
        & _is_prior_event("h.h_t_dat", "h.h_txn_id", "a.t_dat", "a.txn_id"),
        "left",
    )
    cross_decay_w = _decay_weight("h.h_t_dat", "a.t_dat", "h.h_txn_id", "a.txn_id", half_life_days)
    cross_agg = user_item_join.groupBy(F.col("a.txn_id").alias("txn_id")).agg(
        F.count("h.h_txn_id").alias("user_item_repurchase"),
        F.sum(cross_decay_w).alias("user_item_decayed_repurchase"),
        F.avg("h.h_price").alias("candidate_price"),
    )

    item_price_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.article_id") == F.col("h.h_article_id"))
        & _is_prior_event("h.h_t_dat", "h.h_txn_id", "a.t_dat", "a.txn_id"),
        "left",
    )
    item_price_agg = item_price_join.groupBy(F.col("a.txn_id").alias("txn_id")).agg(
        F.avg("h.h_price").alias("candidate_price_from_item_hist")
    )

    merged = (
        item_agg.join(cat_agg, on="txn_id", how="left")
        .join(user_agg, on="txn_id", how="left")
        .join(cross_agg, on="txn_id", how="left")
        .join(item_price_agg, on="txn_id", how="left")
    )

    merged = merged.withColumn(
        "candidate_price",
        F.coalesce(F.col("candidate_price"), F.col("candidate_price_from_item_hist")),
    ).drop("candidate_price_from_item_hist")

    merged = merged.withColumn(
        "days_since_first_sold",
        F.when(
            F.col("first_sold_date").isNotNull(),
            F.datediff(F.col("t_dat"), F.col("first_sold_date")),
        ),
    ).withColumn(
        "user_days_since_last_purchase",
        F.when(
            F.col("last_purchase_date").isNotNull(),
            F.datediff(F.col("t_dat"), F.col("last_purchase_date")),
        ),
    )

    merged = merged.withColumn(
        "item_recent_to_last_180d_ratio",
        F.col("item_pop_30d") / (F.col("item_category_pop_180d") + F.lit(1)),
    ).withColumn(
        "item_category_recent_to_lifetime_ratio",
        F.col("item_category_pop_30d") / (F.col("item_category_pop_180d") + F.lit(1)),
    ).withColumn(
        "item_seasonality_strength",
        F.col("item_pop_7d") / (F.col("item_pop_same_7d_last_year") + F.lit(1)),
    ).withColumn(
        "user_item_decayed_interaction_ratio",
        F.col("user_item_decayed_repurchase") / (F.col("item_pop_180d") + F.lit(1)),
    ).withColumn(
        "user_item_price_decayed_zscore",
        (F.col("candidate_price") - F.col("user_decayed_price_avg"))
        / F.col("user_decayed_price_std"),
    )

    month = F.month(F.col("t_dat"))
    merged = merged.withColumn(
        "txn_month_sin",
        F.sin(month * F.lit(2.0 * math.pi / 12.0)),
    ).withColumn(
        "txn_month_cos",
        F.cos(month * F.lit(2.0 * math.pi / 12.0)),
    )

    return merged


def _add_category_color_preferences(
    base_df: DataFrame,
    hist: DataFrame,
    color_top_n: int = 2,
    category_top_n: int = 3,
) -> DataFrame:
    def top_n_for(value_name: str, prefix: str, n: int) -> DataFrame:
        pref_join = base_df.select("txn_id", "customer_id", "t_dat", "txn_seq").alias("a").join(
            hist.alias("h"),
            (F.col("a.customer_id") == F.col("h.h_customer_id"))
            & _is_prior_event("h.h_t_dat", "h.h_txn_id", "a.t_dat", "a.txn_id")
            & (F.col("h.h_t_dat") >= F.date_sub(F.col("a.t_dat"), 365)),
            "inner",
        )
        ranked = (
            pref_join.groupBy("a.txn_id", F.col(value_name))
            .agg(F.count("*").alias("cnt"))
            .withColumn(
                "rank",
                F.row_number().over(
                    Window.partitionBy("a.txn_id").orderBy(F.desc("cnt"), F.asc(value_name))
                ),
            )
            .filter(F.col("rank") <= n)
        )
        result = base_df.select("txn_id")
        for rank in range(1, n + 1):
            rank_df = ranked.filter(F.col("rank") == rank).select(
                F.col("a.txn_id").alias("txn_id"),
                F.col(value_name).alias(f"{prefix}_rank{rank}"),
            )
            result = result.join(rank_df, on="txn_id", how="left")
        return result

    cat_prefs = top_n_for("h.h_item_category", "user_category_pref_1y", category_top_n)
    color_prefs = top_n_for("h.h_item_color", "user_color_pref_1y", color_top_n)
    return cat_prefs.join(color_prefs, on="txn_id", how="left")


def build_enriched_transactions(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    customers_df: DataFrame,
    category_col: str,
    color_col: str,
    half_life_days: int,
    color_pref_top_n: int = 2,
    category_pref_top_n: int = 3,
) -> DataFrame:
    """Build point-in-time features for every transaction row."""
    base_df = prepare_transaction_base(
        transactions_df,
        articles_df,
        customers_df,
        category_col,
        color_col,
    )
    hist = _history_frame(base_df)
    features = _aggregate_item_user_cross_features(base_df, hist, half_life_days)
    prefs = _add_category_color_preferences(
        base_df,
        hist,
        color_top_n=color_pref_top_n,
        category_top_n=category_pref_top_n,
    )
    enriched = features.join(prefs, on="txn_id", how="left")

    customer_cols = [c for c in customers_df.columns if c != "customer_id"]
    txn_extra_cols = [
        c
        for c in transactions_df.columns
        if c not in {"customer_id", "article_id", "t_dat", "price"}
    ]
    enriched = enriched.join(
        base_df.select("txn_id", *customer_cols, *txn_extra_cols),
        on="txn_id",
        how="left",
    )

    article_cols = [
        c
        for c in articles_df.columns
        if c not in {"article_id", category_col, color_col}
    ]
    enriched = enriched.join(
        articles_df.select("article_id", *article_cols),
        on="article_id",
        how="left",
    )

    feature_cols = [
        "item_category",
        "item_color",
        "item_pop_7d",
        "item_pop_30d",
        "item_pop_180d",
        "item_category_pop_30d",
        "item_category_pop_180d",
        "item_pop_same_7d_last_year",
        "first_sold_date",
        "days_since_first_sold",
        "item_recent_to_last_180d_ratio",
        "item_category_recent_to_lifetime_ratio",
        "item_seasonality_strength",
        "user_category_pref_1y_rank1",
        "user_category_pref_1y_rank2",
        "user_category_pref_1y_rank3",
        "user_color_pref_1y_rank1",
        "user_color_pref_1y_rank2",
        "user_days_since_last_purchase",
        "user_purchase_count_30d",
        "user_purchase_count_180d",
        "user_decayed_price_avg",
        "user_decayed_price_std",
        "user_item_repurchase",
        "user_item_decayed_repurchase",
        "user_item_decayed_interaction_ratio",
        "candidate_price",
        "user_item_price_decayed_zscore",
        "txn_month_sin",
        "txn_month_cos",
    ]
    output_cols = (
        list(transactions_df.columns)
        + customer_cols
        + article_cols
        + feature_cols
    )
    return enriched.select(*output_cols)
