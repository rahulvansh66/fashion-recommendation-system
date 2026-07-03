"""Point-in-time feature engineering helpers for feature_engineering.ipynb."""

from __future__ import annotations

import math

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def _is_prior_event(h_t_dat, a_snap_date):
    """Return a boolean column expression that is true when history row is <= snap_date."""
    return F.col(h_t_dat) <= F.col(a_snap_date)


def _in_last_n_days(h_t_dat, a_snap_date, days: int):
    """True when the history row is <= snap_date AND within ``days`` days."""
    days_ago = F.datediff(F.col(a_snap_date), F.col(h_t_dat))
    return _is_prior_event(h_t_dat, a_snap_date) & (days_ago <= days)


def _in_same_month_last_year(h_t_dat, a_snap_date):
    """True when history row falls in the same calendar month one year before the snap_date."""
    ly_month_start = F.trunc(F.add_months(F.col(a_snap_date), -12), "MM")
    ly_month_end = F.last_day(ly_month_start)
    return (
        _is_prior_event(h_t_dat, a_snap_date)
        & (F.col(h_t_dat) >= ly_month_start)
        & (F.col(h_t_dat) <= ly_month_end)
    )


def _decay_weight(h_t_dat, a_snap_date, half_life_days: int):
    """Exponential decay weight: w = 2^(-days_ago / half_life_days) for prior rows, 0 otherwise."""
    days_ago = F.datediff(F.col(a_snap_date), F.col(h_t_dat))
    prior = _is_prior_event(h_t_dat, a_snap_date)
    return F.when(
        prior,
        F.pow(F.lit(2.0), -days_ago / F.lit(float(half_life_days))),
    ).otherwise(F.lit(0.0))


def prepare_history_frame(
    transactions_df: DataFrame,
    articles_df: DataFrame,
    category_col: str,
    color_col: str,
) -> DataFrame:
    """Project the base transactions to a history-side alias used in self-joins."""
    article_attrs = articles_df.select(
        "article_id",
        F.col(category_col).alias("item_category"),
        F.col(color_col).alias("item_color"),
    )
    return (
        transactions_df.join(article_attrs, on="article_id", how="left")
        .select(
            F.col("customer_id").alias("h_customer_id"),
            F.col("article_id").alias("h_article_id"),
            F.col("t_dat").alias("h_t_dat"),
            F.col("price").alias("h_price"),
            F.col("item_category").alias("h_item_category"),
            F.col("item_color").alias("h_item_color"),
            F.coalesce(F.col("sales_channel_id"), F.lit(0)).alias("h_sales_channel_id"),
        )
    )


def _aggregate_item_user_cross_features(
    anchors: DataFrame,
    hist: DataFrame,
    half_life_days: int,
) -> DataFrame:
    """Compute all item, user, user-category, and user-item cross features via point-in-time joins."""
    # Item features
    item_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.article_id") == F.col("h.h_article_id"))
        & _is_prior_event("h.h_t_dat", "a.snap_date"),
        "left",
    )

    item_agg = item_join.groupBy("a.anchor_id").agg(
        F.first("a.snap_date").alias("snap_date"),
        F.first("a.customer_id").alias("customer_id"),
        F.first("a.article_id").alias("article_id"),
        F.first("a.item_category").alias("item_category"),
        F.first("a.item_color").alias("item_color"),

        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 7), 1).otherwise(0)).alias("item_pop_7d"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 30), 1).otherwise(0)).alias("item_pop_30d"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 180), 1).otherwise(0)).alias("item_pop_180d"),
        F.sum(F.when(_in_same_month_last_year("h.h_t_dat", "a.snap_date"), 1).otherwise(0)).alias("item_pop_same_month_last_year"),

        F.min("h.h_t_dat").alias("first_sold_date"),
        F.avg("h.h_price").alias("item_avg_price"),
        F.max("h.h_t_dat").alias("item_last_sold_date"),
        F.sum(F.when(F.col("h.h_sales_channel_id") == 2, 1).otherwise(0)).alias("item_sales_channel_2_count"),
    )

    # Category-level item features
    cat_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.item_category") == F.col("h.h_item_category"))
        & _is_prior_event("h.h_t_dat", "a.snap_date"),
        "left",
    )
    cat_agg = cat_join.groupBy("a.anchor_id").agg(
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 30), 1).otherwise(0)).alias("item_category_pop_30d"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 180), 1).otherwise(0)).alias("item_category_pop_180d"),
    )

    # User-level features
    user_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.customer_id") == F.col("h.h_customer_id"))
        & _is_prior_event("h.h_t_dat", "a.snap_date"),
        "left",
    )

    decay_w = _decay_weight("h.h_t_dat", "a.snap_date", half_life_days)

    user_counts = user_join.groupBy("a.anchor_id").agg(
        F.max("h.h_t_dat").alias("last_purchase_date"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 30), 1).otherwise(0)).alias("user_purchase_count_30d"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 180), 1).otherwise(0)).alias("user_purchase_count_180d"),
    )

    user_weighted = user_join.withColumn("decay_w", decay_w)
    user_avg = user_weighted.groupBy("a.anchor_id").agg(
        (F.sum(F.col("h.h_price") * F.col("decay_w")) / F.sum("decay_w")).alias("user_decayed_price_avg")
    )
    user_std = (
        user_weighted.join(user_avg, on="anchor_id", how="left")
        .groupBy("anchor_id")
        .agg(
            F.first("user_decayed_price_avg").alias("user_decayed_price_avg"),
            F.sqrt(
                F.sum(F.col("decay_w") * F.pow(F.col("h.h_price") - F.col("user_decayed_price_avg"), 2))
                / F.sum("decay_w")
            ).alias("user_decayed_price_std_raw"),
        )
        .withColumn(
            "user_decayed_price_std",
            F.when(F.col("user_decayed_price_std_raw") < 1e-6, F.lit(1e-6)).otherwise(F.col("user_decayed_price_std_raw")),
        )
        .drop("user_decayed_price_std_raw")
    )
    user_agg = user_counts.join(user_std, on="anchor_id", how="left")

    # User–item cross features
    user_item_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.customer_id") == F.col("h.h_customer_id"))
        & (F.col("a.article_id") == F.col("h.h_article_id"))
        & _is_prior_event("h.h_t_dat", "a.snap_date"),
        "left",
    )
    cross_decay_w = _decay_weight("h.h_t_dat", "a.snap_date", half_life_days)

    cross_agg = user_item_join.groupBy("a.anchor_id").agg(
        F.count("h.h_t_dat").alias("user_item_repurchase"),
        F.sum(cross_decay_w).alias("user_item_decayed_repurchase"),
        F.avg("h.h_price").alias("candidate_price_from_cross"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 30), 1).otherwise(0)).alias("user_item_repurchase_30d"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 90), 1).otherwise(0)).alias("user_item_repurchase_90d"),
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 365), 1).otherwise(0)).alias("user_item_repurchase_365d"),
        F.max("h.h_t_dat").alias("user_item_last_purchase_date"),
        F.sum(F.when(F.col("h.h_sales_channel_id") == 2, 1).otherwise(0)).alias("user_item_sales_channel_2_count"),
    )

    # User–category cross features
    user_cat_join = anchors.alias("a").join(
        hist.alias("h"),
        (F.col("a.customer_id") == F.col("h.h_customer_id"))
        & (F.col("a.item_category") == F.col("h.h_item_category"))
        & _is_prior_event("h.h_t_dat", "a.snap_date"),
        "left",
    )
    user_cat_agg = user_cat_join.groupBy("a.anchor_id").agg(
        F.sum(F.when(_in_last_n_days("h.h_t_dat", "a.snap_date", 365), 1).otherwise(0)).alias("user_purchases_in_candidate_category_1y"),
        F.max("h.h_t_dat").alias("user_last_cat_purchase_date"),
    )

    # Merge all aggregates
    merged = (
        item_agg.join(cat_agg, on="anchor_id", how="left")
        .join(user_agg, on="anchor_id", how="left")
        .join(cross_agg, on="anchor_id", how="left")
        .join(user_cat_agg, on="anchor_id", how="left")
    )

    # candidate_price
    merged = merged.withColumn(
        "candidate_price",
        F.coalesce(F.col("candidate_price_from_cross"), F.col("item_avg_price")),
    ).drop("candidate_price_from_cross")

    # Derived item features
    merged = merged.withColumn(
        "days_since_first_sold",
        F.when(F.col("first_sold_date").isNotNull(), F.datediff(F.col("snap_date"), F.col("first_sold_date"))),
    ).withColumn(
        "item_days_since_last_sold",
        F.when(F.col("item_last_sold_date").isNotNull(), F.datediff(F.col("snap_date"), F.col("item_last_sold_date"))),
    ).withColumn(
        "item_sales_channel_2_share",
        F.col("item_sales_channel_2_count") / (F.col("item_pop_180d") + F.lit(1)),
    )

    merged = merged.withColumn(
        "item_recent_to_last_180d_ratio",
        F.col("item_pop_30d") / (F.col("item_category_pop_180d") + F.lit(1)),
    ).withColumn(
        "item_category_recent_to_lifetime_ratio",
        F.col("item_category_pop_30d") / (F.col("item_category_pop_180d") + F.lit(1)),
    ).withColumn(
        "item_seasonality_strength",
        F.col("item_pop_7d") / (F.col("item_pop_same_month_last_year") + F.lit(1)),
    )

    # Derived user features
    merged = merged.withColumn(
        "user_days_since_last_purchase",
        F.when(F.col("last_purchase_date").isNotNull(), F.datediff(F.col("snap_date"), F.col("last_purchase_date"))),
    )

    # Derived user–item cross features
    merged = merged.withColumn(
        "user_item_decayed_interaction_ratio",
        F.col("user_item_decayed_repurchase") / (F.col("item_pop_180d") + F.lit(1)),
    ).withColumn(
        "user_item_price_decayed_zscore",
        (F.col("candidate_price") - F.col("user_decayed_price_avg")) / F.col("user_decayed_price_std"),
    ).withColumn(
        "user_item_days_since_last_purchase",
        F.when(F.col("user_item_last_purchase_date").isNotNull(), F.datediff(F.col("snap_date"), F.col("user_item_last_purchase_date"))),
    )

    # Derived user–category cross features
    merged = merged.withColumn(
        "user_days_since_last_purchase_in_category",
        F.when(F.col("user_last_cat_purchase_date").isNotNull(), F.datediff(F.col("snap_date"), F.col("user_last_cat_purchase_date"))),
    )

    # Cyclical time features
    month = F.month(F.col("snap_date"))
    merged = merged.withColumn(
        "txn_month_sin",
        F.sin(month * F.lit(2.0 * math.pi / 12.0)),
    ).withColumn(
        "txn_month_cos",
        F.cos(month * F.lit(2.0 * math.pi / 12.0)),
    )

    return merged


def _add_category_color_preferences(
    anchors: DataFrame,
    hist: DataFrame,
    color_top_n: int = 2,
    category_top_n: int = 3,
) -> DataFrame:
    def top_n_for(value_name: str, prefix: str, n: int) -> DataFrame:
        pref_join = anchors.alias("a").join(
            hist.alias("h"),
            (F.col("a.customer_id") == F.col("h.h_customer_id"))
            & _is_prior_event("h.h_t_dat", "a.snap_date")
            & (F.col("h.h_t_dat") >= F.date_sub(F.col("a.snap_date"), 365)),
            "inner",
        )
        ranked = (
            pref_join.groupBy("a.anchor_id", F.col(value_name))
            .agg(F.count("*").alias("cnt"))
            .withColumn(
                "rank",
                F.row_number().over(
                    Window.partitionBy("a.anchor_id").orderBy(F.desc("cnt"), F.asc(value_name))
                ),
            )
            .filter(F.col("rank") <= n)
        )
        result = anchors.select("anchor_id")
        for rank in range(1, n + 1):
            rank_df = ranked.filter(F.col("rank") == rank).select(
                F.col("a.anchor_id").alias("anchor_id"),
                F.col(value_name).alias(f"{prefix}_rank{rank}"),
            )
            result = result.join(rank_df, on="anchor_id", how="left")
        return result

    cat_prefs = top_n_for("h.h_item_category", "user_category_pref_1y", category_top_n)
    color_prefs = top_n_for("h.h_item_color", "user_color_pref_1y", color_top_n)
    return cat_prefs.join(color_prefs, on="anchor_id", how="left")


def build_features(
    anchors_df: DataFrame,
    transactions_df: DataFrame,
    articles_df: DataFrame,
    customers_df: DataFrame,
    category_col: str,
    color_col: str,
    half_life_days: int,
    color_pref_top_n: int = 2,
    category_pref_top_n: int = 3,
) -> DataFrame:
    """Build point-in-time features for a set of anchor pairs at a specific snap_date.

    Parameters
    ----------
    anchors_df : DataFrame
        Pairs to score. Must contain `customer_id`, `article_id`, and `snap_date`.
    transactions_df : DataFrame
        Raw transactions (t_dat, customer_id, article_id, price, sales_channel_id).
    articles_df : DataFrame
        Full article master.
    customers_df : DataFrame
        Full customer master.
    category_col : str
        Article column to use as category.
    color_col : str
        Article column to use as colour.
    half_life_days : int
        Half-life for exponential decay weights.
    color_pref_top_n : int
        How many top-colour ranks to compute.
    category_pref_top_n : int
        How many top-category ranks to compute.

    Returns
    -------
    DataFrame
        One row per anchor with all engineered features.
    """
    # Add monotonic ID to anchors to ensure safe joins
    anchors = anchors_df.withColumn("anchor_id", F.monotonically_increasing_id())

    # Enrich anchors with item_category and item_color
    article_attrs = articles_df.select(
        "article_id",
        F.col(category_col).alias("item_category"),
        F.col(color_col).alias("item_color"),
    )
    anchors = anchors.join(article_attrs, on="article_id", how="left")

    hist = prepare_history_frame(transactions_df, articles_df, category_col, color_col)

    features = _aggregate_item_user_cross_features(anchors, hist, half_life_days)
    prefs = _add_category_color_preferences(
        anchors,
        hist,
        color_top_n=color_pref_top_n,
        category_top_n=category_pref_top_n,
    )
    enriched = features.join(prefs, on="anchor_id", how="left")

    # Add category-match binary flags
    for rank in range(1, category_pref_top_n + 1):
        enriched = enriched.withColumn(
            f"user_category_match_rank{rank}",
            F.when(
                F.col("item_category") == F.col(f"user_category_pref_1y_rank{rank}"),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )

    # Pass through customer columns
    customer_cols = [c for c in customers_df.columns if c != "customer_id"]
    enriched = enriched.join(
        anchors.select("anchor_id", "customer_id").join(customers_df, on="customer_id", how="left"),
        on=["anchor_id", "customer_id"],
        how="left",
    )

    # Pass through article catalog attributes
    article_cols = [
        c for c in articles_df.columns if c not in {"article_id", category_col, color_col}
    ]
    enriched = enriched.join(
        articles_df.select("article_id", *article_cols),
        on="article_id",
        how="left",
    )

    # Also join back any extra columns from anchors_df (like label)
    anchor_extra_cols = [
        c for c in anchors_df.columns if c not in {"customer_id", "article_id", "snap_date"}
    ]
    if anchor_extra_cols:
        enriched = enriched.join(
            anchors.select("anchor_id", *anchor_extra_cols),
            on="anchor_id",
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
        "item_pop_same_month_last_year",
        "first_sold_date",
        "days_since_first_sold",
        "item_recent_to_last_180d_ratio",
        "item_category_recent_to_lifetime_ratio",
        "item_seasonality_strength",
        "item_avg_price",
        "item_days_since_last_sold",
        "item_sales_channel_2_count",
        "item_sales_channel_2_share",
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
        "user_item_repurchase_30d",
        "user_item_repurchase_90d",
        "user_item_repurchase_365d",
        "user_item_days_since_last_purchase",
        "user_item_sales_channel_2_count",
        "user_purchases_in_candidate_category_1y",
        "user_days_since_last_purchase_in_category",
        "user_category_match_rank1",
        "user_category_match_rank2",
        "user_category_match_rank3",
        "candidate_price",
        "user_item_price_decayed_zscore",
        "txn_month_sin",
        "txn_month_cos",
    ]

    output_cols = (
        ["customer_id", "article_id", "snap_date"]
        + anchor_extra_cols
        + customer_cols
        + article_cols
        + feature_cols
    )
    return enriched.select(*output_cols)
