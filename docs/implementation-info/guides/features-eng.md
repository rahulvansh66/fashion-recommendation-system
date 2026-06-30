# Feature Engineering Guide

Features for the data pipeline to capture **trend**, **seasonality**, **recency**, and related signals for retrieval and ranker models.

**Contract:** [`v1-requirements.md`](../../system-design/v1/v1-requirements.md) FR-BATCH-02, [`ranking-model-training-guide.md`](./ranking-model-training-guide.md), [`two-tower-retrieval-training-guide.md`](../two-tower-model/two-tower-retrieval-training-guide.md)

## Conventions

### Temporal splits and feature cutoffs

All offline pipelines share the same snap-date schedule ([`v1-requirements.md`](../../system-design/v1/v1-requirements.md) FR-BATCH-02).
Each snap defines a **feature cutoff** (`t_dat <= snap_date`) and a **7-day forward label window**.

| Snap date | Role | Feature cutoff (`t_dat <=`) | Label window (`t_dat` range) |
|-----------|------|------------------------------|------------------------------|
| `2020-03-24` | **Train 0** | 2020-03-24 | 2020-03-25 – 2020-03-31 |
| `2020-03-31` | **Train 1** | 2020-03-31 | 2020-04-01 – 2020-04-07 |
| `2020-04-07` | **Train 2** | 2020-04-07 | 2020-04-08 – 2020-04-14 |
| `2020-04-14` | **Val 1** | 2020-04-14 | 2020-04-15 – 2020-04-21 |
| `2020-04-28` | **Val 2** | 2020-04-28 | 2020-04-29 – 2020-05-05 |
| `2020-05-15` | **Test** | 2020-05-15 | 2020-05-16 – 2020-05-22 |
| `2020-05-31` | **Drift 1** | 2020-05-31 | 2020-06-01 – 2020-06-07 |
| `2020-06-30` | **Drift 2** | 2020-06-30 | 2020-07-01 – 2020-07-07 |
| `2020-07-31` | **Drift 3** | 2020-07-31 | 2020-08-01 – 2020-08-07 |
| `2020-08-31` | **Drift 4** | 2020-08-31 | 2020-09-01 – 2020-09-07 |
| `2020-09-15` | **Drift 5** | 2020-09-15 | 2020-09-16 – 2020-09-22 |

`cutoff` in the feature tables below always means `snap_date` for the snap being built — never a post-cutoff label date.

**Ranker labels (separate from features):** positives are `(customer_id, article_id)` purchases with `t_dat` in the snap's label window; negatives are **10 window-aware** non-purchases per positive (`seen` exclusion before `snap_date`). Detail: [`ranking-model-training-guide.md`](./ranking-model-training-guide.md).

**Online inference:** user and item features are served from precomputed `features/` artifacts; `txn_month_sin` / `txn_month_cos` use the request `current_date` (not a historical cutoff).


| Topic                  | Rule                                                                                                                                                                                                                              |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Category               | Always use `garment_group_name` for category-level features.                                                                                                                                                                      |
| Hard-window vs decayed | Do **not** duplicate hard-window count features (`item_pop_`*, `user_purchase_count_*`) with decayed variants. Use decay only where hard windows are a poor fit (repurchase timing, price preference, personal-vs-global demand). |


### Decay weight (decayed features only)

Recent transactions receive high weight; older transactions still contribute but less. All decayed features use **half-life = 180 days** relative to `cutoff`.

```
days_ago = cutoff - t_dat   (only rows with t_dat <= cutoff)
w = 2^(-days_ago / 180) = exp(-ln(2) × days_ago / 180)
λ = ln(2) / 180
```


| `days_ago` | Weight `w` |
| ---------- | ---------- |
| 0          | 1.0        |
| 180        | 0.5        |
| 360        | 0.25       |


---

## Item features

Popularity, demand, price, recency, and channel signals at the item and category level.


| Feature                                  | Purpose                              | Formula                                                                                                                              | Look back                  |
| ---------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `item_pop_7d`                            | Current trend                        | `COUNT(*)` of purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 7d, cutoff]`                                      | 7 days                     |
| `item_pop_30d`                           | Recent popularity                    | `COUNT(*)` of purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 30d, cutoff]`                                     | 30 days                    |
| `item_category_pop_30d`                  | Recent popularity of item's category | `COUNT(*)` of purchases where `garment_group_name` = candidate item's category and `t_dat ∈ [cutoff - 30d, cutoff]`                  | 30 days                    |
| `item_pop_180d`                          | Stable demand                        | `COUNT(*)` of purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 180d, cutoff]`                                    | 6 months (180 days)        |
| `item_category_pop_180d`                 | Stable popularity of item's category | `COUNT(*)` of purchases where `garment_group_name` = candidate item's category and `t_dat ∈ [cutoff - 180d, cutoff]`                 | 6 months (180 days)        |
| `item_pop_same_month_last_year`          | Seasonality baseline                 | `COUNT(*)` of purchases where `article_id` = candidate item and `t_dat` falls in the same full calendar month as `cutoff`, one year prior; full-month window gives a stable denominator for `item_seasonality_strength` | Previous year (same calendar month) |
| `first_sold_date`                        | First observed sale date             | `MIN(t_dat)` over purchases where `article_id` = candidate item and `t_dat <= cutoff`                                                | All history (through cutoff) |
| `days_since_first_sold`                  | Catalog freshness / item maturity    | `cutoff - first_sold_date`; null when the item has no purchases before cutoff                                                      | All history (through cutoff) |
| `item_recent_to_lifetime_ratio`          | Item trend strength                  | `item_pop_30d / (item_category_pop_180d + 1)`                                                                                        | 30d vs 180d                |
| `item_category_recent_to_lifetime_ratio` | Category trend strength              | `item_category_pop_30d / (item_category_pop_180d + 1)`                                                                               | 30d vs 180d                |
| `item_seasonality_strength`              | Seasonality strength                 | `item_pop_7d / (item_pop_same_month_last_year + 1)` — 7-day numerator vs full prior-month denominator; month window absorbs weekly noise | 7d (numerator) vs same calendar month last year (denominator) |
| `item_avg_price`                         | Absolute price level of the item     | `AVG(price)` over all purchases where `article_id` = candidate item and `t_dat <= cutoff`; exposed to XGBoost directly to allow price × `age` interactions | All history (through cutoff) |
| `item_days_since_last_sold`              | Recent sales momentum                | `cutoff - MAX(t_dat)` over purchases where `article_id` = candidate item and `t_dat <= cutoff`; null when never sold — captures whether the SKU is still moving | All history (through cutoff) |
| `item_sales_channel_2_count`             | Channel 2 demand count               | `COUNT(*)` of purchases where `article_id` = candidate item, `sales_channel_id = 2`, and `t_dat <= cutoff` | All history (through cutoff) |
| `item_sales_channel_2_share`             | Normalised channel mix               | `item_sales_channel_2_count / (item_pop_180d + 1)` — share of stable 180-day demand coming from channel 2 | All history vs 180d denominator |


---

## User features

User-level behavior, preferences, recency, and demographics.


| Feature                         | Purpose                                                          | Formula                                                                                      | Look back                    |
| ------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------- |
| `user_category_pref_1y_rank1`   | User's most bought category                                      | Top-1 `garment_group_name` by `COUNT(*)` of user purchases, ordered descending               | 1 year                       |
| `user_category_pref_1y_rank2`   | User's second most bought category                               | Top-2 `garment_group_name` by purchase count                                                 | 1 year                       |
| `user_category_pref_1y_rank3`   | User's third most bought category                                | Top-3 `garment_group_name` by purchase count                                                 | 1 year                       |
| `user_color_pref_1y_rank1`      | Most preferred color                                             | Top-1 colour by `COUNT(*)` of user purchases, ordered descending                             | 1 year                       |
| `user_color_pref_1y_rank2`      | Second most preferred color                                      | Top-2 colour by purchase count                                                               | 1 year                       |
| `user_days_since_last_purchase` | Activity recency                                                 | `cutoff - MAX(t_dat)` over all user purchases where `t_dat <= cutoff`                        | All history (through cutoff) |
| `user_purchase_count_30d`       | Recent purchase frequency                                        | `COUNT(*)` of user purchases where `t_dat ∈ [cutoff - 30d, cutoff]`                          | 30 days                      |
| `user_purchase_count_180d`      | Stable purchase frequency                                        | `COUNT(*)` of user purchases where `t_dat ∈ [cutoff - 180d, cutoff]`                         | 6 months (180 days)          |
| `user_decayed_price_avg`        | Recent price preference (recent purchases weighted more)         | `Σ (price × w) / Σ w` over all user purchases where `t_dat <= cutoff`                        | Decayed (half-life 180d)     |
| `user_decayed_price_std`        | Spread of user's recent price preference; used for price z-score | `sqrt( Σ (w × (price - user_decayed_price_avg)²) / Σ w )`; use `max(std, 1e-6)` if near zero | Decayed (half-life 180d)     |
| `age`                           | Customer demographic for price × age interactions                | `age` column from `customers` table joined on `customer_id`; pass-through (not aggregated)   | Static                       |


---

## User–item features

Cross features between a user and the candidate item.

### Lifetime and decayed

| Feature                               | Purpose                                                                                    | Formula                                                                                | Look back                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------ |
| `user_item_repurchase`                | Number of times user repurchased the candidate item                                        | `COUNT(*)` of user purchases where `article_id` = candidate item and `t_dat <= cutoff` | All history (through cutoff)         |
| `user_item_decayed_repurchase`        | Recency-weighted repurchase; complements hard `user_item_repurchase`                       | `Σ w` over user purchases where `article_id` = candidate item and `t_dat <= cutoff`    | Decayed (half-life 180d)             |
| `user_item_decayed_interaction_ratio` | Personal vs global demand — user recently re-buys this item relative to overall popularity | `user_item_decayed_repurchase / (item_pop_180d + 1)`                                   | Decayed (half-life 180d) + item 180d |
| `user_item_price_decayed_zscore`      | How candidate item price fits user's recent budget (positive = above typical spend)        | `(candidate_price - user_decayed_price_avg) / user_decayed_price_std`                  | Decayed (half-life 180d)             |

### Windowed repurchase and pair recency

`user_item_repurchase` is a lifetime count — a user may have bought the item years ago but not recently. These hard-window variants and the pair recency feature capture short-term repeat intent. Do **not** add decayed variants on top of these window counts (see conventions).

| Feature                              | Purpose                                                                | Formula                                                                                                                                   | Look back              |
| ------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `user_item_repurchase_30d`           | Short-term repeat purchase of this SKU                                 | `COUNT(*)` of user purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 30d, cutoff]`                                     | 30 days                |
| `user_item_repurchase_90d`           | Medium-term repeat purchase                                            | `COUNT(*)` of user purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 90d, cutoff]`                                     | 90 days                |
| `user_item_repurchase_365d`          | Annual window; complements decay without duplicating it                | `COUNT(*)` of user purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 365d, cutoff]`                                    | 1 year                 |
| `user_item_days_since_last_purchase` | Pair recency — last time this user bought this article                 | `cutoff - MAX(t_dat)` over user purchases where `article_id` = candidate item and `t_dat <= cutoff`; null when never purchased             | All history            |
| `user_item_sales_channel_2_count`    | How many times user bought this item via channel 2                     | `COUNT(*)` of user purchases where `article_id` = candidate item, `sales_channel_id = 2`, and `t_dat <= cutoff`                           | All history            |


---

## User–category features

Cross features between a user and the **candidate item's `garment_group_name` category**.
These complement `user_category_pref_1y_rank1/2/3` (which tell *which* categories the user prefers) by quantifying *how much* they buy in the candidate item's category and how recently.

| Feature                                      | Purpose                                                                                            | Formula                                                                                                                                                       | Look back              |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `user_purchases_in_candidate_category_1y`    | Purchase volume in candidate's category over the last year                                         | `COUNT(*)` of user purchases where `garment_group_name` = candidate item's category and `t_dat ∈ [cutoff - 365d, cutoff]`                                     | 1 year                 |
| `user_days_since_last_purchase_in_category`  | Recency within candidate's category                                                                | `cutoff - MAX(t_dat)` over user purchases where `garment_group_name` = candidate item's category and `t_dat <= cutoff`; null when never purchased in category | All history            |
| `user_category_match_rank1`                  | Binary — candidate item's category is the user's top-1 preferred category                         | `1` if candidate `garment_group_name` = `user_category_pref_1y_rank1`, else `0`                                                                               | 1-year pref lookup     |
| `user_category_match_rank2`                  | Binary — candidate item's category is the user's top-2 preferred category                         | `1` if candidate `garment_group_name` = `user_category_pref_1y_rank2`, else `0`                                                                               | 1-year pref lookup     |
| `user_category_match_rank3`                  | Binary — candidate item's category is the user's top-3 preferred category                         | `1` if candidate `garment_group_name` = `user_category_pref_1y_rank3`, else `0`                                                                               | 1-year pref lookup     |

---

## Catalog pass-through attributes

Article metadata joined directly from the `articles` table — not aggregated from transactions. Passed to XGBoost as native categorical inputs.

| Feature             | Purpose                                                             | Source                      | Type        |
| ------------------- | ------------------------------------------------------------------- | --------------------------- | ----------- |
| `product_type_name` | Finer-grained product taxonomy below `garment_group_name`           | `articles.product_type_name`| categorical |

---

## Transaction features

Time-based encodings applied at the split's feature cutoff (offline) or request `current_date` (online). Never derived from post-cutoff label-window purchases.


| Feature         | Purpose                                                                                       | Formula                                                         | Look back    |
| --------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------ |
| `txn_month_sin` | Cyclical month encoding so December and January are close in feature space                    | `sin(month × 2π / 12)` where `month` is extracted from `cutoff` | Cutoff month |
| `txn_month_cos` | Paired with `txn_month_sin` to preserve seasonal continuity without treating month as ordinal | `cos(month × 2π / 12)` where `month` is extracted from `cutoff` | Cutoff month |


