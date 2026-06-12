# Feature Engineering Guide

Features for the data pipeline to capture **trend**, **seasonality**, **recency**, and related signals for retrieval and ranker models.

**Contract:** [`v1-requirements.md`](../../system-design/v1/v1-requirements.md) FR-BATCH-02, [`ranking-model-training-guide.md`](./ranking-model-training-guide.md), [`two-tower-retrieval-training-guide.md`](../two-tower-model/two-tower-retrieval-training-guide.md)

## Conventions

### Temporal splits and feature cutoffs

All offline pipelines share the same date partitions ([`v1-requirements.md`](../../system-design/v1/v1-requirements.md) FR-BATCH-02):

| Split | `t_dat` range | Role |
|-------|---------------|------|
| **Train** | start → **2020-03-31** | Model training |
| **Val** | **2020-04-01** → **2020-05-15** | Tuning / early stopping |
| **Test** | **2020-05-16** → **2020-06-30** | Final acceptance |
| **Drift 1–3** | **2020-07-01** → **2020-09-30** | Model Monitor only |

**Feature cutoff** — inclusive end of transaction history used to compute every feature in this guide. No row with `t_dat > cutoff` may contribute to features for that split.

| Split | Feature cutoff (`cutoff`) |
|-------|---------------------------|
| Train | `2020-03-31` |
| Val | `2020-03-31` |
| Test | `2020-05-15` |

**Ranker labels (separate from features):** positives are purchases in the split's **label window** (train dates for train; val/test dates for val/test). Negatives are **5 window-aware** non-purchases per positive (`seen` exclusion before `cutoff`). Detail: [`ranking-model-training-guide.md`](./ranking-model-training-guide.md).

**Online inference:** user and item features are served from precomputed `features/` artifacts; `txn_month_sin` / `txn_month_cos` use the request `current_date` (not a historical cutoff).

`cutoff` in the tables below always means the **feature cutoff for the split being built** — never a post-cutoff label date.


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

Popularity and demand signals at the item and category level.


| Feature                                  | Purpose                              | Formula                                                                                                                              | Look back                  |
| ---------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `item_pop_7d`                            | Current trend                        | `COUNT(*)` of purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 7d, cutoff]`                                      | 7 days                     |
| `item_pop_30d`                           | Recent popularity                    | `COUNT(*)` of purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 30d, cutoff]`                                     | 30 days                    |
| `item_category_pop_30d`                  | Recent popularity of item's category | `COUNT(*)` of purchases where `garment_group_name` = candidate item's category and `t_dat ∈ [cutoff - 30d, cutoff]`                  | 30 days                    |
| `item_pop_180d`                          | Stable demand                        | `COUNT(*)` of purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 180d, cutoff]`                                    | 6 months (180 days)        |
| `item_category_pop_180d`                 | Stable popularity of item's category | `COUNT(*)` of purchases where `garment_group_name` = candidate item's category and `t_dat ∈ [cutoff - 180d, cutoff]`                 | 6 months (180 days)        |
| `item_pop_same_month_last_year`          | Seasonality                          | `COUNT(*)` of purchases where `article_id` = candidate item and `t_dat` falls in the same calendar month as `cutoff`, one year prior | Previous year (same month) |
| `first_sold_date`                        | First observed sale date             | `MIN(t_dat)` over purchases where `article_id` = candidate item and `t_dat <= cutoff`                                                | All history (through cutoff) |
| `days_since_first_sold`                  | Catalog freshness / item maturity    | `cutoff - first_sold_date`; null when the item has no purchases before cutoff                                                      | All history (through cutoff) |
| `item_recent_to_lifetime_ratio`          | Item trend strength                  | `item_pop_30d / (item_category_pop_180d + 1)`                                                                                        | 30d vs 180d                |
| `item_category_recent_to_lifetime_ratio` | Category trend strength              | `item_category_pop_30d / (item_category_pop_180d + 1)`                                                                               | 30d vs 180d                |
| `item_seasonality_strength`              | Seasonality strength                 | `item_pop_7d / (item_pop_same_month_last_year + 1)`                                                                                  | 7d vs same month last year |


---

## User features

User-level behavior, preferences, and recency.


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


---

## User–item features

Cross features between a user and the candidate item.


| Feature                               | Purpose                                                                                    | Formula                                                                                | Look back                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------ |
| `user_item_repurchase`                | Number of times user repurchased the candidate item                                        | `COUNT(*)` of user purchases where `article_id` = candidate item and `t_dat <= cutoff` | All history (through cutoff)         |
| `user_item_decayed_repurchase`        | Recency-weighted repurchase; complements hard `user_item_repurchase`                       | `Σ w` over user purchases where `article_id` = candidate item and `t_dat <= cutoff`    | Decayed (half-life 180d)             |
| `user_item_decayed_interaction_ratio` | Personal vs global demand — user recently re-buys this item relative to overall popularity | `user_item_decayed_repurchase / (item_pop_180d + 1)`                                   | Decayed (half-life 180d) + item 180d |
| `user_item_price_decayed_zscore`      | How candidate item price fits user's recent budget (positive = above typical spend)        | `(candidate_price - user_decayed_price_avg) / user_decayed_price_std`                  | Decayed (half-life 180d)             |


---

## Transaction features

Time-based encodings applied at the split's feature cutoff (offline) or request `current_date` (online). Never derived from post-cutoff label-window purchases.


| Feature         | Purpose                                                                                       | Formula                                                         | Look back    |
| --------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------ |
| `txn_month_sin` | Cyclical month encoding so December and January are close in feature space                    | `sin(month × 2π / 12)` where `month` is extracted from `cutoff` | Cutoff month |
| `txn_month_cos` | Paired with `txn_month_sin` to preserve seasonal continuity without treating month as ordinal | `cos(month × 2π / 12)` where `month` is extracted from `cutoff` | Cutoff month |


