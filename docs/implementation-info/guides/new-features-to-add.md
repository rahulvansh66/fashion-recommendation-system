# New Ranker Features (Proposed)

Additional XGBoost ranker features identified by comparing [`features-eng.md`](./features-eng.md) (current plan) with the Kaggle reference workflow in [`quick-and-easy-model-build-guide.md`](./quick-and-easy-model-build-guide.md).

**Status:** Implemented — added to `notebooks/utils/feature_engineering_core.py`, [`features-eng.md`](./features-eng.md), `configs/features/item_features.yaml`, and `configs/features/cross_features.yaml`.

**Contract:** Same temporal rules as [`features-eng.md`](./features-eng.md) (`cutoff`, FR-BATCH-02 splits or snap dates per [`new-updates.md`](./new-updates.md)). Ranker labels remain separate from features — see [`ranking-model-training-guide.md`](./ranking-model-training-guide.md).

**Schema:** [`schema-info.md`](../../system-design/schema-info.md) — `sales_channel_id` (1 or 2), `age`, `product_type_name`, `garment_group_name`.

---

## Source and scope

The Kaggle notebook groups predictors as `CUST_*` (customer), `ART_*` (article), and `CUSTART_*` (customer–article cross). CUSTART features drove the largest competition score gain (~0.007 → ~0.0247 MAP@12).

This document lists **only net-new signals** — features that add information beyond what [`features-eng.md`](./features-eng.md) already defines, even when the reference notebook uses a different column name.

**Out of scope here:**

- Candidate-generation lists (customer top-100, global top-100, etc.) — handled by two-tower retrieval in v1.
- Bayesian target encoding (`BAYES_*`) — v1 uses native XGBoost `cat_features`.
- Training filters (e.g. `ART_QUANTITY_SOLD_1M > 0`) — data-pipeline rules, not ranker inputs.

---

## Already covered (do not re-add)

| Reference (Kaggle) | Current plan (`features-eng.md`) |
| ------------------ | -------------------------------- |
| `ART_QUANTITY_SOLD_1M` | `item_pop_30d` |
| `ART_DAYS_SINCE_FIRST_PURCHASE` | `days_since_first_sold` / `first_sold_date` |
| `CUSTART_QUANTITY_SOLD_OVERALL` | `user_item_repurchase` |
| User days since last purchase | `user_days_since_last_purchase` |
| User purchase frequency | `user_purchase_count_30d`, `user_purchase_count_180d` |
| Article popularity windows | `item_pop_7d/30d/180d`, category pops, seasonality ratios |
| Lifetime pair repurchase | `user_item_repurchase`, `user_item_decayed_repurchase` |
| Price fit vs user budget | `user_item_price_decayed_zscore` (uses article mean price internally) |

---

## Conventions (same as `features-eng.md`)

| Topic | Rule |
| ----- | ---- |
| Category | Use `garment_group_name` for category-level aggregates. |
| Hard-window vs decayed | Windowed repurchase counts below are **hard windows** — they complement (not replace) lifetime `user_item_repurchase` and decayed `user_item_decayed_repurchase`. Do not add decayed variants of these window counts. |
| Price | `candidate_price` = mean transaction `price` per `article_id` over purchases with `t_dat <= cutoff` (same definition as `user_item_price_decayed_zscore`). |
| Channel | `sales_channel_id = 2` denotes the reference notebook's channel-2 sales; use the H&M schema value consistently in SQL/Spark. |
| Nulls | When a pair or item has no history in a window, use `0` for counts and null (or a large sentinel) for recency features — match imputation in the feature pipeline, not at train time. |

`cutoff` = feature cutoff for the split (or snap date) being built. Only rows with `t_dat <= cutoff` contribute to features.

---

## Item features

Price, recency, and channel signals at the article level.


| Feature | Purpose | Formula | Look back |
| ------- | ------- | ------- | --------- |
| `item_avg_price` | Absolute price level of the candidate item | Mean `price` over purchases where `article_id` = candidate item and `t_dat <= cutoff`; `0` or null when no sales | All history (through cutoff) |
| `item_days_since_last_sold` | Recent sales momentum — is the SKU still moving? | `cutoff - MAX(t_dat)` over purchases where `article_id` = candidate item and `t_dat <= cutoff`; null when never sold | All history (through cutoff) |
| `item_sales_channel_2_count` | Share of demand via channel 2 | `COUNT(*)` of purchases where `article_id` = candidate item, `sales_channel_id = 2`, and `t_dat <= cutoff` | All history (through cutoff) |
| `item_sales_channel_2_share` | Normalized channel mix | `item_sales_channel_2_count / (item_pop_180d + 1)` — optional derived feature if both parent counts exist | 180d denominator |

**Reference mapping:** `ART_AVERAGE_PRICE`, `ART_DAYS_SINCE_LAST_PURCHASE`, `ART_NUM_CHANNEL_2`.

**Note:** `item_avg_price` is already used inside `user_item_price_decayed_zscore` but is not passed to XGBoost as a standalone item feature today. Exposing it lets the model learn interactions (e.g. price × `age`).

---

## User features

Static demographics and category **volume** (not just top-3 ranks).


| Feature | Purpose | Formula | Look back |
| ------- | ------- | ------- | --------- |
| `age` | Customer demographic | `age` from `customers` table joined on `customer_id` | Static |

**Reference mapping:** `AGE` (SHAP interaction with `ART_AVERAGE_PRICE` in the notebook).

**Note:** `user_category_pref_1y_rank1/2/3` capture *which* categories a user prefers, not *how much* they buy in the candidate item's category. Category volume for the candidate lives in user–item cross features below.

---

## User–item features

Highest-priority additions — windowed pair history, pair recency, category affinity, and channel preference per pair.


| Feature | Purpose | Formula | Look back |
| ------- | ------- | ------- | --------- |
| `user_item_repurchase_30d` | Short-term repeat purchase of this SKU | `COUNT(*)` of user purchases where `article_id` = candidate item and `t_dat ∈ [cutoff - 30d, cutoff]` | 30 days |
| `user_item_repurchase_90d` | Medium-term repeat purchase | Same as above with `t_dat ∈ [cutoff - 90d, cutoff]` | 90 days |
| `user_item_repurchase_365d` | Longer hard window (complements decay) | Same as above with `t_dat ∈ [cutoff - 365d, cutoff]` | 1 year |
| `user_item_days_since_last_purchase` | Pair recency — last time user bought **this** article | `cutoff - MAX(t_dat)` over user purchases where `article_id` = candidate item and `t_dat <= cutoff`; null when never purchased | All history (through cutoff) |
| `user_item_sales_channel_2_count` | User bought this item via channel 2 | `COUNT(*)` of user purchases where `article_id` = candidate item, `sales_channel_id = 2`, and `t_dat <= cutoff` | All history (through cutoff) |
| `user_purchases_in_candidate_category_1y` | Volume in candidate's category (not just rank) | `COUNT(*)` of user purchases where `garment_group_name` = candidate item's `garment_group_name` and `t_dat ∈ [cutoff - 365d, cutoff]` | 1 year |
| `user_category_match_rank1` | Binary — candidate category is user's #1 | `1` if candidate `garment_group_name` = `user_category_pref_1y_rank1`, else `0` | 1 year (pref lookup) |
| `user_category_match_rank2` | Binary — candidate category is user's #2 | `1` if candidate `garment_group_name` = `user_category_pref_1y_rank2`, else `0` | 1 year (pref lookup) |
| `user_category_match_rank3` | Binary — candidate category is user's #3 | `1` if candidate `garment_group_name` = `user_category_pref_1y_rank3`, else `0` | 1 year (pref lookup) |
| `user_days_since_last_purchase_in_category` | Recency within candidate's category | `cutoff - MAX(t_dat)` over user purchases where `garment_group_name` = candidate item's `garment_group_name` and `t_dat <= cutoff`; null when never purchased in category | All history (through cutoff) |

**Reference mapping:**

| Proposed feature | Kaggle column |
| ---------------- | ------------- |
| `user_item_repurchase_30d` | `CUSTART_QUANTITY_SOLD_1M` |
| `user_item_repurchase_90d` | `CUSTART_QUANTITY_SOLD_3M` |
| `user_item_repurchase_365d` | `CUSTART_QUANTITY_SOLD_12M` |
| `user_item_days_since_last_purchase` | CUSTART "last purchase" (recency) |
| `user_item_sales_channel_2_count` | `CUSTART_NUM_CHANNEL_2` |
| `user_purchases_in_candidate_category_1y` | CUST "counts across product groups" |
| `user_category_match_rank*` | Implied by category affinity + [`ranking-model-training-guide.md`](./ranking-model-training-guide.md) cross examples |
| `user_days_since_last_purchase_in_category` | [`ranking-model-training-guide.md`](./ranking-model-training-guide.md) cross examples |

**Why windowed repurchase matters:** `user_item_repurchase` is lifetime count. A user may have bought the item years ago (`user_item_repurchase > 0`) but not recently — windowed counts and `user_item_days_since_last_purchase` capture short-term repeat intent that drove the Kaggle lift.

---

## Catalog attributes (pass-through)

Article metadata used as XGBoost categorical inputs — not aggregated from transactions.


| Feature | Purpose | Source | Type |
| ------- | ------- | ------ | ---- |
| `product_type_name` | Finer product taxonomy than `garment_group_name` | `articles.product_type_name` | categorical |

**Reference mapping:** `PRODUCT_TYPE_NAME` in candidate-generation prose; finer granularity than category-level popularity features.

**Note:** `garment_group_name` is already used in engineered popularity and preference features. `product_type_name` is an additional catalog signal for the ranker.

---

## Suggested implementation priority

| Priority | Features | Rationale |
| -------- | -------- | --------- |
| **P0** | `user_item_repurchase_30d`, `user_item_repurchase_90d`, `user_item_repurchase_365d`, `user_item_days_since_last_purchase` | Largest documented lift in reference solution; directly extends cross-feature group |
| **P1** | `user_purchases_in_candidate_category_1y`, `user_category_match_rank1/2/3`, `user_days_since_last_purchase_in_category` | Category affinity beyond top-3 rank labels; aligns with ranking guide cross examples |
| **P2** | `item_avg_price`, `item_days_since_last_sold` | Cheap item-level signals; enable price × demographic interactions |
| **P3** | `item_sales_channel_2_count`, `user_item_sales_channel_2_count` | Exploits `sales_channel_id`; validate signal strength on H&M data |
| **P4** | `age`, `product_type_name` | Static / catalog — low engineering cost |

---

## Next steps

1. ~~Add chosen features to [`features-eng.md`](./features-eng.md) once validated.~~ ✓ Done
2. ~~Extend `configs/features/` YAML and Glue/Spark feature jobs.~~ ✓ Done (`item_features.yaml`, `cross_features.yaml`)
3. Update [`ranking-model-training-guide.md`](./ranking-model-training-guide.md) §2 input feature table to reflect new features.
4. A/B in ranker training (val snap) starting with P0 CUSTART-style window features (`user_item_repurchase_30d/90d/365d`, `user_item_days_since_last_purchase`).

---

## Related docs

| Topic | Document |
| ----- | -------- |
| Current feature definitions | [`features-eng.md`](./features-eng.md) |
| Kaggle reference workflow | [`quick-and-easy-model-build-guide.md`](./quick-and-easy-model-build-guide.md) |
| Ranker training contract | [`ranking-model-training-guide.md`](./ranking-model-training-guide.md) |
| Planned snap-date splits | [`new-updates.md`](./new-updates.md) |
| H&M column definitions | [`schema-info.md`](../../system-design/schema-info.md) |
