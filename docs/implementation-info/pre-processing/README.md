# Notebook Pre-Processing Reference

Concise record of data preparation implemented in notebooks `01`–`05`. For normative v1 rules see [`../guides/pre-processing-guide.md`](../guides/pre-processing-guide.md); feature formulas live in [`../guides/features-eng.md`](../guides/features-eng.md).

**Active dev dataset:** `dataset/{ACTIVE_DATASET}/` (default `sample_2000_users`), overridable via env var.

## Pipeline flow

```text
01 EDA + dtype cast
  → 02 clean, join, labels
  → 03 point-in-time features + post-FE filters
  → 04 supervised EDA + feature dtype cast
  → 05 two-tower temporal split + S3 staging
```

| Step | Notebook | Primary output |
|------|----------|----------------|
| 1 | `01_raw_data_eda_and_cleaning.ipynb` | `dataset/full_casted/*.parquet`, casted sample tables |
| 2 | `02_temporal_framing_and_labels.ipynb` | `dataset/{name}/transactions_with_label/` |
| 3 | `03_feature_engineering.ipynb` | `s3/dataset/{name}/features/` (Hive, `snap_date`) |
| 4 | `04_supervised_eda_and_feature_selection.ipynb` | Re-cast `features/` (in place) |
| 5 | `05_two_tower_retrieval_experiments.ipynb` | S3 `experiments/two_tower/{run_id}/{train,val,test}.parquet` |

---

## 01 — Raw Data EDA and Cleaning

**Role:** Unsupervised EDA on full H&M CSVs; validate the dev sample; cast dtypes. Does **not** apply row-level cleaning (that is notebook 02).

### Preprocessing actions

| Action | Detail | Why |
|--------|--------|-----|
| Load full CSVs + sample Parquet | `article_id` / `customer_id` as string; `t_dat` parsed on load | Preserve keys; enable temporal analysis |
| **Cast & persist** | `cast_table()` from `configs/data/ml_types.yaml` | Correct memory types; Parquet preserves schema for downstream notebooks |
| Full → `dataset/full_casted/{articles,customers,transactions}.parquet` | CSVs unchanged | Reference casted full data |
| Sample → overwrite `dataset/{ACTIVE_DATASET}/{articles,customers,transactions}/` | Transactions partitioned by `year`, `month` | Same Hive layout as pipeline targets |

### EDA-driven decisions (implemented in notebook 02+)

| Finding | Decision |
|---------|----------|
| Skewed ages | Cap 16–100; impute missing `age` (median in nb02) |
| Heavy price tail | Drop null/zero/negative prices; winsorize at 99th percentile |
| Long-tail item popularity | Popularity features expected; filter dead SKUs **after** FE (`item_pop_30d > 0`) |
| Seasonality in transactions | Strict `snap_date` cutoffs; 7-day forward label windows; seasonal encodings (`txn_month_sin/cos`) |
| Sample matches full on age, price, categories | Safe to prototype on `sample_2000_users` |

---

## 02 — Temporal Framing and Labels

**Role:** Core table cleaning, dimension joins, supervised pair construction for the ranker path.

### Preprocessing actions

| Action | Detail | Why |
|--------|--------|-----|
| **Schema normalization** | `t_dat` → datetime; drop null `customer_id` / `article_id` | Valid join keys and temporal logic |
| **Customers** | Drop null `customer_id`; fill missing `age` with median; clip 16–100 → int | Demographics usable by ranker and two-tower |
| **Articles** | Drop null `article_id`; strip + lowercase categoricals (`garment_group_name`, `product_type_name`, `colour_group_name`, `index_group_name`) | Consistent category features |
| **Price validity** | Drop `price <= 0`; winsorize at 99th percentile | Remove bad rows; stabilize price-based aggregates |
| **Deduplication** | Drop exact duplicates on `(t_dat, customer_id, article_id, price, sales_channel_id)` | Avoid double-counting without losing same-day multi-buys |
| **Dimension enrichment** | Inner join articles (category attrs) + customers (`age`) | Orphan filter + attributes for FE |
| **Temporal labels** | 11 snap dates (Mar–Sep 2020); label window = 7 days after snap | Point-in-time training rows per v1 schedule |
| **Positives** | Purchases in `(snap_date, snap_date + 7]` | Target = 1 |
| **Negatives** | 1:5 ratio per customer; exclude items seen before cutoff **or** bought in label window | Window-aware negatives; `scale_pos_weight = 5` for XGBoost |
| **Persist** | Hive Parquet → `transactions_with_label/`, partitioned by `snap_date` | Input to notebook 03 |

**Leakage guard:** Features use history `t_dat <= snap_date` only; label window is strictly forward of the cutoff.

---

## 03 — Feature Engineering

**Role:** Spark-based point-in-time features on labeled anchor pairs; post-FE imputation and sparsity filters.

### Preprocessing actions

| Action | Detail | Why |
|--------|--------|-----|
| Stage data | Copy `dataset/{name}/` → local `s3/` mirror (or upload to S3 on AWS) | Same paths locally and in cloud |
| **Anchors** | Load `transactions_with_label` (`customer_id`, `article_id`, `snap_date`, `label`) | Positives + negatives from nb02 |
| **Point-in-time FE** | `build_features()`: history strictly `<= snap_date` | Prevents temporal leakage |
| Feature families | Item popularity/recency, user activity, user–item cross, user–category cross, catalog pass-throughs | See `features-eng.md` |
| **Post-FE imputation** | `fillna(0)` on count columns + `item_avg_price` | Null counts mean “no history in window” |
| **Dead-SKU filter** | Drop rows where `item_pop_30d == 0` | Remove inactive items from ranker training |
| **Persist** | `dataset/{name}/features/`, partitioned by `snap_date` | Consumed by notebooks 04–05 and ranker training |

---

## 04 — Supervised EDA and Feature Selection

**Role:** Validate features against `label`; finalize ML dtypes for model code. **No row transforms** beyond casting.

### Preprocessing actions

| Action | Detail | Why |
|--------|--------|-----|
| Load `features/` Parquet | Includes `label` and `snap_date` | Supervised analysis on nb03 output |
| **Cast & persist** | `cast_table(df, "features", configs/features/ml_types.yaml)` → overwrite Hive layout | Spark writes int64/float64; cast to Int32/float32/category for XGBoost/two-tower |
| Supervised EDA | Distribution plots; Mann–Whitney U (numeric vs `label`); chi-square (categorical vs `label`) | Confirm signal; guide feature retention |
| Correlation matrix | Numeric features vs `label` and each other | Spot redundancy / collinearity |
| Feature selection | Baseline XGBoost + zero-importance drops | **Planned (TODO)** — commented out in notebook |

**Excluded from modeling (when training):** `customer_id`, `article_id`, `snap_date`, `label`.

---

## 05 — Two-Tower Retrieval Experiments

**Role:** Retrieval-path prep only — schema check, snap-based temporal split, upload splits for SageMaker. Assumes nb03 output includes transaction-level fields needed for retrieval.

### Preprocessing actions

| Action | Detail | Why |
|--------|--------|-----|
| Load features | From Hive-partitioned `s3/dataset/{name}/features/` | Same enriched anchor table as ranker path (nb03 output) |
| **`verify_schema()`** | Requires `customer_id`, `age`, `txn_month_sin`, `txn_month_cos`, `article_id`, `item_category`, `index_group_name`, `snap_date` | Query/candidate tower inputs per v1 spec |
| **`apply_temporal_split()`** | Split by `snap_date` into train/val/test snaps from `configs/models/two_tower.yaml`; keep `label == 1` purchase positives only | FR-BATCH-02; implicit retrieval pairs from forward-window purchases |
| **`stage_splits_s3()`** | Upload `{train,val,test}.parquet` under `experiments/two_tower/{run_id}/` | SageMaker HPO / training jobs |

### Not in this notebook (SageMaker training path)

| Step | Rule |
|------|------|
| Vocabulary | String→index maps on **train rows only**; index `0` = unknown |
| Age normalization | Z-score from train-set mean/std; persist in `preprocess_state.json` |
| Query fields | `customer_id`, `age`, `txn_month_sin`, `txn_month_cos` |
| Candidate fields | `article_id`, `item_category`, `index_group_name` |

---

## Quick reference: where each rule is applied

| Rule | Notebook |
|------|----------|
| Dtype casting (raw tables) | 01 |
| Age cap / impute | 02 |
| Price drop + winsorize | 02 |
| Category normalize | 02 |
| Window-aware negatives | 02 |
| Point-in-time features | 03 |
| Count `fillna(0)` + dead-SKU filter | 03 |
| Feature dtype cast | 04 |
| Two-tower temporal split | 05 |
