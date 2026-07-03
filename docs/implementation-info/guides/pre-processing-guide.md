# Pre-Processing Guide

**Purpose:** Document data preparation steps that run **before** and **after** feature engineering. Feature definitions themselves live in [`features-eng.md`](./features-eng.md); this guide covers ingestion, cleaning, joins, temporal framing, label construction, and model-ready formatting.

**References:**
- [`features-eng.md`](./features-eng.md) — snap schedule, feature cutoff vs label window, feature catalog
- [`quick-and-easy-model-build-guide.md`](./quick-and-easy-model-build-guide.md) — Kaggle reference workflow (Snowflake datamart → pair rows → XGBoost)
- [`multimodels-h-m-personalized-fashion-recommendations.ipynb`](./multimodels-h-m-personalized-fashion-recommendations.ipynb) — basic H&M table loading, null handling, column selection
- [`ranking-model-training-guide.md`](./ranking-model-training-guide.md) — v1 ranker dataset rules (window-aware negatives, temporal splits)
- [`schema-info.md`](../../system-design/schema-info.md) — H&M star schema and column types

**Boundary:** Pre-processing makes raw data **valid, joined, and temporally correct**. Feature engineering turns clean history into predictors. Model-specific encoding (vocabularies, target encoding, XGBoost pools) happens **after** features are materialized.

---

## End-to-end flow

```text
Raw H&M CSVs (articles, customers, transactions)
    → ingest and type-cast
    → per-table cleaning and validation
    → write clean/ Parquet
    → join dimensions onto transactions
    → define snap dates and cutoffs
    → feature engineering (see features-eng.md)
    → post-FE imputation, filters, labels, negatives, splits
    → model-specific formatting (two-tower vocab / XGBoost dataset)
```

---

## Raw data ingestion

Load the three core H&M tables from the local `dataset/` path (dev) or S3 `raw/` prefix (v1 pipeline):

| Table | Source file | Role |
|-------|-------------|------|
| Articles | `articles.csv` | Product catalog — categories, colour, metadata |
| Customers | `customers.csv` | Demographics and engagement fields |
| Transactions | `transactions_train.csv` | Purchase fact table |

Preserve primary keys (`article_id`, `customer_id`) and foreign-key relationships. Do not alter hashed identifiers. Write ingested copies as Parquet under `clean/` once validated.

---

## Schema normalization and type casting

Apply consistent types before any joins or aggregations:

| Column / area | Rule |
|---------------|------|
| `t_dat` | Parse as date; normalize to calendar day (no time component) |
| `article_id`, `customer_id` | String identifiers; preserve leading zeros on articles |
| `price` | Float; reject non-numeric or negative values |
| `age` | Integer where present |
| `sales_channel_id` | Integer (`1` or `2`); treat missing as `0` downstream so channel counts default to zero rather than null |
| Column names | Use lowercase snake_case in pipeline outputs (e.g. `garment_group_name`, not `GARMENT_GROUP_NAME`) |

---

## Articles cleaning

Prepare the catalog for joins and categorical features (`garment_group_name`, `product_type_name`, `colour_group_name`).

| Action | Detail |
|--------|--------|
| Missing-value audit | Count nulls per column before dropping or imputing |
| Critical null handling | Drop rows where join keys (`article_id`) or critical category fields used in features are null. |
| Column retention | Keep identifiers and attributes needed for feature engineering and ranker pass-throughs: at minimum `article_id`, `product_code`, `product_type_name`, `garment_group_name`, `garment_group_no`, `colour_group_code`, `colour_group_name`, `index_group_name` |
| Category normalization | Trim whitespace; unify casing on string categoricals so the same garment group does not appear under multiple spellings |
| Text fields | `detail_desc` and `prod_name` are not used in v1 ranker or two-tower features; retain in clean data only if needed for future content features |
| Orphan filter | Flag or exclude articles with no transaction history before the earliest training snap to prevent noise (applied after transaction join). |

---

## Customers cleaning

Prepare demographics for user features and two-tower query inputs.

| Action | Detail |
|--------|--------|
| Missing-value audit | `age` is the primary numeric field; `FN`, `Active`, and `postal_code` are largely empty in H&M and are out of v1 scope |
| Critical null handling | Require `customer_id`. Apply an explicit `age` policy (e.g., impute missing ages to a sentinel). |
| Outliers & Validity | Cap implausible ages (e.g., below 16 or above 100) before any downstream z-score normalization. |
| Column retention | Keep `customer_id`, `age`, and optionally `club_member_status`, `fashion_news_frequency` for future use; v1 ranker uses `age` as a pass-through per [`features-eng.md`](./features-eng.md) |
| Duplicate check | One row per `customer_id`; deduplicate on key if duplicates exist |

---

## Transactions cleaning

The fact table drives all temporal logic and labels.

| Action | Detail |
|--------|--------|
| Missing-value audit | Confirm null counts on `t_dat`, `customer_id`, `article_id`, `price` |
| Column retention | Keep `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id` |
| Date sanity | Drop or quarantine rows with unparseable dates or dates outside the H&M competition window |
| Price sanity | Drop rows with null, zero, or negative `price`. Winsorize transaction `price` at a high percentile (e.g., 99th) to reduce sensitivity to extreme prices before computing decayed price features. |
| Referential integrity | Inner-join to cleaned articles and customers; drop orphan transactions whose keys are missing from dimension tables |
| Duplicate handling | Deduplicate exact repeats of `(t_dat, customer_id, article_id, price, sales_channel_id)` if present; keep legitimate same-day multi-purchases as separate rows with a sequence ID (`txn_seq`). |

---

## Data validation

Run after cleaning, before feature engineering:

| Check | Purpose |
|-------|---------|
| Schema contract | Required columns present with expected types ([`schema-info.md`](../../system-design/schema-info.md)) |
| Key uniqueness | No duplicate primary keys in articles or customers |
| Null thresholds | Fail or warn when critical columns exceed configured null rates |
| Date coverage | Transactions span expected min/max dates for snap schedule |
| Row counts | Baseline counts logged for drift detection between pipeline runs |
| Leakage guard (early) | No label-window dates mixed into rows used only for feature cutoff without explicit snap tagging |

---

## Dimension enrichment (pre–feature engineering joins)

Before computing aggregates, enrich transactions with attributes needed by [`features-eng.md`](./features-eng.md):

| Join | Adds |
|------|------|
| Articles → transactions | `garment_group_name` as item category, `colour_group_name` as item colour, `product_type_name`, `index_group_name` |
| Customers → transactions | `age` and other retained demographic columns |

Use left joins from transactions so row count is preserved; orphan rows should already have been removed in cleaning. Alias category and colour columns consistently (e.g. `item_category`, `item_color`) to match feature-engineering code paths.

---

## Temporal framework setup

Establish snap dates and windows **before** feature engineering so every aggregate respects the correct cutoff.

| Concept | Rule |
|---------|------|
| Snap date | Feature cutoff — only transactions with `t_dat <= snap_date` enter feature computation |
| Label window | Seven calendar days after the snap: `(snap_date + 1)` through `(snap_date + 7)` |
| One training example | One `(customer_id, article_id, snap_date)` pair with features frozen at the snap and label from the forward window |

Authoritative snap schedule (train, val, test, drift): see the table in [`features-eng.md`](./features-eng.md). All offline pipelines share this schedule (FR-BATCH-02).

**Important:** Feature cutoff and label window are disjoint by design — never use label-window purchases when computing features for the same row.

---

## Feature engineering

Compute user, item, user–item, user–category, catalog pass-through, and transaction encodings per snap date, using only history through each snap's cutoff.

Full feature definitions, formulas, and look-back windows: [`features-eng.md`](./features-eng.md).

Feature engineering is a separate documented stage; do not repeat feature formulas here.

---

## Post–feature engineering preprocessing

Steps that run on feature-enriched tables before model training or export.

### Null and default imputation

Aggregates are null when history does not exist (e.g. a customer never bought an article). Apply explicit defaults **after** features are computed:

| Pattern | v1 / reference rule |
|---------|---------------------|
| Count features with no history | Fill with `0` (no purchases in window) |
| Recency features with no history | Leave null or fill with a large sentinel only if the model requires non-null inputs; ranker docs prefer handling at FE time |
| Cross-feature nulls (no pair history) | Reference notebook fills `CUSTART_*`-style quantity and channel columns with `0` |
| Item average price null | Reference fills with `0` when the article has no sales history |
| Decayed price std near zero | Use `max(std, 1e-6)` when computing z-scores (see [`features-eng.md`](./features-eng.md)) |
| `sales_channel_id` missing on source row | Coalesce to `0` before channel count features |

### Active-item and sparsity filters

| Filter | Purpose |
|--------|---------|
| Dead SKUs | Apply active-item filters *after* feature engineering. Drop ranking pair rows where `item_pop_30d` (or `item_pop_7d`) equals zero. |
| Optional user activity | Exclude customers with zero purchases before cutoff if they cannot produce meaningful user features |

### Label construction

Build supervised `(customer_id, article_id, snap_date, label)` rows:

| Label | Definition |
|-------|------------|
| Positive (`1`) | Customer purchased the article with `t_dat` inside **that row's** label window |
| Negative (`0`) | Sampled pair with no purchase in the same label window |

### Negative sampling

| Approach | When |
|----------|------|
| **v1 window-aware (ranker)** | For each positive, sample **5** articles the customer did **not** buy in the label window and did **not** purchase before cutoff (`seen` exclusion). Draw negatives **per snap date**. Set `scale_pos_weight = 5`. |
| **Reference random global (Kaggle)** | ~4M random `(customer, article)` pairs not sold in the label window (~1:8 ratio). Simpler but less aligned with serving. |

### Train / validation / test separation

| Rule | v1 | Reference notebook |
|------|----|--------------------|
| Split axis | `snap_date` role (train / val / test / drift) | `SNAP_DATE` mask |
| Leakage | Val, test, and drift rows **never** in training | Test snap included in training mask — metrics are optimistic |

Always subset by snap role **after** imputation and filters so train statistics do not see eval rows.

### Identifier and metadata exclusion

Columns used for joins and sampling but **not** passed as model inputs: `customer_id`, `article_id`, `snap_date`, `label`. Catalog pass-through categoricals (`product_type_name`, `garment_group_name`) **are** model inputs for the v1 ranker.

### Ranker-specific formatting (XGBoost)

| Step | Rule |
|------|------|
| Categorical columns | Pass native `cat_features` to XGBoost — no target encoding in v1 |
| Numeric columns | Use as computed; imputation handled in post-FE step above |
| Class weight | `scale_pos_weight = 5` for 1:5 negative ratio |

### Retrieval-specific formatting (two-tower)

After anchor features from notebook 03 (`features/`, partitioned by `snap_date`):

| Step | Rule |
|------|------|
| Temporal split | Keep rows whose `snap_date` matches train/val/test snaps in `configs/models/two_tower.yaml`; retain **`label == 1`** purchase positives only |
| Query-side fields | `customer_id`, `age`, `txn_month_sin`, `txn_month_cos` |
| Candidate-side fields | `article_id`, `garment_group_name` (or `item_category`), `index_group_name` |
| Vocabulary | Build string-to-index maps on **training rows only**; index `0` = unknown |
| Age normalization | Z-score normalize `age` using train-set mean and std; persist in `preprocess_state.json` |

---

## Inference-time preprocessing

When scoring candidates online or in batch:

| Step | Detail |
|------|--------|
| Side tables | Load precomputed user and item feature tables for the request date or latest snap |
| Encoder state | Apply vocabularies and age normalizer fit on training data only |
| Chunked reads | Process large candidate files in chunks to bound memory |

Customer and article side tables should use **`fillna(0)`** before merge — mirroring the same defaults used in training imputation.
