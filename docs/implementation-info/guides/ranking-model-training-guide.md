# Ranking Model Training Guide

Reference implementation: `tmp/notebooks/3_tp_training_ranking_model.ipynb` and `tmp/recsys/`.

This guide documents how the **second-stage ranking model** is trained in the H&M hands-on recommender pipeline. The ranker scores (customer, article) pairs produced by the two-tower retrieval stage. Training data is built upstream in `1_fp_computing_features.ipynb` via `compute_ranking_dataset()` and stored in the Hopsworks **ranking** feature group.

---

## 1. Input features for training

The ranking **feature view** (`create_ranking_feature_views`) exposes the following columns as model inputs. Identifiers `customer_id` and `article_id` are excluded from `X`; `label` is the target.

| Feature | Source | Type |
|---------|--------|------|
| `age` | customers | numeric |
| `product_type_name` | articles | categorical |
| `product_group_name` | articles | categorical |
| `graphical_appearance_name` | articles | categorical |
| `colour_group_name` | articles | categorical |
| `perceived_colour_value_name` | articles | categorical |
| `perceived_colour_master_name` | articles | categorical |
| `department_name` | articles | categorical |
| `index_name` | articles | categorical |
| `index_group_name` | articles | categorical |
| `section_name` | articles | categorical |
| `garment_group_name` | articles | categorical |
| `month_sin` | transactions (on-demand transform) | numeric |
| `month_cos` | transactions (on-demand transform) | numeric |

**Target:** `label` — `1` if the customer purchased the article, `0` otherwise.

**Not used at training time (engineered elsewhere but omitted from the ranking feature view):** article text embeddings, `article_description`, `club_member_status`, `age_group`, `postal_code`, `price`, `sales_channel_id`, and raw calendar fields (`year`, `month`, `day`, `day_of_week`).

---

## 2. Engineered features (upstream pipeline)

Features below are created in the feature-pipeline notebook (`1_fp_computing_features.ipynb`) before the ranking dataset is materialized. Only the subset in §1 is fed to the ranker.

### 2.1 Articles (`compute_features_articles`)

| Feature | Description |
|---------|-------------|
| `article_id` | Cast to string for consistent joins across tables. |
| `prod_name_length` | Character length of `prod_name`. |
| `article_description` | Structured text built from product name, type, group, appearance, color, category hierarchy, and optional `detail_desc`. |
| `embeddings` | Dense vector from `sentence-transformers` (`all-MiniLM-L6-v2`) encoding `article_description`. Used for retrieval/indexing, not ranking. |
| `image_url` | H&M CDN URL derived from `article_id` (folder prefix + image name). |

**Cleaning:** columns that are entirely null are dropped; `detail_desc` and `detail_desc_length` are removed after description is built.

### 2.2 Customers (`compute_features_customers`)

| Feature | Description |
|---------|-------------|
| `club_member_status` | Missing values filled with `"ABSENT"`. |
| `age_group` | Binned age: `0-18`, `19-25`, `26-35`, `36-45`, `46-55`, `56-65`, `66+`. |
| `age` | Cast to `Float64`; rows with null age are dropped. |

### 2.3 Transactions (`compute_features_transactions`)

| Feature | Description |
|---------|-------------|
| `article_id` | Cast to string. |
| `year`, `month`, `day`, `day_of_week` | Calendar parts extracted from `t_dat`. |
| `t_dat` | Converted to epoch milliseconds for the feature store event time. |
| `month_sin`, `month_cos` | Cyclical month encoding: `sin(2π × month / 12)` and `cos(2π × month / 12)` to capture seasonality without treating December and January as far apart. |

Hopsworks also registers `month_sin` / `month_cos` as **on-demand transformation functions** on the transactions feature group so they can be computed at serving time from the raw `month` column.

### 2.4 Ranking dataset (`compute_ranking_dataset`)

| Feature | Description |
|---------|-------------|
| `label` | Binary target: `1` for observed purchase pairs, `0` for synthetic negative pairs. |

Article metadata columns listed in §1 are joined onto each (customer, article) row from the articles feature group.

---

## 3. Preprocessing

### 3.1 Data volume (before ranking dataset)

Full H&M transactions (~31M rows) are subsampled by customer:

| `CUSTOMER_DATA_SIZE` | Customers sampled |
|----------------------|-------------------|
| `SMALL` (default) | 1,000 |
| `MEDIUM` | 5,000 |
| `LARGE` | 50,000 |

`DatasetSampler` draws customers with `random.seed(27)` and keeps only their transactions.

### 3.2 Ranking dataset construction

- `article_id` cast to string before all joins.
- Positive rows: every `(customer_id, article_id)` pair from transactions, joined with customer `age` and article attributes.
- Negative rows: synthetic pairs (see §4).
- Final join with deduplicated article features on `article_id`.

### 3.3 Model-time preprocessing (`RankingModelTrainer`)

- **Train/validation split:** 90% / 10% via Hopsworks `feature_view_ranking.train_test_split(test_size=0.1)`.
- **Categorical detection:** columns with `string` or `object` dtype are passed to CatBoost as `cat_features` in a `Pool`.
- **No additional scaling or imputation** at fit time — cleaning and typing are handled in the feature pipeline.
- **Identifiers dropped:** `customer_id` and `article_id` are not in `X_train` / `X_val`.

---

## 4. Positive and negative sample strategy

The ranking task is framed as **binary classification** on (customer, article) pairs.

### Positive samples

- Every transaction in the (sampled) transactions table is a positive example.
- Each row carries the customer's `age` and the purchased article's metadata.
- Label: **`label = 1`**.

### Negative samples

- Count: **`10 ×`** number of positive pairs (`n_neg = len(positive_pairs) * 10`).
- Built by **independent random sampling with replacement** (not hard-negative mining):
  - `article_id` — sampled from unique articles in transactions (`seed=2`)
  - `customer_id` — sampled from transaction customers (`seed=3`)
  - `age` — sampled from transaction ages (`seed=4`)
- Label: **`label = 0`**.

Positive and negative frames are concatenated. With the default small sample this yields roughly **20,376 positives** and **203,760 negatives** (~10:1 ratio), matching `RANKING_SCALE_POS_WEIGHT = 10`.

**Note:** negatives are random cross-pairs, not explicitly verified as non-purchases. At catalog scale most random pairs are unlikely to be true purchases.

---

## 5. Model

**Algorithm:** [CatBoost](https://catboost.ai/) **`CatBoostClassifier`** — gradient boosted decision trees for binary classification.

**Role in the system:** After the two-tower model retrieves ~100 candidate articles per user (FAISS / Hopsworks embedding index), the ranker assigns a purchase-likelihood score to each (customer, candidate) pair. Candidates are sorted by score descending to produce the final top-N list.

**Why CatBoost here:**

- Handles mixed numeric + high-cardinality categorical features natively.
- No one-hot encoding required for article/customer attribute columns.
- Fast to train on tabular pair features compared to a second neural ranker.

Implementation: `recsys/training/ranking.py` — `RankingModelFactory.build()` and `RankingModelTrainer`.

---

## 6. Finalized training parameters

From `recsys/config.py` (`Settings`) and `RankingModelFactory`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `RANKING_LEARNING_RATE` | `0.2` | Step size for gradient boosting |
| `RANKING_ITERATIONS` | `100` | Maximum boosting rounds |
| `depth` | `10` | Tree depth (hardcoded in factory) |
| `RANKING_SCALE_POS_WEIGHT` | `10` | Class weight for positives; aligns with 10:1 neg:pos sampling |
| `RANKING_EARLY_STOPPING_ROUNDS` | `5` | Stop if validation metric does not improve |
| `use_best_model` | `True` | Persist the best iteration from early stopping |
| `RANKING_DATASET_VALIDATON_SPLIT_SIZE` | `0.1` | 10% holdout for validation |
| `loss_function` | default (Logloss) | Binary classification |
| `eval_set` | validation `Pool` | Used during `fit()` for early stopping |

**Related pipeline settings (not CatBoost hyperparameters):**

| Parameter | Value |
|-----------|-------|
| `CUSTOMER_DATA_SIZE` | `SMALL` (1,000 customers) |
| `FEATURES_EMBEDDING_MODEL_ID` | `all-MiniLM-L6-v2` (articles only; not ranking input) |

---

## 7. Evaluation strategy

### Validation split

- Same 10% split from the ranking feature view used as `eval_set` during training and for post-hoc metrics.

### Metrics (`RankingModelTrainer.evaluate`)

| Metric | How computed |
|--------|----------------|
| **Classification report** | `sklearn.metrics.classification_report` on validation predictions — per-class precision, recall, F1, support |
| **Binary precision / recall / F1** | `precision_recall_fscore_support(..., average="binary")` returned as a dict |
| **Feature importance** | CatBoost `feature_importances_` mapped to column names, sorted descending |

### Example validation results (notebook run)

On ~40,720 validation rows (90% of ~224k total ranking rows):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 0 (negative) | 1.00 | 1.00 | 1.00 | 38,778 |
| 1 (positive) | 0.96 | 1.00 | 0.98 | 1,942 |

### Feature importance (same run)

Dominant signal from seasonal encoding:

1. `month_cos` (~59)
2. `month_sin` (~34)
3. `product_type_name`, `age`, and remaining categoricals (low single digits)

Article embeddings are noted in the notebook as a potential future improvement but are **not** included in v1 ranking features.

### Model registry

After evaluation, the model is serialized with `joblib`, registered in Hopsworks Model Registry with validation metrics and an `input_example`, and linked to the ranking feature view for batch/online scoring.

---

## 8. End-to-end flow (reference)

```text
Raw H&M CSVs
    → feature engineering (articles, customers, transactions)
    → customer subsample (DatasetSampler)
    → compute_ranking_dataset (positives + 10× negatives)
    → Hopsworks ranking feature group + feature view
    → train_test_split (90/10)
    → CatBoostClassifier.fit (early stopping on val)
    → evaluate (classification report + feature importance)
    → register in Hopsworks Model Registry
```

At inference (`ranking_transformer.py`), the ranker receives the same feature schema: customer `age`, article categoricals, and request-time `month_sin` / `month_cos` for the current month.

---

## 9. Source file map

| Topic | File |
|-------|------|
| Ranking dataset (pos/neg sampling) | `tmp/recsys/features/ranking.py` |
| Articles FE | `tmp/recsys/features/articles.py` |
| Customers FE | `tmp/recsys/features/customers.py` |
| Transactions FE + month sin/cos | `tmp/recsys/features/transactions.py` |
| Feature view definition | `tmp/recsys/hopsworks_integration/feature_store.py` |
| Model factory & trainer | `tmp/recsys/training/ranking.py` |
| Hyperparameters | `tmp/recsys/config.py` |
| Training notebook | `tmp/notebooks/3_tp_training_ranking_model.ipynb` |
| Feature pipeline notebook | `tmp/notebooks/1_fp_computing_features.ipynb` |
