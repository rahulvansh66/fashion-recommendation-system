# Ranking Model Training Guide

**Contract:** [`v1-requirements.md`](../../system-design/v1/v1-requirements.md) FR-BATCH-02 / FR-BATCH-04, [`v1-hld.md`](../../system-design/v1/v1-hld.md) §11.3 / §11.4  
**Reference implementation:** `tmp/notebooks/3_tp_training_ranking_model.ipynb` and `tmp/recsys/` (hands-on tutorial; v1 differs on splits, negatives, features, and eval)  
**Purpose:** Document how the Stage-2 **CatBoost ranker** is trained and evaluated for v1.

The ranker scores `(customer, article)` pairs after two-tower retrieval. It outputs `P(buy soon)` per candidate; the online pipeline sorts by score and returns the **top-10** list (after filter and diversity reorder).

**V1 offline objective:** predict whether a customer will purchase an article **soon**, where **soon** means a purchase in the split's **label window**. Online, the ranker scores candidates as of a **feature-cutoff date** with no fixed calendar horizon at request time.

---

## 1. Role in the system

```text
Two-tower retrieval (~100 candidates)
    → filter seen items
    → CatBoost ranker scores each (customer, candidate) pair
    → sort by P(buy soon)
    → diversity reorder → top-10
```

The ranker is **binary pair classification**, not listwise learning-to-rank. List quality is validated separately via `hit_rate@10` on the test window (§7.3).

---

## 2. Input features

Feature definitions: [`features-eng.md`](./features-eng.md). At training and inference, each pair vector includes **user**, **item**, and **cross** features (FR-PIPE-05).

Identifiers `customer_id` and `article_id` are used for joins and sampling but are **not** passed to CatBoost as model inputs.

| Group | Examples (v1) | Type |
|-------|---------------|------|
| User | `user_purchase_count_30d`, `user_days_since_last_purchase`, `user_category_pref_1y_rank1`, `user_decayed_price_avg`, `age` | numeric / categorical |
| Item | `item_pop_30d`, `days_since_first_sold`, `product_type_name`, `garment_group_name`, … | numeric / categorical |
| Cross | `user_item_repurchase`, `user_item_price_decayed_zscore`, category-match vs preferred category, days since last purchase in category | numeric |
| Context | `txn_month_sin`, `txn_month_cos` (from feature cutoff month at train/eval; from `current_date` at inference) | numeric |

**Target:** `label` — `1` if the customer purchased the article in the split's label window; `0` otherwise.

**Not used at ranker training time:** article text embeddings, `article_description`, retrieval embeddings.

---

## 3. Temporal splits and feature cutoffs

Authoritative split (FR-BATCH-02):

| Split | `t_dat` range | Role |
|-------|---------------|------|
| **Train** | start → **2020-03-31** | Ranker training |
| **Val** | **2020-04-01** → **2020-05-15** | Early stopping, hyperparameter tuning |
| **Test** | **2020-05-16** → **2020-06-30** | Final acceptance |
| **Drift 1–3** | **2020-07-01** → **2020-09-30** | Model Monitor only — not for model selection |

**Feature cutoff** (inclusive end of transaction history used to compute features):

| Split | Feature cutoff |
|-------|----------------|
| Train | `2020-03-31` |
| Val | `2020-03-31` |
| Test | `2020-05-15` |

No feature may use `t_dat` after the cutoff for that split.

---

## 4. Positive and negative sample strategy

The ranking task is **binary classification** on `(customer, article)` pairs.

### 4.1 Positive samples

- **Train:** every `(customer_id, article_id)` purchase with `t_dat <= 2020-03-31`.
- **Val:** every purchase in `2020-04-01` → `2020-05-15`.
- **Test:** every purchase in `2020-05-16` → `2020-06-30`.

Each positive row uses user/item/cross features computed as of that split's feature cutoff. Label: **`label = 1`**.

### 4.2 Negative samples — window-aware (5 per positive)

For **each positive** `(customer_id, article_id)`:

1. Keep the customer's features fixed.
2. Sample **5** `article_id` values such that:
   - the customer **did not** purchase that article in the split's **label window**, and
   - the article is **not** in the customer's `seen` set (purchases with `t_dat <=` feature cutoff).
3. Attach the negative article's item (+ cross) features. Label: **`label = 0`**.

**Ratio:** 1 positive : 5 negatives. Set **`scale_pos_weight = 5`** (or equivalent class weighting in CatBoost).

This matches inference: same user, many candidate articles, score each pair.

**Not used in v1:** Project 1's independent random cross-pairs (`customer`, `article`, and `age` sampled separately with replacement).

### 4.3 Dataset construction (SageMaker Processing)

Glue / SageMaker Processing (FR-BATCH-04 step 1) must:

1. Build user and item feature tables per feature cutoff.
2. Materialize positives from the label window for each split.
3. Generate 5 window-aware negatives per positive.
4. Write train / val / test Parquet to `features/ranking/` (or pipeline artifact path).

---

## 5. Preprocessing

| Step | Rule |
|------|------|
| **Split** | Temporal per §3 — not a random row fraction |
| **Categorical columns** | Passed to CatBoost as `cat_features` |
| **Numeric columns** | Used as-is; imputation handled in feature engineering |
| **Identifiers** | `customer_id`, `article_id` excluded from `X` |
| **Class weight** | `scale_pos_weight = 5` aligned with 1:5 sampling |

**Reference implementation (`tmp/recsys`):** 90/10 random `train_test_split` on a pre-built ranking table — replace with temporal splits for v1.

---

## 6. Model

| Aspect | Choice |
|--------|--------|
| **Algorithm** | `CatBoostClassifier` (gradient boosted trees) |
| **Task** | Binary classification — `P(buy soon)` |
| **Loss** | Logloss (default) |
| **Training compute** | SageMaker Training Job, `ml.m5.large` |
| **Serving** | SageMaker Endpoint `catboost-ranker`, `ml.t3.medium` |
| **Artifact** | `catboost_model.cbm` |

**Why CatBoost:** native handling of mixed numeric + high-cardinality categoricals; fast training on tabular pair features.

**Reference hyperparameters** (`tmp/recsys/config.py` — tune on val):

| Parameter | Reference value | v1 note |
|-----------|-----------------|---------|
| `learning_rate` | `0.2` | Tune on val |
| `iterations` | `100` | Early stopping on val |
| `depth` | `10` | |
| `scale_pos_weight` | `10` in reference | **Use `5`** for 1:5 sampling |
| `early_stopping_rounds` | `5` | On val `AUC-PR` or Logloss |

---

## 7. Evaluation strategy

### 7.1 Ranker metrics (pair-level, val / test)

Computed row-by-row on the ranking dataset for each split:

| Metric | How |
|--------|-----|
| **AUC-PR** | Primary ranker metric; threshold-free; **pipeline gate on test** |
| **Precision / Recall / F1** | Binary, default threshold 0.5; diagnostic |
| **Feature importance** | CatBoost `feature_importances_` |

Val is used for early stopping and hyperparameter tuning. Test is used for acceptance and pipeline promotion.

### 7.2 Retrieval metric (separate stage)

`recall@100` on val/test label-window purchases — see [`two-tower-retrieval-training-guide.md`](../two-tower-model/two-tower-retrieval-training-guide.md). Part of the same pipeline gate (FR-BATCH-04).

### 7.3 System metric (list-level, test only)

**`hit_rate@10`** — for each user with at least one test-window purchase:

1. As of test feature cutoff (`2020-05-15`), run the full online path: retrieve → filter seen → rank → diversity reorder.
2. **Hit** if **any** test-window purchase appears in the served **top-10**.

`hit_rate@10` = fraction of such users with at least one hit. This is the **pipeline promotion gate** alongside `recall@100` and ranker `AUC-PR`.

Pair-level precision/recall alone does not validate top-10 serving quality.

### 7.4 Drift monitoring

Drift slices (`2020-07-01` → `2020-09-30`) are for SageMaker Model Monitor input/score distribution checks only. Do not use them for hyperparameter tuning or promotion gates.

---

## 8. End-to-end flow (v1)

```text
Raw H&M CSVs → Glue clean → Glue features (per cutoff)
    → SageMaker Processing: ranking tables (positives + 5× window-aware negatives)
    → SageMaker Training: CatBoostClassifier (early stop on val)
    → SageMaker Processing: eval — AUC-PR, hit_rate@10, recall@100 on test
    → Model Registry → endpoint deploy
```

At inference, the ranker receives the same feature schema as training: user + item + cross features, with `txn_month_sin` / `txn_month_cos` from request `current_date`.

---

## 9. Reference vs v1 summary

| Topic | `tmp/recsys` (reference) | v1 |
|-------|--------------------------|-----|
| Train/val split | Random 90/10 | Temporal (§3) |
| Positives | All sampled transactions | Purchases in label window |
| Negatives | 10 random cross-pairs | **5 window-aware** per positive |
| `scale_pos_weight` | 10 | **5** |
| Features | age + article cats + month sin/cos | [`features-eng.md`](./features-eng.md) user/item/cross |
| Eval gate | Row-level P/R / F1 | **AUC-PR** + **`hit_rate@10`** + `recall@100` |
| Platform | Hopsworks | SageMaker Pipelines + S3 |

---

## 10. Source file map

| Topic | Reference | v1 target |
|-------|-----------|-----------|
| Ranking dataset (pos/neg) | `tmp/recsys/features/ranking.py` | SageMaker Processing job |
| Model factory & trainer | `tmp/recsys/training/ranking.py` | `src/.../training/ranking.py` |
| Hyperparameters | `tmp/recsys/config.py` | `configs/` YAML + SageMaker job defs |
| Training notebook | `tmp/notebooks/3_tp_training_ranking_model.ipynb` | SageMaker Pipeline |
| Inference scoring | `tmp/recsys/inference/ranking_predictor.py` | SageMaker `catboost-ranker` endpoint |
