# Item Embeddings and Inference Pipeline Guide

**Source:** `tmp/notebooks/4_ip_computing_item_embeddings.ipynb`, `tmp/recsys/inference/`, `tmp/recsys/hopsworks_integration/`  
**Purpose:** Document how item embeddings are precomputed offline and how the end-to-end inference pipeline (user embedding → ANN retrieval → ranking) works at serving time.

This guide is a companion to:
- [`two-tower-retrieval-training-guide.md`](two-tower-retrieval-training-guide.md) — training the two towers
- [`ranking-model-training-guide.md`](ranking-model-training-guide.md) — training the CatBoost ranker

---

## 1. Why Precompute Item Embeddings

At inference the system must retrieve the top-~100 candidate articles for a given user in low latency. The ItemTower (candidate tower) maps `(article_id, garment_group_name, index_group_name)` → 16-dim vector. Running this forward pass for all ~11k catalog articles on every request would be prohibitively slow.

The solution is to **run the ItemTower once offline**, store every article's 16-dim vector in a vector index, and then serve retrieval as a cheap ANN (approximate nearest-neighbor) lookup against precomputed vectors.

---

## 2. Item Embedding Computation (Notebook 4)

**Notebook:** `tmp/notebooks/4_ip_computing_item_embeddings.ipynb`  
**Library:** `tmp/recsys/features/embeddings.py`

### 2.1 Step-by-step flow

```
Hopsworks Model Registry
  → download trained candidate_model (ItemTower SavedModel)
  → load retrieval feature view train split
  → deduplicate by article_id (one row per item)
  → batch through ItemTower (batch_size=2048)
  → store {article_id, embeddings[16-float]} → Hopsworks candidate_embeddings FG
  → create EmbeddingIndex (dim=16) on candidate_embeddings
```

### 2.2 Input: which articles are embedded

Only articles seen during **retrieval training** are embedded. The notebook loads the same feature-view split used for training:

```python
train_df, val_df, test_df, _, _, _ = feature_view.train_validation_test_split(
    validation_size=0.1, test_size=0.1, ...
)
```

Only `train_df` is used. It is deduplicated on `article_id` to produce one row per unique catalog item.

### 2.3 Features fed into the ItemTower

| Feature | Processing in tower |
|---|---|
| `article_id` | `StringLookup` → `Embedding(num_items+1, 16)` — learned collaborative identity signal |
| `garment_group_name` | `StringLookup` → `tf.one_hot` — fine-grained garment category |
| `index_group_name` | `StringLookup` → `tf.one_hot` — top-level catalog division (Ladieswear, Menswear, etc.) |

These are the only three features. All other article attributes (text description, SentenceTransformer embeddings, price, colour) are **not** used in the retrieval embedding.

> **Important distinction:** `articles.py` also builds 384-dim SentenceTransformer embeddings (`all-MiniLM-L6-v2`) from `article_description`. Those live in a separate embedding index and are used only for the LLM/text-similarity search path — **not** for the two-tower retrieval path.

### 2.4 Output: candidate_embeddings feature group

| Column | Type | Description |
|---|---|---|
| `article_id` | string | Primary key |
| `embeddings` | list[float, 16] | 16-dim ItemTower output vector |

An `EmbeddingIndex(dim=16)` is created on the `embeddings` column so that `find_neighbors()` can be called at serving time.

---

## 3. Inference Pipeline — End to End

At serving time the pipeline processes one request: a `(customer_id, transaction_date)` and returns a ranked list of article IDs.

### 3.1 Architecture overview

```
Request: { customer_id, transaction_date }
              |
              v
    ┌─────────────────────┐
    │   query_transformer  │  (Step 1: embed the user)
    └─────────────────────┘
       - Fetch age from customers FV
       - Derive month from transaction_date
       - Compute month_sin / month_cos (on-demand UDF)
       - Run QueryTower → query_emb [16-dim]
              |
              v
    ┌──────────────────────┐
    │  ranking_transformer  │  (Step 2: retrieve + featurize candidates)
    └──────────────────────┘
       - candidate_embeddings.find_neighbors(query_emb, k=100)
       - Filter items already purchased by this customer
       - Join article attributes + customer age for each candidate
       - Assemble feature matrix [~100 rows × 14 features]
              |
              v
    ┌──────────────────────┐
    │  ranking_predictor    │  (Step 3: score and sort)
    └──────────────────────┘
       - CatBoost.predict_proba → purchase probability per candidate
       - Sort descending by score
       - Return top-N article IDs
```

### 3.2 Step 1 — Query embedding (user side)

**File:** `tmp/recsys/inference/query_transformer.py`, `tmp/recsys/hopsworks_integration/two_tower_serving.py`

| Action | Detail |
|---|---|
| Fetch `age` | Point lookup from `customers` feature view using `customer_id` |
| Compute `month` | Extract from `transaction_date` |
| Compute `month_sin`, `month_cos` | On-demand transform registered on the transactions FG |
| Run QueryTower | `compute_embedding({customer_id, age, month_sin, month_cos})` |
| Output | `{customer_id, month_sin, month_cos, query_emb[16]}` |

The QueryTower SavedModel exposes a `compute_embedding` `@tf.function`:

```python
@tf.function()
def compute_embedding(self, instances):
    query_embedding = self.model(instances)
    return {
        "customer_id": instances["customer_id"],
        "month_sin": instances["month_sin"],
        "month_cos": instances["month_cos"],
        "query_emb": query_embedding,
    }
```

The output dict is forwarded directly to the ranking transformer as its input payload.

### 3.3 Step 2 — ANN retrieval + feature assembly (ranking_transformer)

**File:** `tmp/recsys/inference/ranking_transformer.py`

#### ANN search

```python
neighbors = self.candidate_index.find_neighbors(
    inputs["query_emb"],
    k=100,
)
```

`find_neighbors` queries the `candidate_embeddings` EmbeddingIndex (Hopsworks internal ANN), returning up to 100 `article_id` values whose 16-dim vectors are closest to the user's 16-dim query embedding (by dot product / cosine similarity).

#### Post-retrieval filter

Items the customer has already purchased are fetched from the `transactions` feature group and removed from the candidate list. This prevents recommending articles the user demonstrably already owns.

#### Feature assembly

For each surviving candidate, the transformer looks up:
- Article attributes from the `articles` feature group (11 categorical columns)
- Customer `age` (from the earlier lookup or passed through the payload)
- `month_sin`, `month_cos` (carried from query_transformer output)

This builds the **same feature schema used during ranking training** — ~100 rows × 14 columns — as a CatBoost `Pool`.

### 3.4 Step 3 — CatBoost scoring (ranking_predictor)

**File:** `tmp/recsys/inference/ranking_predictor.py`

- Calls `model.predict_proba(pool)[:, 1]` on the assembled feature matrix.
- Associates each score with its `article_id`.
- Returns candidates sorted by score descending.
- Top-N (typically top-12 for the Streamlit UI) are surfaced to the user.

---

## 4. Features Used at Each Inference Stage

### 4.1 Query embedding (retrieval stage, user side)

| Feature | Source at inference |
|---|---|
| `customer_id` | Request input |
| `age` | Point lookup: `customers` feature view |
| `month_sin` | Derived from `transaction_date` via on-demand UDF |
| `month_cos` | Derived from `transaction_date` via on-demand UDF |

### 4.2 ANN retrieval (vector search)

| What | Detail |
|---|---|
| Index | 16-dim ItemTower vectors, precomputed offline for all training-set articles |
| Query | 16-dim QueryTower output for the current user + context |
| `k` | 100 candidates returned |
| Distance | Dot product / cosine similarity in the shared 16-dim embedding space |

### 4.3 Ranking features (re-scoring stage)

Same as ranking training — see [`ranking-model-training-guide.md §1`](ranking-model-training-guide.md):

| Category | Features |
|---|---|
| Customer | `age` |
| Temporal | `month_sin`, `month_cos` |
| Article (categorical) | `product_type_name`, `product_group_name`, `graphical_appearance_name`, `colour_group_name`, `perceived_colour_value_name`, `perceived_colour_master_name`, `department_name`, `index_name`, `index_group_name`, `section_name`, `garment_group_name` |

---

## 5. Evaluation Strategy

### 5.1 Retrieval evaluation

Training and offline evaluation of the retrieval model is done via `tfrs.metrics.FactorizedTopK` during `model.fit()`:

| Metric | Meaning |
|---|---|
| `top_1_categorical_accuracy` | True purchased article is ranked #1 among all training-set items |
| `top_5_categorical_accuracy` | True article in top 5 |
| `top_10_categorical_accuracy` | True article in top 10 |
| `top_50_categorical_accuracy` | True article in top 50 |
| `top_100_categorical_accuracy` | True article in top 100 — **headline metric** |

**Why top-100 is the headline metric:** The ranker only sees the candidates that retrieval returns. If the truly purchased article is not in the top-100 retrieved, the ranker can never surface it. Top-100 recall directly measures whether Stage 1 preserves the correct answer for Stage 2.

**Evaluation procedure:**
1. Deduplicate training articles → build candidate corpus (one embedding per `article_id`).
2. For each validation batch, run both towers to get embeddings.
3. `FactorizedTopK` scores each query embedding against the full corpus (brute-force dot product).
4. Report top-K hit rates and validation loss each epoch.

**Data split used:**
| Split | Fraction | Use |
|---|---|---|
| Train | 80% | Model training + vocabulary construction |
| Validation | 10% | Top-K metrics + loss each epoch |
| Test | 10% | Reserved by feature view; **not evaluated in this reference** |

**Gaps in this evaluation:**
- The 10% test split is never evaluated — only train/val reported.
- No per-user precision@k, NDCG, MAP, or Hit Rate evaluation loop.
- Cold-start users/articles (not in training vocabulary) fall back to the unknown-slot embedding; their performance is not separately tracked.

### 5.2 Ranking evaluation

The CatBoost ranker is evaluated as binary classification on the 10% held-out validation split:

| Metric | Library |
|---|---|
| Per-class precision, recall, F1 | `sklearn.metrics.classification_report` |
| Binary precision, recall, F1 | `precision_recall_fscore_support(..., average="binary")` |
| Feature importance | CatBoost `feature_importances_` sorted descending |

**Example results from the reference notebook run (~224k ranking rows, 90/10 split):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (no purchase) | 1.00 | 1.00 | 1.00 | 38,778 |
| 1 (purchase) | 0.96 | 1.00 | 0.98 | 1,942 |

**Top feature importances (same run):**
1. `month_cos` (~59) — dominant seasonal signal
2. `month_sin` (~34)
3. `product_type_name`, `age`, and other article categoricals (low single digits)

**Data split used:**
| Split | Fraction | Use |
|---|---|---|
| Train | 90% | CatBoost training with early stopping on val |
| Validation | 10% | Early stopping + post-hoc classification metrics |
| Test | — | No separate held-out test set |

**Gaps in this evaluation:**
- No list-wise ranking metrics (NDCG, MAP, MRR, precision@k).
- Evaluation is over the same random split used for early stopping — not a fully independent test set.
- Negatives are random cross-pairs, not hard negatives drawn from retrieval output. Reported metrics (~0.96–1.00 F1) are inflated by easy negatives.
- No end-to-end pipeline evaluation combining retrieval recall and ranking precision.

---

## 6. Key Gaps vs Production Evaluation

| Gap | Why it matters |
|---|---|
| No retrieval test-split evaluation | Reported top-K accuracy is validation-only; generalization to unseen users/items unknown |
| No ranking@k metrics (NDCG, MAP) | Binary F1 on random negatives does not measure ordering quality within the retrieved candidate set |
| No hard negatives for ranking | Ranking negatives should ideally be drawn from retrieval output (in-list negatives) to measure what the ranker must actually distinguish at serving time |
| No end-to-end offline eval | Retrieval recall × ranking precision@k measured together — the metric that directly proxies live recommendation quality |
| No cold-start evaluation | Users/articles absent from training vocabulary are silently handled by the unknown-slot embedding with no reported performance |

---

## 7. Optional: LLM Ranker Path

A GPT-4o-mini based ranker (`llm_ranking_predictor.py`) is available as an alternative to CatBoost. It uses the same feature set but prompts an LLM for a purchase probability. It is limited to **20 candidates per request** due to latency. This path is deployed separately (notebook 7) and used in the Streamlit demo via a flag.

---

## 8. Source File Map

| Concern | File |
|---|---|
| Item embedding batch job | `tmp/notebooks/4_ip_computing_item_embeddings.ipynb` |
| Embedding computation library | `tmp/recsys/features/embeddings.py` |
| Query embedding (user tower at serving) | `tmp/recsys/inference/query_transformer.py` |
| QueryTower SavedModel wrapper | `tmp/recsys/hopsworks_integration/two_tower_serving.py` |
| ANN retrieval + feature assembly | `tmp/recsys/inference/ranking_transformer.py` |
| CatBoost scoring | `tmp/recsys/inference/ranking_predictor.py` |
| LLM ranker (optional) | `tmp/recsys/inference/llm_ranking_predictor.py` |
| Deployment notebook | `tmp/notebooks/5_ip_creating_deployments.ipynb` |
| Streamlit UI | `tmp/recsys/ui/recommenders.py` |
| Retrieval training guide | `two-tower-retrieval-training-guide.md` |
| Ranking training guide | `ranking-model-training-guide.md` |
