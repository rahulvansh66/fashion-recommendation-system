# Industry-Grade Strategy Guide for H&M Next-7-Day Fashion Recommendation

**Dataset:** H&M Personalized Fashion Recommendations  
**Goal:** Predict which articles each customer is likely to purchase in the next 7 days.  
**Architecture decision:** Two-stage recommender system:

1. **Candidate generation:** Two-tower neural retrieval model with ANN retrieval.
2. **Ranking:** LightGBM ranker trained on generated candidates.

This guide is written for an offline-to-production recommender system, not only for a Kaggle notebook. It assumes the available data includes:

- `transactions_train.csv`: customer purchases with `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id`.
- `articles.csv`: article metadata such as product type, group, color, department, section, garment group, and description.
- `customers.csv`: customer metadata such as age, club member status, fashion news frequency, and postal code.

The dataset scale is large enough to require production-style design: roughly 105K articles, 1.37M customers, and 31.8M transaction rows.

---

## 1. System Overview

### 1.1 Why a two-stage system?

A real fashion catalog can contain tens or hundreds of thousands of active articles. Ranking every article for every customer is expensive and unnecessary.

A two-stage system solves this by splitting the problem:

| Stage | Purpose | Output size | Model type |
|---|---|---:|---|
| Candidate generation | Quickly retrieve plausible articles | 100-1000 candidates per customer | Two-tower neural model + ANN index |
| Ranking | Sort candidates by purchase likelihood | Top 12 articles per customer | LightGBM ranker |

The two-tower model optimizes **coverage and recall**. The LightGBM ranker optimizes **ordering quality**.

### 1.2 Serving-time flow

```text
Customer ID
   |
   v
Build / fetch customer features as of prediction cutoff date
   |
   v
Two-tower user tower creates customer embedding
   |
   v
ANN search retrieves top K article embeddings
   |
   v
Generate candidate rows: one row per customer-article pair
   |
   v
Compute LightGBM features using only data before cutoff
   |
   v
LightGBM scores candidates
   |
   v
Business rules and fallback logic
   |
   v
Top 12 article recommendations
```

### 1.3 Core principle

Every training and evaluation step must answer this question:

> At this cutoff date, what would the system have known, and what would it have recommended for the next 7 days?

If any feature, label, split, or candidate uses information after the cutoff date, the evaluation is contaminated by leakage.

---

## 2. Temporal Split Strategy

### 2.1 Objective

The model should predict purchases in the next 7 days. Therefore, validation should simulate the real prediction task.

Do not use random row-level train/test splits for recommender evaluation. Random splits mix future and past behavior, overestimate quality, and can leak customer and article popularity signals.

### 2.2 Recommended rolling validation design

Use rolling weekly validation folds.

Example assuming the last available transaction date is `2020-09-22`:

| Fold | Feature / training cutoff | Label window | Purpose |
|---|---|---|---|
| Fold 1 | up to 2020-08-25 | 2020-08-26 to 2020-09-01 | Early validation |
| Fold 2 | up to 2020-09-01 | 2020-09-02 to 2020-09-08 | Stability check |
| Fold 3 | up to 2020-09-08 | 2020-09-09 to 2020-09-15 | Model selection |
| Fold 4 | up to 2020-09-15 | 2020-09-16 to 2020-09-22 | Final local validation |

Each fold must be built independently:

```text
For each fold:
  1. Define cutoff_date.
  2. Use only transactions where t_dat <= cutoff_date for features and candidate generation.
  3. Use transactions where cutoff_date < t_dat <= cutoff_date + 7 days as labels.
  4. Train / evaluate candidate generator and ranker on this fold.
```

### 2.3 Example implementation sketch

```python
from datetime import timedelta

folds = [
    "2020-08-25",
    "2020-09-01",
    "2020-09-08",
    "2020-09-15",
]

for cutoff in folds:
    cutoff = pd.Timestamp(cutoff)
    label_start = cutoff + timedelta(days=1)
    label_end = cutoff + timedelta(days=7)

    history = transactions[transactions["t_dat"] <= cutoff]
    labels = transactions[
        (transactions["t_dat"] >= label_start) &
        (transactions["t_dat"] <= label_end)
    ]

    # Build features, candidates, and labels from this fold only.
```

### 2.4 Train/validation/test usage

Use folds differently depending on the task:

| Task | Recommended folds |
|---|---|
| Feature debugging | One validation fold |
| Hyperparameter tuning | 2-3 rolling folds |
| Final local score | Last available fold |
| Final training | Train on all available history before the real test cutoff |

### 2.5 Leakage checklist

Before trusting a validation result, confirm:

- No transaction after `cutoff_date` is used in features.
- Article popularity is computed only before `cutoff_date`.
- Customer purchase counts are computed only before `cutoff_date`.
- Two-tower item embeddings are trained only from historical data for the fold.
- Candidate generation does not use label-window purchases.
- Negative samples are selected from candidates generated at the cutoff, not from future purchases.
- LightGBM labels are created only from the next-7-day label window.

---

## 3. Candidate Generation with Two-Tower Neural Retrieval

### 3.1 Objective

The candidate generator should retrieve a manageable set of articles that contains as many future purchased articles as possible.

It is not responsible for perfect ranking. Its main target is **high recall at K**.

Recommended candidate sizes:

| Use case | Candidate count per customer |
|---|---:|
| Offline experimentation | 100, 300, 500, 1000 |
| LightGBM training | 200-1000 |
| Production serving | Usually 200-500 before filtering |

### 3.2 Two-tower architecture

The two-tower model learns two embedding functions:

```text
user_embedding = UserTower(customer features, purchase history)
item_embedding = ItemTower(article features)
score(customer, article) = dot(user_embedding, item_embedding)
```

At serving time:

1. Precompute article embeddings.
2. Store them in an ANN index such as FAISS, ScaNN, Milvus, or pgvector.
3. Compute customer embedding online or batch it daily.
4. Retrieve nearest articles by dot product or cosine similarity.

### 3.3 Recommended user tower inputs

Use a mix of stable customer metadata and behavioral history.

| Feature group | Examples |
|---|---|
| Customer metadata | age bucket, club member status, fashion news frequency, postal code bucket |
| Purchase behavior | total purchases, purchases in last 7/14/30/90 days |
| Category affinity | top product groups, sections, garment groups, colors |
| Price affinity | average price, median price, price bucket preference |
| Channel behavior | share of purchases by sales channel |
| Sequence summary | last N purchased article IDs or category IDs |

Example user tower input:

```text
customer_id_embedding
+ age_bucket_embedding
+ club_status_embedding
+ last_20_article_sequence_encoder
+ product_group_affinity_vector
+ price_bucket_affinity_vector
+ channel_preference_embedding
```

A strong practical baseline is:

- customer ID embedding for warm customers,
- age bucket embedding,
- club status embedding,
- fashion news frequency embedding,
- average of embeddings of last N purchased articles,
- average of embeddings of recently purchased product groups.

### 3.4 Recommended item tower inputs

Use article metadata and optionally text/image embeddings.

| Feature group | Examples |
|---|---|
| Article ID | article ID embedding |
| Product taxonomy | product type, product group, department, section, garment group |
| Visual attributes | color, perceived color, graphical appearance |
| Text | product name, detail description embeddings |
| Price | typical price bucket, recent median price |
| Lifecycle | article age, first seen date, days since first purchase |

Example item tower input:

```text
article_id_embedding
+ product_type_embedding
+ product_group_embedding
+ department_embedding
+ section_embedding
+ garment_group_embedding
+ color_embedding
+ text_description_embedding
+ price_bucket_embedding
```

### 3.5 Positive training examples

Each purchase is a positive `(customer_id, article_id, t_dat)` interaction.

For temporal correctness, train using only transactions before the fold cutoff.

Example:

```python
positive_pairs = history[["customer_id", "article_id", "t_dat"]].drop_duplicates()
```

For customers with repeated purchases of the same article, either:

- keep repeat purchases and weight by recency, or
- deduplicate per customer-article and add repeat count as a weight.

Recommended weighting:

```text
sample_weight = recency_weight * repeat_weight

recency_weight = exp(-days_since_purchase / half_life_days)
repeat_weight = log1p(customer_article_purchase_count)
```

A half-life of 14-30 days is usually a good starting point for fashion.

### 3.6 Negative sampling for two-tower training

Use in-batch negatives plus optional sampled negatives.

#### In-batch negatives

For a batch of positive pairs:

```text
(customer_1, item_1)
(customer_2, item_2)
...
(customer_B, item_B)
```

Each customer's positive item is the target, and the other `B-1` items act as negatives.

This is efficient and common for retrieval training.

#### Correcting popularity bias

Because popular articles appear more often as negatives, use log-q correction or sampling correction when possible.

Conceptually:

```text
adjusted_score(customer, item) = raw_score(customer, item) - log(item_sampling_probability)
```

This helps avoid a model that retrieves only globally popular articles.

#### Additional hard negatives

After the first model version, mine hard negatives:

1. Train two-tower model v1.
2. For each training customer, retrieve top K candidates at historical cutoffs.
3. Mark retrieved but unpurchased articles as hard negatives.
4. Retrain or fine-tune the model with positives + hard negatives.

Hard negatives are useful because they teach the model to distinguish similar plausible items.

### 3.7 Avoiding candidate leakage

For each validation fold:

- Train item and user towers using history only.
- Build ANN index using article embeddings available at cutoff.
- Generate candidates for validation customers at cutoff.
- Compare generated candidates against label-window purchases.

Do not train candidate embeddings using purchases from the validation label window.

### 3.8 ANN retrieval design

#### Offline experiment

For local experiments, exact dot-product retrieval may be acceptable if the catalog is small enough.

```python
scores = user_embeddings @ item_embeddings.T
candidate_indices = np.argpartition(scores, -K, axis=1)[:, -K:]
```

#### Production-like retrieval

Use approximate nearest neighbor indexing:

```text
Article embeddings -> ANN index
Customer embedding -> ANN search -> top K article IDs
```

Recommended metadata stored with each item vector:

```text
article_id
embedding_version
article_status
first_seen_date
last_seen_date
product_group
section
price_bucket
availability flag, if available
```

### 3.9 Candidate generation output schema

The output of retrieval should be a table like this:

| customer_id | article_id | candidate_source | retrieval_rank | retrieval_score | cutoff_date |
|---|---|---|---:|---:|---|
| C001 | A123 | two_tower_ann | 1 | 0.842 | 2020-09-15 |
| C001 | A456 | two_tower_ann | 2 | 0.818 | 2020-09-15 |
| C001 | A789 | two_tower_ann | 3 | 0.801 | 2020-09-15 |

Even if you only use two-tower candidates, keep `candidate_source`. It makes debugging, ablations, and future expansion easier.

### 3.10 Candidate generation quality gates

Before training the ranker, measure:

| Metric | Why it matters |
|---|---|
| Recall@100 | Can the ranker see enough true positives? |
| Recall@500 | Is candidate coverage acceptable? |
| Recall@1000 | Is retrieval bottleneck too narrow? |
| Avg candidates per customer | Ensures enough ranking options |
| Candidate duplicate rate | Detects candidate generation bugs |
| Catalog coverage | Measures over-concentration on popular articles |
| Segment recall | Checks cold/warm user and long-tail coverage |

---

## 4. Candidate Evaluation

### 4.1 Main metric: recall@K against next-7-day labels

For each customer:

```text
Recall@K = (# purchased articles in next 7 days that appear in top K candidates)
           / (# purchased articles in next 7 days)
```

Aggregate over customers who purchased at least one item in the label window.

### 4.2 Example

Customer C001 purchased these articles in the next 7 days:

```text
Ground truth: [A10, A20, A30]
```

Two-tower top-10 candidates:

```text
[A11, A10, A50, A21, A30, A90, A91, A92, A93, A94]
```

Matched labels:

```text
[A10, A30]
```

So:

```text
Recall@10 = 2 / 3 = 0.667
```

### 4.3 Candidate evaluation table

Create a table for every fold:

| Fold cutoff | Recall@100 | Recall@300 | Recall@500 | Recall@1000 | Avg candidates | Catalog coverage |
|---|---:|---:|---:|---:|---:|---:|
| 2020-08-25 | 0.XX | 0.XX | 0.XX | 0.XX | 500 | XX% |
| 2020-09-01 | 0.XX | 0.XX | 0.XX | 0.XX | 500 | XX% |
| 2020-09-08 | 0.XX | 0.XX | 0.XX | 0.XX | 500 | XX% |
| 2020-09-15 | 0.XX | 0.XX | 0.XX | 0.XX | 500 | XX% |

### 4.4 Segment-level candidate evaluation

Report recall by customer segment:

| Segment | Definition | Why it matters |
|---|---|---|
| New / cold customer | 0 prior purchases | Requires fallback behavior |
| Light customer | 1-2 prior purchases | Sparse personalization |
| Medium customer | 3-10 prior purchases | Standard personalization |
| Heavy customer | >10 prior purchases | Strong behavioral signal |
| Young age bucket | e.g. age < 25 | Different fashion preference |
| Older age bucket | e.g. age >= 45 | Different fashion preference |

Report recall by item segment:

| Segment | Definition |
|---|---|
| Popular articles | Top 10% by historical purchases |
| Mid-tail articles | Middle 40% |
| Long-tail articles | Bottom 50% |
| New articles | First seen within last 14 days |

If Recall@500 is high overall but poor for new articles or light users, the system is not robust.

---

## 5. Ranker Strategy with LightGBM

### 5.1 Objective

The ranker receives candidate customer-article pairs and predicts their relative ordering.

Input:

```text
customer_id, article_id, cutoff_date, retrieval features, customer features, article features, interaction features
```

Output:

```text
score(customer, article)
```

The final recommendation list is the top 12 articles by LightGBM score after filtering and business rules.

### 5.2 Why LightGBM?

LightGBM is a strong choice for tabular ranking because it handles:

- sparse and dense engineered features,
- non-linear feature interactions,
- missing values,
- large datasets,
- learning-to-rank objectives such as LambdaRank.

For this task, use either:

| Training mode | When to use |
|---|---|
| `LGBMRanker` with `objective="lambdarank"` | Preferred for ranking candidates per customer |
| `LGBMClassifier` with binary labels | Useful baseline, easier to debug |

Recommended production path:

1. Start with `LGBMClassifier` to validate features and leakage.
2. Move to `LGBMRanker` once candidate generation and labels are stable.
3. Compare both using MAP@12, not only AUC.

### 5.3 Ranker training data

The ranker dataset should be built from generated candidates, not from arbitrary random product pairs.

For each fold:

```text
1. Generate top K candidates for each customer using the two-tower model at cutoff.
2. Label candidate as 1 if customer purchased article in next 7 days.
3. Label candidate as 0 otherwise.
4. Compute features using only history before cutoff.
5. Train LightGBM using customer groups.
```

### 5.4 Example ranker table

| customer_id | article_id | label | retrieval_rank | retrieval_score | user_purchase_30d | item_pop_7d | user_item_category_affinity | cutoff_date |
|---|---|---:|---:|---:|---:|---:|---:|---|
| C001 | A123 | 1 | 12 | 0.772 | 5 | 230 | 0.80 | 2020-09-15 |
| C001 | A456 | 0 | 3 | 0.851 | 5 | 810 | 0.15 | 2020-09-15 |
| C001 | A789 | 0 | 41 | 0.643 | 5 | 44 | 0.20 | 2020-09-15 |
| C002 | A222 | 1 | 8 | 0.799 | 2 | 92 | 0.70 | 2020-09-15 |

### 5.5 Grouping for LightGBM ranking

For `LGBMRanker`, rows must be grouped by query. In recommendations, the query is usually the customer at a cutoff date.

Group key:

```text
query_id = customer_id + cutoff_date
```

Rows must be sorted by this query key, and LightGBM must receive group sizes.

Example:

```python
ranker_df = ranker_df.sort_values(["query_id", "article_id"])

feature_cols = [
    "retrieval_rank",
    "retrieval_score",
    "user_purchase_7d",
    "user_purchase_30d",
    "item_pop_7d",
    "item_pop_30d",
    "user_product_group_affinity",
    "user_section_affinity",
    "price_distance_from_user_avg",
    "days_since_user_last_purchase",
]

X = ranker_df[feature_cols]
y = ranker_df["label"]
group = ranker_df.groupby("query_id").size().to_numpy()

model = lightgbm.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model.fit(
    X,
    y,
    group=group,
    eval_at=[12],
)
```

### 5.6 Ranking labels

For simple next-7-day purchase prediction:

```text
label = 1 if customer bought article in label window else 0
```

You can also use graded relevance:

| Behavior in label window | Label |
|---|---:|
| No purchase | 0 |
| Purchased once | 1 |
| Purchased multiple times | 2 |
| Purchased recently within label window | 2 or 3 |

Binary labels are simpler and usually sufficient for this dataset.

### 5.7 Candidate/ranker mismatch to avoid

Bad pattern:

```text
Train ranker on random negatives from full catalog.
Serve ranker on two-tower candidates.
```

Better pattern:

```text
Train ranker on the same kind of candidates it will see at serving time.
Serve ranker on two-tower candidates.
```

This ensures the ranker learns to order plausible alternatives rather than separating purchases from obviously irrelevant articles.

---

## 6. Negative Sampling Strategy

### 6.1 Recommended principle

For the LightGBM ranker, negatives should be generated candidates that were not purchased in the next 7 days.

This creates realistic hard negatives.

### 6.2 Ranker negative construction

For each customer and cutoff:

```text
candidates = top K two-tower articles
positives = candidates intersect next-7-day purchases
negatives = candidates minus next-7-day purchases
```

Example:

```text
Top 10 candidates: [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]
Next-7-day purchases: [A3, A8, A20]

Positive rows: A3, A8
Negative rows: A1, A2, A4, A5, A6, A7, A9, A10
Missed positive: A20, because candidate generator failed to retrieve it
```

The missed positive should be counted against candidate recall, but it cannot be used by the ranker unless you intentionally inject positives.

### 6.3 Should positives missed by candidate generation be added?

There are two options.

#### Option A: Strict serving simulation

Do not add missed positives.

Pros:

- Honest evaluation of full pipeline.
- Ranker sees only candidates it would see in production.

Cons:

- Ranker may have very few positive examples if retrieval recall is low.

#### Option B: Positive injection for training only

Add missed positives to the training candidate set, but not to validation/test candidate sets.

Pros:

- More positive examples for ranker learning.
- Helpful early in development.

Cons:

- Can create train/serve mismatch.

Recommended approach:

- Use positive injection only for training if the positive rate is too low.
- Never inject positives into validation or test.
- Track which rows were injected with `candidate_source = "label_injected_positive"`.

### 6.4 Negative downsampling

If each customer has 1000 candidates, the training set can become huge and highly imbalanced.

Use controlled downsampling:

```text
Keep all positives.
Keep top-ranked hard negatives.
Sample additional negatives across retrieval rank buckets.
```

Example per customer:

| Candidate bucket | Sampling strategy |
|---|---|
| Rank 1-50 | Keep all or most negatives |
| Rank 51-200 | Sample 50% |
| Rank 201-1000 | Sample 10-20% |

This preserves hard negatives while controlling dataset size.

### 6.5 Negative sampling table

| Negative type | Source | Use for two-tower | Use for LightGBM |
|---|---|---:|---:|
| In-batch negatives | Other positives in same batch | Yes | No |
| Random catalog negatives | Random articles not bought | Optional | Limited baseline only |
| Popularity negatives | Popular articles not bought | Optional | Useful but insufficient alone |
| ANN hard negatives | Retrieved but not purchased | Yes, for fine-tuning | Strongly recommended |
| Recent popular negatives | Trending but not purchased | Optional | Recommended |

### 6.6 False-negative risk

A negative label means:

```text
The customer did not purchase this item in the next 7 days.
```

It does not mean:

```text
The customer dislikes this item.
```

This matters because many unpurchased items may still be relevant. Use ranking metrics and avoid overinterpreting binary labels as explicit preference.

---

## 7. Feature Engineering Strategy

### 7.1 Feature principles

Features must be:

1. **Cutoff-safe:** computed only from data before the cutoff.
2. **Serving-available:** computable in production at recommendation time.
3. **Behaviorally meaningful:** connected to fashion purchase behavior.
4. **Segment-aware:** robust for cold and warm customers.

### 7.2 Feature categories

| Category | Examples |
|---|---|
| Retrieval features | two-tower rank, score, embedding similarity |
| User features | purchase count, recency, frequency, age bucket |
| Item features | popularity, recency, product group, color, section |
| User-item features | category affinity, price affinity, repeat signals |
| Time features | day of week, season, recent trend windows |
| Diversity/business features | duplicate product code, availability, newness |

### 7.3 Retrieval features

These are among the most important ranker features because they pass information from the two-tower model.

| Feature | Meaning |
|---|---|
| `retrieval_score` | Dot product or cosine similarity from two-tower model |
| `retrieval_rank` | ANN rank position |
| `retrieval_score_percentile` | Normalized score within customer candidate set |
| `score_gap_from_top1` | Difference from best candidate score |
| `candidate_source` | Source identifier, even if currently only `two_tower_ann` |

Example:

```python
candidate_df["score_gap_from_top1"] = (
    candidate_df.groupby("query_id")["retrieval_score"].transform("max")
    - candidate_df["retrieval_score"]
)
```

### 7.4 User behavior features

| Feature | Example definition |
|---|---|
| `user_purchase_count_7d` | Purchases by customer in last 7 days before cutoff |
| `user_purchase_count_30d` | Purchases in last 30 days |
| `user_purchase_count_all` | Lifetime purchases before cutoff |
| `days_since_last_purchase` | Cutoff date minus most recent purchase date |
| `user_unique_articles_30d` | Distinct articles bought in last 30 days |
| `user_avg_price_90d` | Average purchased price in last 90 days |
| `user_channel_2_share` | Fraction of purchases in sales channel 2 |

Example:

```python
def user_purchase_count(history, cutoff, days):
    start = cutoff - pd.Timedelta(days=days)
    recent = history[(history.t_dat > start) & (history.t_dat <= cutoff)]
    return recent.groupby("customer_id").size().rename(f"user_purchase_count_{days}d")
```

### 7.5 Item popularity and trend features

| Feature | Example definition |
|---|---|
| `item_pop_7d` | Article purchases in last 7 days before cutoff |
| `item_pop_30d` | Article purchases in last 30 days |
| `item_pop_ratio_7d_30d` | Short-term trend vs medium-term popularity |
| `item_unique_buyers_30d` | Unique customers who bought article in last 30 days |
| `days_since_item_last_purchase` | Recency of item demand |
| `article_age_days` | Days since first observed purchase |

Example:

```python
item_pop_7d = (
    history[(history.t_dat > cutoff - pd.Timedelta(days=7)) & (history.t_dat <= cutoff)]
    .groupby("article_id")
    .size()
    .rename("item_pop_7d")
)
```

### 7.6 User-category affinity features

Fashion recommendations depend heavily on category preference.

Examples:

| Feature | Meaning |
|---|---|
| `user_product_group_affinity` | Share of user's purchases in candidate article's product group |
| `user_section_affinity` | Share of user's purchases in candidate article's section |
| `user_garment_group_affinity` | Share of user's purchases in candidate garment group |
| `user_color_affinity` | Share of user's purchases in candidate color group |

Example:

```text
Customer bought 20 items historically.
8 were product_group_name = "Garment Upper body".
Candidate article belongs to "Garment Upper body".

user_product_group_affinity = 8 / 20 = 0.40
```

### 7.7 User-item repeat and similarity features

| Feature | Meaning |
|---|---|
| `user_bought_same_article_before` | 1 if customer bought article before cutoff |
| `user_bought_same_product_code_before` | 1 if customer bought another variant of same product code |
| `days_since_user_bought_article` | Recency of repeat purchase |
| `user_bought_same_product_type_count` | Historical count for candidate product type |

Fashion customers often rebuy basics or buy variants of the same product. Product-code-level features are especially useful because one product may have multiple article variants.

### 7.8 Price affinity features

| Feature | Meaning |
|---|---|
| `candidate_price` | Recent median price of article |
| `user_avg_price_90d` | Customer's average historical purchase price |
| `price_distance_from_user_avg` | Absolute or relative price difference |
| `candidate_price_percentile` | Price bucket within catalog or product group |

Example:

```python
ranker_df["price_distance_from_user_avg"] = (
    ranker_df["candidate_price"] - ranker_df["user_avg_price_90d"]
).abs()
```

### 7.9 Time-decay features

Recent behavior is usually more predictive than old behavior.

Use exponential decay:

```text
decayed_count = sum(exp(-days_since_event / half_life_days))
```

Example:

```python
import numpy as np

history["days_since"] = (cutoff - history["t_dat"]).dt.days
history["weight_30d_half_life"] = np.exp(-history["days_since"] / 30)

user_decayed_purchase_count = (
    history.groupby("customer_id")["weight_30d_half_life"]
    .sum()
    .rename("user_decayed_purchase_count")
)
```

### 7.10 Missing values

LightGBM handles missing values, but missingness should still be meaningful.

Recommended practice:

- Keep numeric missing values as `NaN` when absence is informative.
- Add explicit missing indicators for important features.
- Use category value `UNKNOWN` for categorical features.

Example:

```python
ranker_df["age_missing"] = ranker_df["age"].isna().astype("int8")
ranker_df["age"] = ranker_df["age"].fillna(-1)
```

### 7.11 Feature leakage examples

Bad:

```python
# Uses all transactions, including label window and future data.
item_popularity = transactions.groupby("article_id").size()
```

Good:

```python
# Uses only history available at cutoff.
history = transactions[transactions.t_dat <= cutoff]
item_popularity = history.groupby("article_id").size()
```

Bad:

```python
# Computes customer's latest purchase using full dataset.
latest_purchase = transactions.groupby("customer_id").t_dat.max()
```

Good:

```python
latest_purchase = history.groupby("customer_id").t_dat.max()
```

---

## 8. Final Evaluation Strategy

### 8.1 Main metric: MAP@12

The final output is a ranked list of 12 articles per customer. Therefore, the main offline metric should be MAP@12.

For a single customer:

```text
AP@12 = sum(precision@k for each relevant item found at rank k <= 12)
        / min(number of relevant items, 12)
```

Overall:

```text
MAP@12 = mean(AP@12 over customers)
```

### 8.2 Example MAP@12 calculation

Ground truth purchases:

```text
[A, B, C]
```

Predictions:

```text
Rank 1: D
Rank 2: A  -> hit, precision@2 = 1/2
Rank 3: E
Rank 4: B  -> hit, precision@4 = 2/4
Rank 5: C  -> hit, precision@5 = 3/5
```

```text
AP@12 = (1/2 + 2/4 + 3/5) / 3
      = (0.5 + 0.5 + 0.6) / 3
      = 0.5333
```

### 8.3 Evaluation pipeline

For each validation fold:

```text
1. Train two-tower model using history before cutoff.
2. Generate top K candidates for validation customers.
3. Compute candidate recall@K.
4. Build LightGBM features for candidates using history before cutoff.
5. Score candidates with LightGBM.
6. Sort candidates by score per customer.
7. Apply post-processing rules.
8. Keep top 12.
9. Compute MAP@12 against next-7-day purchases.
```

### 8.4 Required evaluation tables

#### Candidate generator table

| Fold | Recall@100 | Recall@300 | Recall@500 | Recall@1000 | Catalog coverage |
|---|---:|---:|---:|---:|---:|
| Fold 1 |  |  |  |  |  |
| Fold 2 |  |  |  |  |  |
| Fold 3 |  |  |  |  |  |
| Fold 4 |  |  |  |  |  |

#### Ranker table

| Fold | MAP@12 | NDCG@12 | Precision@12 | HitRate@12 |
|---|---:|---:|---:|---:|
| Fold 1 |  |  |  |  |
| Fold 2 |  |  |  |  |
| Fold 3 |  |  |  |  |
| Fold 4 |  |  |  |  |

#### Segment table

| Segment | MAP@12 | Recall@500 | Avg candidates | Notes |
|---|---:|---:|---:|---|
| Cold users |  |  |  |  |
| Light users |  |  |  |  |
| Medium users |  |  |  |  |
| Heavy users |  |  |  |  |
| Popular items |  |  |  |  |
| Long-tail items |  |  |  |  |
| New items |  |  |  |  |

### 8.5 Baselines to include

Never evaluate only the advanced model. Include baselines:

| Baseline | Description |
|---|---|
| Global popularity | Recommend most popular articles from recent history |
| Age-bucket popularity | Recommend popular articles within customer age bucket |
| Repurchase baseline | Recommend articles customer bought recently |
| Two-tower only | Use two-tower scores directly without LightGBM |
| Two-tower + LightGBM | Full model |

A model is not production-ready unless it consistently beats simple popularity and repurchase baselines across rolling folds.

---

## 9. Production Readiness Strategy

### 9.1 Offline vs online metrics

Offline metrics are necessary but not sufficient.

| Metric type | Examples | Purpose |
|---|---|---|
| Offline | MAP@12, Recall@500, NDCG@12 | Model selection |
| Online engagement | CTR, add-to-cart rate, purchase rate | User response |
| Business | revenue per user, margin, return rate | Commercial quality |
| System | latency, index freshness, failure rate | Reliability |
| Fairness/diversity | catalog coverage, long-tail exposure | Marketplace health |

### 9.2 Batch training pipeline

Recommended daily or weekly batch flow:

```text
1. Ingest transactions, customers, articles.
2. Validate schema and data freshness.
3. Build cutoff-safe feature snapshots.
4. Train / refresh two-tower model.
5. Generate article embeddings.
6. Build ANN index.
7. Generate candidate sets for active customers.
8. Build LightGBM feature table.
9. Score and produce top-N recommendations.
10. Store recommendations for serving.
11. Log model version, feature version, index version, and cutoff date.
```

### 9.3 Feature store design

Minimum feature stores:

| Store | Key | Contents |
|---|---|---|
| Customer features | `customer_id`, `as_of_date` | recency, frequency, age, affinity vectors |
| Article features | `article_id`, `as_of_date` | popularity, taxonomy, price, lifecycle |
| Customer-article features | `customer_id`, `article_id`, `as_of_date` | affinity, repeat, similarity features |
| Retrieval features | `customer_id`, `article_id`, `as_of_date` | retrieval rank and score |

Feature tables should include:

```text
feature_version
as_of_date
created_at
source_data_max_date
```

### 9.4 ANN index readiness

For ANN retrieval, track:

| Check | Why |
|---|---|
| Embedding dimension consistency | Prevent serving failures |
| Index build date | Ensure freshness |
| Article count in index | Detect missing items |
| Recall vs exact search sample | Validate ANN quality |
| Latency p50/p95/p99 | Ensure serving performance |
| Index version | Reproducibility |

### 9.5 Fallback logic

Two-tower retrieval may fail or return weak candidates for cold customers.

Fallback hierarchy:

```text
1. Personalized two-tower candidates.
2. Customer segment popularity, e.g. age bucket or membership segment.
3. Recent global popularity.
4. New arrivals / editorial picks.
5. Safe default popular catalog.
```

For final top 12, backfill missing slots:

```python
def backfill_top12(personalized, fallback):
    result = []
    seen = set()

    for article in personalized + fallback:
        if article not in seen:
            result.append(article)
            seen.add(article)
        if len(result) == 12:
            break

    return result
```

### 9.6 Business rules

Common production filters:

| Rule | Example |
|---|---|
| Availability | Remove out-of-stock articles |
| Duplicate suppression | Avoid recommending many variants of same product code |
| Recently purchased suppression | Optional, depending on repeat-purchase behavior |
| Age/market eligibility | Remove ineligible products |
| Diversity rule | Limit same product group or color over-concentration |

For Kaggle, stock availability is not provided, so do not simulate availability unless you have external data.

### 9.7 Monitoring

Monitor the full pipeline:

| Area | Metrics |
|---|---|
| Data freshness | latest transaction date, row counts, missing values |
| Candidate generation | recall proxy, candidates per user, ANN latency |
| Ranking | score distribution, top feature drift, prediction drift |
| Recommendation output | duplicate rate, catalog coverage, top article concentration |
| Business | purchase rate, revenue, conversion |

### 9.8 A/B testing plan

A production recommender should be validated online.

Example A/B test:

| Group | Strategy |
|---|---|
| Control | Current popularity or existing recommender |
| Treatment | Two-tower + LightGBM recommender |

Primary metrics:

- purchase conversion rate,
- revenue per user,
- add-to-cart rate.

Guardrail metrics:

- latency,
- return rate,
- customer complaints,
- catalog concentration,
- out-of-stock exposure.

Run the experiment long enough to cover weekly shopping cycles.

---

## 10. End-to-End Training Blueprint

### 10.1 Offline fold loop

```python
for cutoff in validation_cutoffs:
    history = transactions[transactions.t_dat <= cutoff]
    labels = get_next_7_day_labels(transactions, cutoff)

    # Candidate generator
    two_tower_train = build_two_tower_pairs(history)
    two_tower_model = train_two_tower(two_tower_train)

    item_embeddings = encode_items(two_tower_model, articles, cutoff)
    ann_index = build_ann_index(item_embeddings)

    user_embeddings = encode_users(two_tower_model, customers, history, cutoff)
    candidates = retrieve_candidates(ann_index, user_embeddings, top_k=500)

    candidate_metrics = evaluate_candidate_recall(candidates, labels)

    # Ranker
    ranker_rows = label_candidates(candidates, labels)
    ranker_features = build_ranker_features(ranker_rows, history, articles, customers, cutoff)

    lgbm_model = train_lgbm_ranker(ranker_features)
    predictions = score_candidates(lgbm_model, ranker_features)

    top12 = make_top12(predictions)
    ranking_metrics = evaluate_map12(top12, labels)
```

### 10.2 Final training for production or submission

After model selection:

```text
1. Pick the best candidate K, features, and LightGBM parameters from validation.
2. Train two-tower on all available history before final prediction cutoff.
3. Build final ANN index.
4. Generate final candidates for target customers.
5. Train LightGBM using rolling folds or a final training window.
6. Score final candidates.
7. Backfill to 12 recommendations per customer.
8. Export predictions.
```

---

## 11. Recommended File / Module Structure

```text
recommender_project/
  configs/
    data.yaml
    two_tower.yaml
    lightgbm_ranker.yaml
    validation_folds.yaml

  data/
    raw/
    processed/
    features/
    candidates/
    models/
    submissions/

  src/
    data/
      load.py
      schema.py
      validation.py

    splits/
      temporal_folds.py

    features/
      user_features.py
      item_features.py
      interaction_features.py
      retrieval_features.py
      feature_store.py

    retrieval/
      dataset.py
      model.py
      train.py
      encode.py
      ann_index.py
      retrieve.py
      evaluate.py

    ranking/
      build_dataset.py
      train_lgbm.py
      score.py
      evaluate.py

    evaluation/
      mapk.py
      recall.py
      segments.py

    pipelines/
      train_two_tower.py
      generate_candidates.py
      train_ranker.py
      score_final.py

  notebooks/
    01_eda.ipynb
    02_baselines.ipynb
    03_two_tower_debug.ipynb
    04_lgbm_ranker_debug.ipynb

  tests/
    test_temporal_split.py
    test_no_feature_leakage.py
    test_map12.py
    test_candidate_labels.py
```

---

## 12. Practical Milestones

### Milestone 1: Baselines

Deliverables:

- global popularity baseline,
- recent popularity baseline,
- repurchase baseline,
- MAP@12 evaluation.

Success criterion:

```text
Advanced model must beat recent popularity and repurchase baselines on rolling folds.
```

### Milestone 2: Two-tower candidate generator

Deliverables:

- two-tower training pipeline,
- item embeddings,
- customer embeddings,
- top K candidate generation,
- Recall@100/300/500/1000.

Success criterion:

```text
Recall@500 is high enough that the ranker has room to improve MAP@12.
```

### Milestone 3: LightGBM ranker

Deliverables:

- candidate-labeled ranker dataset,
- cutoff-safe features,
- LightGBM ranker,
- MAP@12 evaluation,
- feature importance.

Success criterion:

```text
Two-tower + LightGBM improves MAP@12 over two-tower-only ranking.
```

### Milestone 4: Segment robustness

Deliverables:

- metrics by customer activity level,
- metrics by item popularity bucket,
- cold-user fallback.

Success criterion:

```text
No major segment is dramatically worse without explanation or fallback.
```

### Milestone 5: Production-like pipeline

Deliverables:

- reproducible configs,
- versioned artifacts,
- retraining pipeline,
- monitoring reports,
- ANN index validation.

Success criterion:

```text
A new model can be trained, evaluated, scored, and reproduced from configuration.
```

---

## 13. Common Failure Modes

| Failure mode | Symptom | Fix |
|---|---|---|
| Random split leakage | Great validation, poor real test | Use temporal folds |
| Low candidate recall | Ranker MAP@12 capped | Improve two-tower features/training or increase K |
| Easy negatives | High AUC, poor MAP@12 | Use generated hard negatives |
| Feature leakage | Unrealistically high feature importance for popularity/recency | Recompute features by cutoff |
| Popularity collapse | Same articles recommended to everyone | Add personalization, diversity, and segment monitoring |
| Cold-user weakness | Bad performance for low-history users | Add metadata and fallback candidates |
| Candidate/ranker mismatch | Ranker works offline but fails in serving | Train ranker on served candidate distribution |
| Overlarge candidate set | Slow training/scoring | Rank-bucket negative downsampling |

---

## 14. Minimum Acceptance Checklist

Before calling the system industry-grade, verify:

- [ ] Rolling temporal validation is implemented.
- [ ] Features are computed using cutoff-safe history only.
- [ ] Two-tower candidate recall is measured at multiple K values.
- [ ] Ranker is trained on generated candidates, not arbitrary random pairs.
- [ ] LightGBM grouping is by customer-cutoff query.
- [ ] MAP@12 is the main ranking metric.
- [ ] Baselines are included and beaten consistently.
- [ ] Metrics are reported by customer and item segments.
- [ ] Cold-start and fallback logic exist.
- [ ] ANN index build and retrieval are versioned.
- [ ] Model, feature, candidate, and index versions are tracked.
- [ ] Monitoring is defined for data, model, output, and business metrics.

---

## 15. Suggested Initial Configuration

### 15.1 Two-tower retrieval

```yaml
two_tower:
  embedding_dim: 128
  batch_size: 2048
  epochs: 5
  optimizer: adam
  learning_rate: 0.001
  loss: sampled_softmax_or_in_batch_cross_entropy
  negatives: in_batch
  log_q_correction: true
  recency_half_life_days: 30
  max_user_sequence_length: 20
  candidate_k_values: [100, 300, 500, 1000]
```

### 15.2 LightGBM ranker

```yaml
lightgbm_ranker:
  objective: lambdarank
  metric: ndcg
  eval_at: [12]
  n_estimators: 1000
  learning_rate: 0.03
  num_leaves: 63
  min_child_samples: 100
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  early_stopping_rounds: 100
```

### 15.3 Candidate/ranker dataset

```yaml
ranker_dataset:
  candidate_source: two_tower_ann
  candidates_per_customer: 500
  keep_all_positives: true
  positive_injection_for_training: optional
  positive_injection_for_validation: false
  negative_downsampling:
    rank_1_50: 1.0
    rank_51_200: 0.5
    rank_201_500: 0.2
```

---

## 16. Recommended Next Steps

1. Build the temporal fold generator first.
2. Implement MAP@12 and candidate Recall@K before training advanced models.
3. Build simple popularity and repurchase baselines.
4. Train the first two-tower candidate generator.
5. Generate top 500 candidates per customer for each validation fold.
6. Train a LightGBM classifier baseline on generated candidates.
7. Upgrade to `LGBMRanker` with customer-cutoff groups.
8. Add feature leakage tests.
9. Add segment-level reporting.
10. Add ANN index versioning and fallback logic.

The most important design rule is simple:

> Candidate generation should maximize recall under realistic serving constraints; ranking should optimize MAP@12 using the same candidate distribution that will be served in production.
