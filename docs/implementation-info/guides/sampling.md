Below is an industry-grade sampling strategy for your H&M-style recommender, assuming:

* **Task:** predict purchases in the next 7 days.
* **Candidate generator:** two-tower neural network.
* **Ranker:** LightGBM.
* **Dataset:** articles, customers, and transactions; the uploaded schema shows ~105K articles, ~1.37M customers, and ~31.8M transactions. 

---

# Industry-Grade Data Sampling Strategy for Cost-Efficient Recommender Training

## 1. Core principle: do not randomly sample transactions blindly

Avoid doing this:

```python
transactions_sample = transactions.sample(frac=0.1)
```

That breaks recommender-system structure because:

* user histories become incomplete,
* temporal order is damaged,
* item popularity becomes distorted,
* repeat-purchase behavior is weakened,
* validation no longer simulates production,
* ranker training candidates become unrealistic.

Instead, sample along these dimensions:

| Dimension  | Good practice                                                      |
| ---------- | ------------------------------------------------------------------ |
| Time       | Use recent time windows                                            |
| Users      | Sample users, then keep their full history in the window           |
| Items      | Keep full eligible catalog, avoid overly aggressive item filtering |
| Positives  | Downsample positives carefully, preferably with recency weighting  |
| Negatives  | Sample many realistic negatives from retrieved candidates          |
| Candidates | Generate smaller top-K lists during experimentation                |
| Validation | Keep validation temporally correct and stable                      |

---

# 2. Recommended dataset sizes by experiment phase

Use different dataset sizes for different stages.

## Phase 1: Smoke test

Use this only to verify pipeline correctness.

```text
Train window: last 4 weeks
Validation window: next 7 days
Users: 20K to 50K
Two-tower candidates: top 100
LightGBM negatives per positive: 20 to 50
```

Purpose:

* check joins,
* check feature generation,
* check label creation,
* check model training,
* check MAP@12 implementation.

Do not trust model quality from this phase.

---

## Phase 2: Fast iteration

Use this for feature engineering and model debugging.

```text
Train window: last 8 to 12 weeks
Validation window: next 7 days
Users: 100K to 300K
Two-tower candidates: top 200 to 500
LightGBM negatives per positive: 50 to 100
```

Purpose:

* compare feature sets,
* tune two-tower architecture,
* tune LightGBM parameters,
* test negative-sampling strategies,
* validate leakage controls.

This is usually the best cost-quality tradeoff.

---

## Phase 3: Strong offline experiment

Use this before final training or serious benchmarking.

```text
Train window: last 16 to 24 weeks
Validation window: next 7 days
Users: 500K to 1M
Two-tower candidates: top 500 to 1000
LightGBM negatives per positive: 100 to 300
```

Purpose:

* reliable model comparison,
* stable MAP@12,
* segment metrics,
* candidate recall measurement.

---

## Phase 4: Final training

Use as much data as practical.

```text
Train window: 6 to 12 months, or all useful history
Validation: rolling temporal folds
Two-tower candidates: top 1000+
LightGBM candidates: top 500 to 1000 per user
```

For fashion, very old transactions may be less useful because trends change. Recent data often matters more than full historical volume.

---

# 3. Temporal sampling

For this task, time-based sampling is the most important.

## Recommended split

Assume the validation target week starts at `T`.

```text
Feature/candidate cutoff: T
Validation labels: T to T+7 days
Training history: T-N weeks to T
```

Example:

```text
Training window: 2020-06-01 to 2020-09-15
Validation label window: 2020-09-16 to 2020-09-22
```

For fast experiments, use only the last 8 to 12 weeks before the validation week.

```python
train_start = "2020-07-01"
cutoff_date = "2020-09-15"
label_start = "2020-09-16"
label_end = "2020-09-22"

train_txn = transactions[
    (transactions["t_dat"] >= train_start) &
    (transactions["t_dat"] <= cutoff_date)
]

valid_labels = transactions[
    (transactions["t_dat"] >= label_start) &
    (transactions["t_dat"] <= label_end)
]
```

## Why this is industry-grade

This simulates the real production situation:

```text
At time T, recommend items using only data available until T.
Then measure whether the customer purchased those items in the next 7 days.
```

---

# 4. User sampling

For cost reduction, sample users rather than transactions.

## Bad approach

```python
transactions.sample(frac=0.1)
```

## Better approach

```python
sampled_customers = customers.sample(n=200_000, random_state=42)

train_txn_sample = train_txn[
    train_txn["customer_id"].isin(sampled_customers["customer_id"])
]
```

But this is still not enough. You should use **stratified user sampling**.

---

## Recommended user segments

Create segments such as:

| Segment             | Definition                                               |
| ------------------- | -------------------------------------------------------- |
| Cold users          | 1 to 2 historical purchases                              |
| Light users         | 3 to 5 historical purchases                              |
| Medium users        | 6 to 20 historical purchases                             |
| Heavy users         | 20+ historical purchases                                 |
| New users           | no transaction before cutoff but known in customer table |
| Active recent users | purchased in last 14 or 30 days                          |
| Dormant users       | purchased historically but not recently                  |

Example:

```python
user_stats = (
    train_txn
    .groupby("customer_id")
    .agg(
        txn_count=("article_id", "count"),
        unique_items=("article_id", "nunique"),
        last_purchase_date=("t_dat", "max")
    )
    .reset_index()
)

def user_segment(txn_count):
    if txn_count <= 2:
        return "cold"
    elif txn_count <= 5:
        return "light"
    elif txn_count <= 20:
        return "medium"
    else:
        return "heavy"

user_stats["segment"] = user_stats["txn_count"].apply(user_segment)
```

Then sample from each group.

```python
sampled_users = (
    user_stats
    .groupby("segment", group_keys=False)
    .apply(lambda x: x.sample(
        n=min(len(x), 50_000),
        random_state=42
    ))
)
```

## Good sampling ratio

For fast experiments:

| Segment      | Suggested count |
| ------------ | --------------: |
| Cold users   |             50K |
| Light users  |             50K |
| Medium users |             50K |
| Heavy users  |             50K |

This gives a 200K-user training sample while preserving different user behaviors.

---

# 5. Deterministic hash-based user sampling

In industry, sampling should be reproducible.

Instead of random sampling every time, assign users to buckets.

```python
import hashlib

def user_bucket(customer_id, num_buckets=100):
    h = hashlib.md5(customer_id.encode()).hexdigest()
    return int(h, 16) % num_buckets

customers["bucket"] = customers["customer_id"].apply(user_bucket)
```

Then choose fixed buckets.

```python
sampled_customers = customers[customers["bucket"].isin([0, 1, 2, 3, 4])]
```

This gives roughly a 5% user sample.

Benefits:

* reproducible,
* easy to scale from 1% to 5% to 20%,
* avoids accidental train/validation mismatch,
* makes experiments comparable.

---

# 6. Keep full histories for sampled users

After sampling users, keep all their transactions inside the selected time window.

```python
sampled_train_txn = train_txn[
    train_txn["customer_id"].isin(sampled_customers["customer_id"])
]
```

Do not sample only a few transactions per user unless the user is extremely heavy.

For very heavy users, you may cap history length.

```python
sampled_train_txn = (
    sampled_train_txn
    .sort_values(["customer_id", "t_dat"])
    .groupby("customer_id")
    .tail(100)
)
```

Recommended cap:

| User type             | Max transactions to keep |
| --------------------- | -----------------------: |
| Cold/light users      |                 keep all |
| Medium users          |                 keep all |
| Heavy users           |          last 100 to 300 |
| Extremely heavy users |          last 300 to 500 |

For fashion, recent behavior is usually more valuable than very old behavior.

---

# 7. Article/item sampling

Be careful with item sampling.

## Do not aggressively remove rare articles

This is risky:

```python
articles = articles[articles["purchase_count"] >= 20]
```

Why?

* removes long-tail products,
* hurts cold-start item learning,
* distorts catalog distribution,
* makes validation easier than production.

## Better strategy

Use different item sets for different purposes.

### 7.1 Candidate-generator training item set

For two-tower training, you may filter extremely rare items only for initial experiments.

```text
Keep articles with at least 3 to 5 purchases in the training window.
Always keep articles that appear in validation labels.
Always keep active catalog items if available.
```

Example:

```python
item_counts = train_txn.groupby("article_id").size().reset_index(name="cnt")

eligible_items = item_counts[item_counts["cnt"] >= 5]["article_id"]

train_txn_two_tower = train_txn[
    train_txn["article_id"].isin(eligible_items)
]
```

### 7.2 Retrieval index item set

For evaluation and serving simulation, the item index should be broader.

```text
ANN index should include all candidate-eligible articles,
not only frequent training articles.
```

For H&M Kaggle-style work, if you do not know active inventory, use articles seen recently as eligible.

```python
recent_items = transactions[
    (transactions["t_dat"] >= recent_item_start) &
    (transactions["t_dat"] <= cutoff_date)
]["article_id"].unique()
```

A practical choice:

```text
Eligible item catalog = articles purchased at least once in the last 4 to 8 weeks before cutoff.
```

---

# 8. Positive sampling for two-tower model

Two-tower training usually uses positive user-item interactions.

For a sampled user, positives are purchased articles before cutoff.

```text
Input: customer_id, article_id
Label: positive interaction
```

## Recency-weighted positive sampling

Do not treat a purchase from 18 months ago the same as a purchase yesterday.

Use recency weights.

Example:

```python
import numpy as np

train_txn["days_before_cutoff"] = (
    pd.to_datetime(cutoff_date) - pd.to_datetime(train_txn["t_dat"])
).dt.days

train_txn["sample_weight"] = np.exp(-train_txn["days_before_cutoff"] / 30)
```

Interpretation:

```text
Recent purchases receive higher weight.
Old purchases still contribute, but less.
```

Typical decay values:

| Half-life / scale | Use case             |
| ----------------- | -------------------- |
| 7 days            | very trend-sensitive |
| 14 days           | fast fashion         |
| 30 days           | balanced             |
| 60 to 90 days     | more stable catalog  |

For H&M, start with 30 days.

---

# 9. Negative sampling for two-tower model

For two-tower retrieval, use one or more of:

## 9.1 In-batch negatives

Each batch contains positive pairs:

```text
(customer_1, item_1)
(customer_2, item_2)
(customer_3, item_3)
...
```

For `customer_1`, the other items in the batch become negatives:

```text
item_2, item_3, ...
```

This is efficient and standard.

## 9.2 Popularity-adjusted negatives

If you sample negatives from item popularity, avoid overfitting to only popular items.

Use smoothed popularity:

```python
item_pop = train_txn["article_id"].value_counts()
sampling_prob = item_pop ** 0.75
sampling_prob = sampling_prob / sampling_prob.sum()
```

Why exponent `0.75`?

It reduces the dominance of very popular products while still sampling realistic items.

## 9.3 Hard negatives

After training an initial two-tower model:

1. retrieve top-K items for each user,
2. remove purchased positives,
3. use high-scoring unpurchased items as hard negatives,
4. continue training or use them for the LightGBM ranker.

Example:

```text
User bought: item A
Two-tower retrieves: A, B, C, D, E
Hard negatives: B, C, D, E
```

These are much better negatives than random catalog items.

---

# 10. Candidate generation sampling with two-tower model

Since you want to stick with a two-tower model for candidate generation, the candidate pipeline should look like this:

```text
1. Train two-tower model on sampled temporal training data.
2. Encode all sampled users.
3. Encode all eligible articles.
4. Build ANN index over article embeddings.
5. Retrieve top-K articles per user.
6. Use retrieved candidates to train LightGBM ranker.
```

## Experiment-time top-K

To save cost, reduce K during early experiments.

| Phase              | Two-tower top-K |
| ------------------ | --------------: |
| Smoke test         |             100 |
| Fast iteration     |             200 |
| Serious experiment |             500 |
| Final              |           1000+ |

For LightGBM training, top 200 to 500 candidates per user is often enough for fast iteration.

---

# 11. LightGBM ranker sampling

The LightGBM ranker should not be trained on random user-item pairs.

It should be trained on **candidate rows generated by the two-tower model**.

## Correct ranker training format

Each row should look like:

| customer_id | article_id | label | feature_1 | feature_2 | feature_3 |
| ----------- | ---------- | ----: | --------: | --------: | --------: |
| C1          | A1         |     1 |       ... |       ... |       ... |
| C1          | A2         |     0 |       ... |       ... |       ... |
| C1          | A3         |     0 |       ... |       ... |       ... |
| C2          | A4         |     1 |       ... |       ... |       ... |
| C2          | A5         |     0 |       ... |       ... |       ... |

Label:

```text
1 = customer purchased this article in next 7 days
0 = customer did not purchase this article in next 7 days
```

Group:

```text
group = number of candidate rows per customer
```

Example:

```python
group = ranker_train_df.groupby("customer_id").size().values
```

Then LightGBM ranks items within each user group.

---

# 12. Negative sampling for LightGBM

This is where most cost savings should happen.

If you have:

```text
200K users × 500 candidates = 100M ranker rows
```

That may be too expensive.

So sample negatives per user.

## Keep all positives

Always keep every positive candidate.

```python
positives = candidates[candidates["label"] == 1]
```

## Sample negatives

```python
negatives = candidates[candidates["label"] == 0]
```

Recommended negative sampling per user:

| Phase              |               Negatives per user |
| ------------------ | -------------------------------: |
| Smoke test         |                               20 |
| Fast iteration     |                               50 |
| Serious experiment |                       100 to 300 |
| Final              | 300 to 1000, depending on budget |

Example:

```python
def sample_negatives(group, max_negatives=100):
    pos = group[group["label"] == 1]
    neg = group[group["label"] == 0]

    neg_sample = neg.sample(
        n=min(len(neg), max_negatives),
        random_state=42
    )

    return pd.concat([pos, neg_sample])

ranker_train_sample = (
    candidates
    .groupby("customer_id", group_keys=False)
    .apply(sample_negatives, max_negatives=100)
)
```

---

# 13. Better LightGBM negative sampling: mix easy and hard negatives

Do not sample negatives uniformly only.

Use a mix:

| Negative type    | Meaning                                | Share |
| ---------------- | -------------------------------------- | ----: |
| Hard negatives   | high two-tower score but not purchased |   60% |
| Medium negatives | middle-ranked candidates               |   25% |
| Easy negatives   | low-ranked candidates                  |   15% |

Example:

```python
def sample_ranker_negatives(group, n_hard=60, n_medium=25, n_easy=15):
    pos = group[group["label"] == 1]
    neg = group[group["label"] == 0].sort_values("two_tower_score", ascending=False)

    hard = neg.head(200).sample(
        n=min(n_hard, len(neg.head(200))),
        random_state=42
    )

    medium_pool = neg.iloc[200:1000]
    medium = medium_pool.sample(
        n=min(n_medium, len(medium_pool)),
        random_state=42
    )

    easy_pool = neg.iloc[1000:]
    easy = easy_pool.sample(
        n=min(n_easy, len(easy_pool)),
        random_state=42
    )

    return pd.concat([pos, hard, medium, easy])
```

For top-K recommendation, hard negatives are very important because the ranker must learn subtle differences among plausible products.

---

# 14. User sampling for LightGBM ranker

For ranker training, not all users are equally useful.

Prioritize users who have at least one validation-period purchase.

```python
positive_users = valid_labels["customer_id"].unique()
```

But do not train only on positive users, because production also includes users who may buy nothing.

Recommended ranker user mix:

| User type                          |      Share |
| ---------------------------------- | ---------: |
| Users with next-7-day purchases    | 60% to 80% |
| Users without next-7-day purchases | 20% to 40% |

Example:

```python
buyers = valid_user_table[valid_user_table["has_purchase_next_7d"] == 1]
non_buyers = valid_user_table[valid_user_table["has_purchase_next_7d"] == 0]

sampled_buyers = buyers.sample(n=150_000, random_state=42)
sampled_non_buyers = non_buyers.sample(n=50_000, random_state=42)

ranker_users = pd.concat([sampled_buyers, sampled_non_buyers])
```

Why include non-buyers?

Because in real serving, many users will not purchase anything. Their candidate rows still help calibrate scores and reduce overconfident recommendations.

---

# 15. Sampling validation data

Be careful here.

## Best practice

Do not heavily sample validation when reporting final metrics.

For development, you may use a smaller validation sample, but keep it deterministic and representative.

```text
Validation sample:
- fixed hash bucket users,
- all their candidates,
- all their next-7-day labels.
```

Example:

```python
valid_users = customers[customers["bucket"].isin([10, 11])]
```

For final reporting:

```text
Use all validation users if possible.
```

At minimum, report:

```text
Sampled validation MAP@12
Full validation MAP@12, if affordable
```

---

# 16. Rolling validation with sampled data

For industry-grade evaluation, use multiple temporal folds.

Example:

```text
Fold 1:
Train: weeks 1-8
Validate: week 9

Fold 2:
Train: weeks 2-9
Validate: week 10

Fold 3:
Train: weeks 3-10
Validate: week 11
```

For cheaper experiments:

```text
Use 1 fold for quick iteration.
Use 3 folds for serious comparison.
Use 5+ folds for final validation.
```

Important: use the same sampled user buckets across folds.

```text
User sample = buckets 0-9
Fold changes time, not user-sampling logic.
```

This makes experiments comparable.

---

# 17. Feature sampling and precomputation

Feature engineering can be expensive. Use precomputed rolling features.

## Recommended feature windows

For each cutoff date, compute:

| Feature type           | Windows                                    |
| ---------------------- | ------------------------------------------ |
| item popularity        | 1 day, 3 days, 7 days, 14 days, 30 days    |
| user category affinity | 7 days, 30 days, 90 days                   |
| user price affinity    | 30 days, 90 days                           |
| user color affinity    | 30 days, 90 days                           |
| item recency           | days since first/last purchase             |
| user recency           | days since last purchase                   |
| user-item history      | previous purchase count, last purchase gap |

Example:

```python
def item_popularity_features(transactions, cutoff_date):
    cutoff = pd.to_datetime(cutoff_date)

    features = []

    for window in [1, 3, 7, 14, 30]:
        start = cutoff - pd.Timedelta(days=window)

        tmp = transactions[
            (transactions["t_dat"] > start) &
            (transactions["t_dat"] <= cutoff)
        ]

        pop = (
            tmp.groupby("article_id")
            .size()
            .reset_index(name=f"item_pop_{window}d")
        )

        features.append(pop)

    return features
```

During sampling, do not recompute features on the entire dataset every time. Cache features by cutoff date.

---

# 18. Cost-saving feature strategy

Use feature tiers.

## Tier 1: cheap features

Use these in all experiments:

```text
two_tower_score
two_tower_rank
item_popularity_7d
item_popularity_30d
user_purchase_count_30d
user_last_purchase_days
article_last_purchase_days
price
age
product_type
product_group
department
section
colour_group
```

## Tier 2: medium-cost features

Use after pipeline is stable:

```text
user-product_type affinity
user-colour affinity
user-department affinity
user-price bucket affinity
item co-purchase score
repeat-purchase flag
same-product-code flag
```

## Tier 3: expensive features

Use only for serious experiments:

```text
sequence embeddings
image embeddings
text embeddings
multi-cutoff rolling features
deep user history features
```

---

# 19. Recommended end-to-end sampled pipeline

## Step 1: Choose cutoff

```text
cutoff = 2020-09-15
label window = 2020-09-16 to 2020-09-22
training window = previous 12 weeks
```

## Step 2: Sample users deterministically

```text
Use hash buckets 0-9 for 10% sample.
Stratify if needed.
```

## Step 3: Build two-tower training data

```text
Use sampled users.
Keep full histories in training window.
Use recency-weighted positives.
Use in-batch negatives.
```

## Step 4: Train two-tower model

```text
Train on sampled interactions.
Evaluate Recall@100, Recall@500.
```

## Step 5: Generate candidates

```text
Encode sampled validation users.
Encode eligible item catalog.
Retrieve top 500 items per user.
```

## Step 6: Label candidates

```text
label = 1 if user bought article in next 7 days else 0
```

## Step 7: Sample LightGBM ranker rows

```text
Keep all positives.
Sample 100 negatives per user:
- 60 hard
- 25 medium
- 15 easy
```

## Step 8: Train LightGBM ranker

```text
Use customer_id as group.
Optimize ranking objective.
Evaluate MAP@12.
```

## Step 9: Compare against baseline

Always compare against:

```text
recent popularity baseline
repurchase baseline
two-tower only
two-tower + LightGBM
```

---

# 20. Practical recommended configuration for your project

For your current project, I would start with this:

```yaml
experiment_name: hm_sampled_industry_v1

time_split:
  train_weeks: 12
  validation_days: 7
  rolling_folds: 1

user_sampling:
  method: deterministic_hash_bucket
  train_user_fraction: 0.10
  preserve_full_user_history: true
  stratify_by_activity: true

item_sampling:
  min_item_purchases_for_training: 5
  retrieval_catalog: items_purchased_in_last_8_weeks
  keep_validation_label_items: true

two_tower:
  positives: sampled_user_transactions
  positive_weighting: recency_decay
  recency_decay_days: 30
  negatives: in_batch
  embedding_dim: 64
  candidate_top_k_for_training: 500
  candidate_top_k_for_eval: 500

lightgbm_ranker:
  candidate_source: two_tower_top_k
  keep_all_positives: true
  negatives_per_user: 100
  negative_mix:
    hard: 0.60
    medium: 0.25
    easy: 0.15
  group_key: customer_id
  objective: lambdarank
  eval_metric: map_at_12

evaluation:
  candidate_metrics:
    - recall_at_100
    - recall_at_500
  ranker_metrics:
    - map_at_12
    - ndcg_at_12
    - precision_at_12
  segment_metrics:
    - cold_users
    - light_users
    - medium_users
    - heavy_users
```

---

# 21. What not to sample

Avoid sampling these aggressively:

## Do not sample validation labels randomly

Bad:

```python
valid_labels.sample(frac=0.1)
```

Better:

```python
sample validation users, then keep all their labels
```

## Do not sample away all cold users

Cold users are common in production.

## Do not remove long-tail items from evaluation

You may filter rare items during training, but evaluation should reflect realistic catalog behavior.

## Do not generate ranker negatives from the full random catalog only

Random negatives are usually too easy.

Use two-tower retrieved but unpurchased items.

## Do not train LightGBM on positives plus random article negatives only

That creates a classifier, not a recommender ranker.

---

# 22. Simple sampling recipe to implement first

This is the most practical starting point:

```python
# 1. Select recent training window
train_txn = transactions[
    (transactions["t_dat"] >= train_start) &
    (transactions["t_dat"] <= cutoff_date)
]

valid_txn = transactions[
    (transactions["t_dat"] > cutoff_date) &
    (transactions["t_dat"] <= label_end)
]

# 2. Deterministic user sampling
customers["bucket"] = customers["customer_id"].apply(lambda x: user_bucket(x, 100))
sampled_users = customers[customers["bucket"] < 10]["customer_id"]

# 3. Keep full histories for sampled users
train_txn_sample = train_txn[
    train_txn["customer_id"].isin(sampled_users)
]

# 4. Two-tower training positives
two_tower_train = train_txn_sample.copy()

# 5. Train two-tower with in-batch negatives

# 6. Retrieve top 500 candidates per sampled validation user

# 7. Label candidates
# label = 1 if (customer_id, article_id) appears in valid_txn

# 8. LightGBM row sampling
# keep all positives, sample 100 negatives per user

# 9. Train LightGBM ranker grouped by customer_id

# 10. Evaluate MAP@12
```

---

# Final recommendation

For your project, the best industry-grade cost-reduction strategy is:

```text
Do not sample transactions randomly.

Sample users deterministically,
keep full recent histories,
train two-tower on recency-weighted positives,
retrieve top-K candidates,
train LightGBM only on sampled candidate rows,
keep all positives,
sample realistic hard negatives,
and evaluate on a fixed temporal validation set.
```

Start with:

```text
12-week training window
10% deterministic user sample
top 500 two-tower candidates
100 LightGBM negatives per user
1 temporal validation fold
MAP@12 evaluation
```

Then scale up gradually:

```text
10% users → 25% users → 50% users → full
top 200 → top 500 → top 1000
1 fold → 3 folds
```
