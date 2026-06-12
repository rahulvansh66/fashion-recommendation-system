Yes. In industry systems, **training on only the last 6 months of transactions does not mean you must throw away older history completely**.

A good design is:

> Use recent transactions for **model training labels and high-signal behavior**, but use longer historical data to create **compressed, cutoff-safe features, priors, embeddings, seasonal indicators, and pretraining signals**.

For H&M, this matters because the transactions table spans multiple years and supports seasonal analysis, while the dataset is large enough that using all raw transactions everywhere may be expensive: ~31.8M transaction rows, ~1.37M customers, and ~105K products. 

---

# Recommended strategy

## 1. Use 6 months for supervised training rows, but 1-3 years for feature history

For the LightGBM ranker, you can limit training examples to recent customer-article candidate rows:

```text
Training rows:
  customer-article candidates from last 6 months

Labels:
  purchase in next 7 days

Features:
  computed from multiple lookback windows:
    7 days
    14 days
    30 days
    90 days
    180 days
    365 days
    same season last year
```

This gives you smaller training data while preserving long-term behavior.

Example feature table:

| Feature                             |      Lookback | Purpose                   |
| ----------------------------------- | ------------: | ------------------------- |
| `item_pop_7d`                       |        7 days | Current trend             |
| `item_pop_30d`                      |       30 days | Recent popularity         |
| `item_pop_180d`                     |      6 months | Stable demand             |
| `item_pop_365d`                     |        1 year | Long-term popularity      |
| `item_pop_same_month_last_year`     | previous year | Seasonality               |
| `user_category_affinity_365d`       |        1 year | Long-term user preference |
| `user_recent_category_affinity_30d` |       30 days | Current intent            |

So the raw **training examples** are recent, but the **features** can summarize older behavior.

---

## 2. Use multi-window features instead of full-history rows

You usually do not need all historical transaction rows as individual training samples. You need their signal.

For example, instead of training on every purchase from 3 years, compute:

```text
item_pop_7d
item_pop_30d
item_pop_90d
item_pop_180d
item_pop_365d
```

Then LightGBM can learn patterns like:

```text
High 7-day popularity + low 365-day popularity = emerging trend
Low 7-day popularity + high same-season-last-year popularity = seasonal item
High 365-day popularity + stable 30-day popularity = evergreen product
```

Example:

| article_id | item_pop_7d | item_pop_30d | item_pop_365d | item_pop_same_month_last_year | Interpretation               |
| ---------- | ----------: | -----------: | ------------: | ----------------------------: | ---------------------------- |
| A001       |         800 |         1400 |          1600 |                            50 | New trend                    |
| A002       |          30 |          120 |          9000 |                          1100 | Evergreen but currently weak |
| A003       |         200 |          700 |          1200 |                           950 | Seasonal comeback            |
| A004       |           5 |           20 |          3000 |                            40 | Declining item               |

This is much cheaper than using every old transaction as a training row.

---

## 3. Add explicit seasonality features

For fashion, seasonality is very important. You can keep seasonal memory without training on all old rows.

Useful seasonality features:

| Feature                              | Example                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| Month                                | September                                                |
| Week of year                         | Week 38                                                  |
| Season                               | autumn                                                   |
| Same-month item popularity last year | item purchases in September previous year                |
| Same-category popularity last year   | product group purchases in same week/month previous year |
| Color seasonality                    | black/white/bright color trends by month                 |
| Garment seasonality                  | swimwear in summer, knitwear in winter                   |

Example:

```text
candidate article: wool sweater
current cutoff: 2020-09-15

Features:
  product_group = Garment Upper body
  garment_group = Knitwear
  item_pop_30d = 120
  garment_group_pop_same_month_last_year = 5000
  product_type_pop_same_month_last_year = 1800
  current_month = 9
  current_season = autumn
```

Even if the article itself has little recent activity, the model can learn that knitwear becomes relevant around autumn.

---

## 4. Use decayed historical features

Instead of hard-cutting history at 6 months, use time decay.

Recent transactions receive high weight. Older transactions still contribute, but less.

Example:

```text
weight = exp(-days_since_event / half_life_days)
```

For fashion, you can use multiple half-lives:

| Feature      |    Half-life | Captures          |
| ------------ | -----------: | ----------------- |
| Fast decay   |    7-14 days | short-term trend  |
| Medium decay |   30-60 days | current season    |
| Slow decay   | 180-365 days | stable preference |

Example:

```python
history["days_since"] = (cutoff_date - history["t_dat"]).dt.days

history["weight_30d"] = np.exp(-history["days_since"] / 30)
history["weight_180d"] = np.exp(-history["days_since"] / 180)
history["weight_365d"] = np.exp(-history["days_since"] / 365)
```

Then compute:

```python
item_decayed_pop_30d = history.groupby("article_id")["weight_30d"].sum()
item_decayed_pop_180d = history.groupby("article_id")["weight_180d"].sum()
item_decayed_pop_365d = history.groupby("article_id")["weight_365d"].sum()
```

This lets older transactions influence the model without dominating recent intent.

---

## 5. Use long history for user preference profiles

For users, older purchases are often useful because style preferences can be stable.

But you should separate:

```text
Recent intent = what the user is shopping for now
Long-term taste = what the user generally likes
```

Example user features:

| Feature                            |    Lookback | Meaning                 |
| ---------------------------------- | ----------: | ----------------------- |
| `user_product_group_affinity_30d`  |     30 days | current shopping intent |
| `user_product_group_affinity_365d` |      1 year | long-term taste         |
| `user_color_affinity_365d`         |      1 year | preferred colors        |
| `user_price_avg_180d`              |    6 months | price preference        |
| `user_price_avg_730d`              |     2 years | long-term price level   |
| `days_since_last_purchase`         | all history | activity recency        |

Example:

```text
Customer long-term history:
  60% Garment Upper body
  20% Shoes
  10% Accessories
  10% Other

Customer last 30 days:
  70% Baby/Children
  20% Accessories
  10% Garment Upper body
```

This could mean the customer usually buys womenswear, but currently has child-related shopping intent. Both signals matter.

---

## 6. Use long history for item lifecycle features

Some articles may be old, seasonal, evergreen, or declining. You can detect this from long history.

Useful features:

| Feature                            | Meaning                  |
| ---------------------------------- | ------------------------ |
| `article_first_seen_date`          | when item first appeared |
| `article_age_days`                 | how old the item is      |
| `article_last_seen_date`           | last purchase date       |
| `days_since_article_last_purchase` | demand recency           |
| `item_lifetime_sales`              | total historical demand  |
| `item_recent_to_lifetime_ratio`    | trend strength           |
| `item_same_season_sales_last_year` | seasonal demand          |

Example:

```text
item_recent_to_lifetime_ratio = item_pop_30d / item_pop_all_time
```

Interpretation:

|  Ratio | Meaning            |
| -----: | ------------------ |
|   High | currently trending |
|    Low | old or declining   |
| Medium | stable demand      |

---

## 7. Pretrain two-tower on long history, fine-tune on recent 6 months (my note: skip this step)

For your two-tower model, this is one of the best strategies.

Instead of training only on 6 months:

```text
Step 1: Pretrain two-tower on 1-3 years of transactions
Step 2: Fine-tune on recent 6 months with higher recency weighting
Step 3: Generate candidates using the fine-tuned model
```

This gives you:

| Stage                          | Learns                             |
| ------------------------------ | ---------------------------------- |
| Pretraining on long history    | broad user-item/category structure |
| Fine-tuning on recent 6 months | current trends and recent demand   |

Example:

```text
Pretraining:
  history = all transactions before cutoff
  objective = in-batch retrieval loss
  sample_weight = slow decay, e.g. half-life 365 days

Fine-tuning:
  history = last 6 months before cutoff
  objective = same retrieval loss
  sample_weight = faster decay, e.g. half-life 30 days
```

This is a strong industry pattern because neural retrieval models benefit from more interaction data, but final candidate generation should still reflect recent behavior.

---

## 8. Train LightGBM on 6-month rows, but include long-history features

For LightGBM, you usually do **not** need to train on 3 years of rows. That can overweight outdated fashion behavior.

Better:

```text
LightGBM training examples:
  generated candidates from recent 6 months

LightGBM features:
  short-term + medium-term + long-term aggregates
```

Example LightGBM feature set:

```text
retrieval_score
retrieval_rank

item_pop_7d
item_pop_30d
item_pop_180d
item_pop_365d
item_same_month_pop_last_year

user_purchase_count_30d
user_purchase_count_180d
user_purchase_count_365d

user_product_group_affinity_30d
user_product_group_affinity_365d

price_distance_from_user_avg_90d
price_distance_from_user_avg_365d

article_age_days
days_since_article_last_purchase
item_recent_to_lifetime_ratio
```

This lets the model learn both:

```text
what is currently popular
```

and

```text
what has historically worked during this season or for this user
```

---

## 9. Use sampling weights instead of hard filtering

If you include some older training examples, do not treat them equally.

You can train on:

```text
last 6 months: full sampling rate
6-12 months ago: partial sampling rate
12-24 months ago: small sampling rate
```

Example:

| Transaction age | Sampling rate | Weight |
| --------------- | ------------: | -----: |
| 0-6 months      |          100% |    1.0 |
| 6-12 months     |           30% |    0.5 |
| 12-24 months    |           10% |    0.2 |
| >24 months      |            5% |    0.1 |

This keeps historical coverage without letting old behavior dominate.

For LightGBM:

```python
sample_weight = np.where(days_old <= 180, 1.0,
                np.where(days_old <= 365, 0.5, 0.2))
```

For two-tower:

```python
sample_weight = np.exp(-days_since_purchase / 180)
```

---

## 10. Use historical data to create priors for cold-start

Long history is especially useful for cold users and low-activity users.

For customers with little recent behavior, use priors such as:

```text
age_bucket_popularity
postal_code_bucket_popularity
club_member_status_popularity
global_recent_popularity
seasonal_category_popularity
```

Example:

```text
Customer:
  age = 24
  no purchases in last 6 months
  historical purchases: very sparse

Fallback:
  top articles among age 20-25 users in last 30 days
  plus same-season popular categories
```

So even when training rows are recent, older data can support stable segment-level priors.

---

# Best design for your case

Since you want:

* **Two-tower neural network** for candidate generation
* **LightGBM** for ranking
* Training efficiency with last 6 months
* Preservation of seasonality and long-term patterns

I would use this design:

## Two-tower

```text
Pretrain:
  data = all history before cutoff, up to 1-3 years
  sample weighting = exponential decay
  goal = learn broad customer/article representation

Fine-tune:
  data = last 6 months
  sample weighting = stronger recent weighting
  goal = adapt to current fashion trends

Candidate generation:
  retrieve top 500 or top 1000 candidates per customer
```

## LightGBM

```text
Training rows:
  candidates generated from folds inside last 6 months

Labels:
  purchase in next 7 days

Features:
  7d, 30d, 90d, 180d, 365d, same-season-last-year aggregates

Objective:
  LambdaRank or binary classifier baseline

Evaluation:
  MAP@12 and candidate Recall@K on rolling temporal folds
```

## Feature windows

Use this feature window plan:

|                        Window | Use                                      |
| ----------------------------: | ---------------------------------------- |
|                        7 days | hot trend                                |
|                       14 days | very recent intent                       |
|                       30 days | current demand                           |
|                       90 days | season-level behavior                    |
|                      180 days | training window / medium-term preference |
|                      365 days | yearly seasonality and stable taste      |
| same week/month previous year | explicit seasonality                     |

---

# Important warning

Do not compute long-history features using future data.

For every fold:

```text
cutoff = 2020-09-15

Allowed:
  transactions where t_dat <= 2020-09-15

Not allowed:
  transactions where t_dat > 2020-09-15
```

Even if a feature is called `item_pop_365d`, it must mean:

```text
365 days before the cutoff date
```

not 365 days from the full dataset.

---

# Practical recommendation

Do **not** choose between:

```text
Use only 6 months
```

and

```text
Use all years
```

Use both, but for different purposes:

| Data age             | Use it for                                                    |
| -------------------- | ------------------------------------------------------------- |
| Last 7-30 days       | trend, recency, recent intent                                 |
| Last 6 months        | main supervised training examples                             |
| Last 1 year          | long-term user/item features                                  |
| Previous same season | seasonality features                                          |
| 1-3 years            | two-tower pretraining, category priors, stable taste profiles |

That gives you a smaller, faster training pipeline while still preserving long-term trend, seasonality, and customer preference signals.
