---
⚠️ **REFERENCE PROJECT DISCLAIMER** ⚠️

**THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION**

- **DO NOT USE** unless explicitly asked to reference old code
- **CURRENT IMPLEMENTATION** is in `system-design/` directory
- This file is for **REFERENCE ONLY** to understand legacy approaches
- All new development should follow current system design specifications

---

In this project, **ranking** means the second stage of recommendation: after the system has found a smaller set of candidate articles, it predicts which of those candidates are most likely to be relevant to the customer and orders them from best to worst.

The flow is roughly:

```text
Full article catalog
   ↓
Retrieval model / vector search
   ↓
Top candidate articles, e.g. 100 items
   ↓
Ranking model
   ↓
Final ordered recommendations
```

So ranking is not the original H&M dataset itself. It is a **modeling task** built on top of the dataset.

## What Ranking Predicts

Here, ranking is framed as a **binary purchase-likelihood problem**:

- `label = 1`: this customer purchased this article
- `label = 0`: this customer did not purchase this article, generated as a negative sample

The ranking model learns patterns like:

> Given a customer’s age, time/season, and product attributes, how likely is this customer-item pair to be a purchase?

Then at inference time, the model scores candidate articles and sorts them by predicted probability.

## How The Ranking Dataset Is Created

The source H&M dataset has transactions, not explicit ranking labels. So the project creates ranking labels synthetically.

The relevant doc says the original inputs are the H&M tables:

```19:31:docs/project-info/cur-project/model-imp/feature-pipeline-analysis.md
The pipeline processes three main datasets from the H&M fashion dataset:

1. **Articles Dataset** (105,542 records, 25 columns)
   - Product catalog with hierarchical classification
   - Contains product metadata, descriptions, and categorical attributes
```

Then the ranking dataset is assembled separately:

```269:274:docs/project-info/cur-project/model-imp/feature-pipeline-analysis.md
- The ranking dataset is assembled in stages to control peak memory:
  - Read only transaction `article_id` and `customer_id`.
  - Read only customer `age`.
  - Read article metadata without embedding/text-heavy columns.
  - Create positive/negative query pairs.
  - Join the compact item feature frame at the end.
```

The key part is the current positive/negative strategy:

1. **Positive examples** come from real purchases in `transactions_train.csv`.
   - If customer `C` bought article `A`, then `(C, A)` becomes a positive training pair.
   - Label: `1`.
   - These rows are the only labels grounded directly in observed H&M behavior.

2. **Negative examples** are generated synthetically. Negatives are not guaranteed to be true non-purchases, as it is randomly chosen.
   - The pipeline starts from the transaction-derived ranking frame, then samples `article_id`, `customer_id`, and `age` columns with replacement.
   - `article_id` is sampled from the unique article IDs present in the transaction-derived frame, not from the full article catalog.
   - `customer_id` is sampled from transaction rows.
   - `age` is sampled separately from customer IDs, so a sampled age is not guaranteed to belong to the sampled customer.
   - These sampled rows are treated as non-purchase examples.
   - Label: `0`.

3. **Negative sampling uses a 10:1 ratio**.
   - For every positive pair, it creates about 10 negative pairs.

```256:256:docs/project-info/cur-project/model-imp/feature-pipeline-analysis.md
- Ranking dataset negative sampling is vectorized in bulk. The pipeline creates `10x` as many negative pairs as positive pairs by sampling article IDs, customer IDs, and ages with replacement using deterministic seeds, then concatenates the positive and negative frames. This avoids generating negatives through a nested customer-item loop.
```

The final distribution confirms that:

```305:309:docs/project-info/cur-project/model-imp/feature-pipeline-analysis.md
5. **Ranking** (224,136 records) - Training data for ranking model

### Final Data Distribution
- **Interaction Scores**: 0 (73,710), 1 (38,304), 2 (23,799)
- **Ranking Labels**: 0 (203,760), 1 (20,376) - 10:1 negative to positive ratio
```

This means the current ranking data does not use the synthetic `interactions` feature group as the source of negative labels. The `interactions` data has ignore/click/purchase-style scores, but the ranking dataset is built separately from `transactions`, `articles`, and `customers`.

## Current Strategy In Plain Language

The current strategy is:

1. Treat every observed purchase as a positive customer-item example.
2. Create many random customer-item examples and label them as negatives.
3. Join article metadata to both positive and negative rows.
4. Train the ranking model to distinguish observed purchases from random sampled pairs.

For example, if the source transactions contain:

```text
customer_id  article_id  meaning
C1           A1          C1 purchased A1
C1           A2          C1 purchased A2
C2           A3          C2 purchased A3
```

The positive ranking rows become:

```text
customer_id  article_id  label
C1           A1          1
C1           A2          1
C2           A3          1
```

The negative rows are then randomly assembled from available customers, purchased article IDs, and ages:

```text
customer_id  article_id  label
C2           A1          0
C1           A3          0
C2           A2          0
```

These negatives mean "treat this random customer-article pair as not purchased." They do not mean the user viewed, ignored, disliked, or was shown the article.

## Flaws In The Synthetic Ranking Strategy

The current strategy is simple and fast, but it has several important limitations compared with real recommendation data:

1. **Negatives are not guaranteed to be true non-purchases.** The implementation does not explicitly remove known positive `(customer_id, article_id)` pairs from the sampled negatives. A purchased pair can therefore be sampled again and labeled `0`, creating contradictory labels.

2. **Customer age can become inconsistent.** Because `age` is sampled separately from `customer_id`, a negative row can combine one customer's ID with another customer's age. In real data, customer attributes should remain attached to the customer.

3. **Duplicate negative rows can appear.** Sampling with replacement allows the same negative customer-item pair to appear multiple times. Exact duplicates mostly act as accidental extra training weight.

4. **Negative labels are not based on exposure.** In real life, a strong negative usually means the customer saw the item and did not click or buy it. Here, a negative only means a random pair was generated. The customer may never have seen the item.

5. **Random negatives can be too easy.** A random item may be obviously irrelevant to the customer. The model can learn broad category shortcuts instead of learning how to choose between realistic candidate items.

6. **The article pool is transaction-derived.** Since negative article IDs are sampled from articles present in the transaction-derived frame, the negatives do not fully represent the complete catalog or cold-start articles.

7. **It optimizes purchase classification more than ranking quality.** The model is trained to separate purchases from random non-purchases, but real ranking needs to order plausible candidate items for the same customer.

## Suggestions To Improve The Synthetic Ranking Strategy

A better strategy would keep the same basic positive/negative idea but make the negative examples more realistic:

1. **Exclude known positives from negatives.** For each customer, sample negative articles only from articles that customer did not purchase in the training window.

2. **Keep customer features attached to the customer.** Sample `customer_id` first, then join the customer's real features such as `age`, `club_member_status`, and `age_group`.

3. **Deduplicate negative pairs.** Ensure each `(customer_id, article_id)` negative pair appears once unless there is a deliberate weighting strategy.

4. **Sample negatives per customer.** For each customer, create several negative articles around their positive purchases. This better matches the ranking task, where many candidate articles are compared for one customer.

5. **Use harder negatives.** Instead of purely random articles, include articles that are plausible but not purchased, such as popular items, same-category items, same-color items, items retrieved by the two-tower model, or items bought by similar customers.

6. **Use exposure or interaction data if available.** If the system has impressions, clicks, ignores, wishlist events, or product-page views, use those signals to distinguish "shown but not purchased" from "never seen."

7. **Make labels richer when possible.** A graded label can express preference strength better than binary purchase/non-purchase. For example: purchase > add-to-cart > click > impression without click > random unknown.

8. **Evaluate with ranking metrics.** Use metrics such as Recall@K, NDCG@K, MAP@K, or Precision@K in addition to binary classification metrics. These better reflect whether the model orders recommendations well.

## Features Used For Ranking

The ranking model uses customer, product, and time features.

From the training analysis:

```110:124:docs/project-info/cur-project/model-imp/training-pipeline-analysis.md
**Input Features Used:**
- `age`: Customer age at time of purchase
- `product_type_name`: Specific product type classification
- `product_group_name`: Higher-level product grouping
- `graphical_appearance_name`: Visual pattern/design features
- `colour_group_name`: Color classification
- `perceived_colour_value_name`: Perceived color intensity
- `perceived_colour_master_name`: Primary color classification
- `department_name`: Retail department classification
- `index_name`: Internal product classification
- `index_group_name`: Gender/category grouping
- `section_name`: Store section classification  
- `garment_group_name`: Garment type classification
- `month_sin`, `month_cos`: Temporal purchase patterns
```

So the ranking model is not using raw image pixels or article text embeddings here. It is mostly using structured metadata:

- customer age
- product category
- color
- garment type
- department
- seasonality from month sine/cosine

The doc also notes that heavy fields are excluded:

```266:268:docs/project-info/cur-project/model-imp/feature-pipeline-analysis.md
  - `article_description` is needed for embeddings but not for ranking joins.
  - `embeddings` are needed for retrieval/item representation but not for classical ranking feature joins.
  - `image_url` is useful for UI display but unnecessary during ranking dataset construction.
```

## Model Used For Ranking

The traditional ranking model here is a **CatBoostClassifier**.

```96:107:docs/project-info/cur-project/model-imp/training-pipeline-analysis.md
## CatBoost Ranking Model

### Model Configuration
```python
CatBoostClassifier(
    learning_rate=0.2,
    iterations=100, 
    depth=10,
    scale_pos_weight=10,
    early_stopping_rounds=5,
    use_best_model=True
)
```
```

Even though the docs call it “ranking,” the implementation described here is essentially **classification-based ranking**:

1. Train a classifier to predict purchase probability.
2. At serving time, score each candidate article.
3. Sort candidates by the predicted positive-class probability.

So the model output is not “rank 1, rank 2, rank 3” directly. It outputs scores, and those scores are used to rank items.

## How Ranking Works During Inference

At inference time:

1. The query/retrieval model creates a user/query embedding.
2. Vector search retrieves candidate articles, for example top 100.
3. The ranking model computes features for those candidate pairs.
4. CatBoost predicts a probability for each candidate.
5. The candidates are sorted by probability.

The inference doc describes this separation:

```9:12:docs/project-info/cur-project/model-imp/inference-pipeline-analysis.md
The inference pipeline consists of two main components:
1. **Query Model Pipeline**: Handles user query processing and candidate retrieval via vector similarity search
2. **Ranking Model Pipeline**: Scores and ranks candidates using either CatBoost or LLM-based models
```

And the ranking output looks like score/article pairs:

```126:134:docs/project-info/cur-project/model-imp/inference-pipeline-analysis.md
{
  "predictions": {
    "ranking": [
      [0.9234, "592846001"],
      [0.8765, "536139006"],
      [0.8123, "408554004"]
    ]
  }
}
```

Meaning:

```text
article 592846001 → score 0.9234 → rank 1
article 536139006 → score 0.8765 → rank 2
article 408554004 → score 0.8123 → rank 3
```

## Important Caveat

The ranking labels are partly synthetic because the negatives are sampled rows treated as non-purchases. That does **not always mean the customer disliked the item**. It only means the purchase was not observed or the random pair was generated as a negative training example.

In recommender systems, this is common, but it introduces some noise:

- A customer may have liked an item but never saw it.
- A customer may have wanted it but did not buy it.
- A sampled negative could actually be a plausible recommendation.
- In the current implementation, a sampled negative could even overlap with a known purchase if positives are not filtered out.

So the ranking model is learning from **observed purchase vs sampled non-purchase**, not true explicit preference ratings.
---
⚠️ **END OF REFERENCE PROJECT FILE** ⚠️

Remember: This is archived code. Use `system-design/` for current implementation.

---
