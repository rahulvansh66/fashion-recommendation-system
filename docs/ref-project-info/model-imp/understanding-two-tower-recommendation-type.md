---
⚠️ **REFERENCE PROJECT DISCLAIMER** ⚠️

**THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION**

- **DO NOT USE** unless explicitly asked to reference old code
- **CURRENT IMPLEMENTATION** is in `system-design/` directory
- This file is for **REFERENCE ONLY** to understand legacy approaches
- All new development should follow current system design specifications

---

# Understanding What Type Of Recommender The Two-Tower System Is

This note explains the recommendation style used by the current two-tower system. The goal is to answer a few closely related questions:

- Is the two-tower model collaborative filtering?
- Why does it use `customer_id` and `article_id` when tree models usually should not?
- Is the full system content-based because it has a ranking model?
- How do in-batch negatives work, and are transactions outside the batch ignored?

The short answer: the current system is a **hybrid recommender**, but the two-tower part itself is mostly **collaborative filtering**. It learns from purchase behavior: which customers bought which articles. The ranking model then adds structured product metadata and time context to refine the retrieved candidates.

---

## 1. The Two-Tower Model Is Mostly Collaborative Filtering

Collaborative filtering means:

> Recommend items by learning from user-item interaction patterns.

In this project, the interaction is a transaction row:

```text
customer C bought article A on date D
```

The model is not first trying to understand fashion text like "ribbed cotton top" or "oversized black hoodie". It is mainly learning from behavior:

```text
Alice bought: beanie, scarf, wool coat
Bob bought: beanie, scarf, gloves
Carol bought: sundress, sandals, linen shirt
```

From this pattern, the model can learn that Alice and Bob have similar taste, and that beanies, scarves, and gloves live in a similar behavior region. This is collaborative filtering: "people who behaved similarly may like similar items."

In the current code, the two towers look like this conceptually:

```text
QueryTower(customer_id, age, month_sin, month_cos)
    -> 16-dim customer/query vector

ItemTower(article_id, garment_group_name, index_group_name)
    -> 16-dim article/item vector
```

Both outputs live in the same vector space. Training tries to place a customer vector near articles the customer actually bought.

So the core collaborative filtering signal is:

```text
customer_id <-> article_id
```

The additional features add useful context:

- `age` gives a small demographic signal.
- `month_sin` and `month_cos` let recommendations shift by season.
- `garment_group_name` and `index_group_name` give coarse item category context.

But the main personalization still comes from learned customer and article embeddings shaped by purchase rows.

---

## 2. Why Passing `article_id` Makes Sense Here

Your concern about IDs is correct for many tabular models.

In a model like XGBoost, a high-cardinality ID can be dangerous. If we pass `article_id` directly into a tree model, the model may learn brittle rules like:

```text
if article_id in {A123, A987, A555}, predict high probability
```

That can become memorization. It often performs well on known IDs but does not generalize well.

The two-tower model uses `article_id` differently. It does not use the ID as a tree split. It uses the ID as a lookup key into a trainable embedding table:

```text
article_id -> StringLookup -> Embedding layer -> dense vector
```

So `article_id = A17` does not become a raw number with numeric meaning. It becomes a learned vector:

```text
A17 -> [0.12, -0.44, 0.08, ...]
```

That vector is updated during training. Every time article A17 appears in a real purchase row, the loss nudges its vector toward the customer who bought it.

This is similar to classic matrix factorization:

```text
learn one vector per customer
learn one vector per item
place purchased customer-item pairs close together
```

So yes, the model does memorize article identity in one sense. But that is not a bug here. It is the core mechanism that lets the retriever distinguish one known article from another.

If the item tower did not use `article_id`, and only used:

```text
garment_group_name
index_group_name
```

then many articles would look almost identical. Two different dresses in the same garment group and index group would produce very similar item vectors. The model would lose the ability to learn that one specific dress is popular with one customer group while another similar-looking dress is popular with another group.

The tradeoff is:

- Strong for known articles with purchase history.
- Weak for brand-new articles with no training interactions.

That weakness is called the **item cold-start problem**.

---

## 3. Why `detail_desc` Is Not Used By The Two-Tower Item Tower

The current item tower consumes:

```text
article_id
garment_group_name
index_group_name
```

It does not consume `detail_desc` directly.

That is intentional in the current architecture. The two-tower item embedding is behavior-based. It learns what an article means from who buys it, not from the words in its product description.

The project does use `detail_desc`, but in a different embedding:

```text
detail_desc + product metadata
    -> article_description
    -> SentenceTransformer
    -> 384-dim semantic article embedding
```

That semantic embedding describes the article's text. It is useful for content search or LLM-style flows, but it is separate from the 16-dim two-tower candidate embedding used by the main retrieval flow.

Adding raw `detail_desc` directly to the two-tower model would not be ideal because it is long free text. A simple `StringLookup` over whole descriptions would mostly memorize text values rather than understand them.

A better experiment would be:

```text
ItemTower input =
    article_id embedding
    + garment/index features
    + projected SentenceTransformer article embedding
```

That would make the two-tower retriever more hybrid: still collaborative, but with content awareness. It may help with sparse items and cold-start articles.

---

## 4. Is The Full System Content-Based Because It Uses Ranking?

Partly, yes.

The full recommendation pipeline has two stages:

```text
Stage 1: Two-tower retrieval
Stage 2: CatBoost ranking
```

The two-tower retrieval stage is mostly collaborative filtering:

```text
"Which articles are close to this customer based on purchase behavior?"
```

The CatBoost ranking stage is more content/context-aware. It scores the retrieved articles using structured features such as:

- product type
- product group
- color
- department
- index group
- section
- garment group
- customer age
- month seasonality

So the full system is best described as:

```text
Hybrid recommender =
    collaborative retrieval
    + content/context-aware ranking
```

It is not pure content-based filtering, because the candidate set comes from collaborative two-tower retrieval. It is also not pure collaborative filtering, because the final ordering uses item metadata and temporal/customer context.

The clean mental model is:

```text
Two-tower model:
    "Find behaviorally relevant candidates quickly."

Ranking model:
    "Among those candidates, reorder using richer item and context features."
```

### How CatBoost Ranking Is Personalized Per User

CatBoost ranking is personalized in two ways, but one is much stronger than the other.

The strongest personalization happens **before CatBoost**, in the two-tower retrieval step. The `QueryTower` creates a query embedding from the customer-specific inputs:

```text
customer_id
age
month_sin
month_cos
```

That query embedding is used to search the `candidate_embeddings` vector index and retrieve the top candidate articles. So customer A and customer B usually enter CatBoost with different candidate sets.

That means CatBoost is not starting from the full catalog. It is starting from:

```text
100 items already selected for this specific customer
```

Then CatBoost applies a second layer of personalization through the features it directly receives. In this project, those direct user/context features are mostly:

```text
age
month_sin
month_cos
```

So CatBoost can learn patterns like:

```text
age=25 + winter + knitwear -> higher score
age=60 + summer + light dresses -> higher score
December + partywear -> higher score
```

But CatBoost does **not** have the same deep per-user memory as the two-tower model. It is not remembering every customer's purchase history by itself. It is using a personalized candidate set from two-tower retrieval, then reordering that set using age, season, and product metadata.

The accurate summary is:

```text
CatBoost personalization =
    indirect strong personalization from two-tower candidates
    + direct light personalization from age/month features
```

---

## 5. What Training Rows Mean In The Two-Tower Model

The two-tower training data comes from transactions. Each row is a real purchase:

```text
customer_id="C42"
age=34
month_sin=-0.5
month_cos=-0.866
article_id="A17"
garment_group_name="Knitwear"
index_group_name="Ladieswear"
```

There is no explicit label like:

```text
label = 1
```

The row itself is the positive signal. Its existence means:

```text
Customer C42 bought article A17.
```

The model learns by turning both sides into vectors:

```text
QueryTower(C42, age, month) -> u_C42
ItemTower(A17, categories)  -> v_A17
```

Then the retrieval loss tries to make:

```text
u_C42 dot v_A17
```

large compared to scores for other articles.

---

## 6. The Batch Question: Are Other Transactions Ignored?

This is the subtle part.

During training, the model does not process the full dataset in one giant step. It processes a **batch** at a time. In this project, the configured two-tower batch size is `2048`.

The code builds the TensorFlow dataset like this conceptually:

```text
training DataFrame
    -> one TensorFlow example per row
    -> group rows into batches of 2048
    -> train for multiple epochs
```

So if there are 200,000 training transactions and batch size is 2048, one epoch has roughly:

```text
200,000 / 2048 ~= 98 batches
```

Each batch is just a chunk of training rows. In the current code, the data is batched first, then cached, then shuffled:

```text
df_to_ds(train_df)
    -> batch(2048)
    -> cache()
    -> shuffle(...)
```

That means the rows inside a batch are mostly determined by the order of `train_df` before batching. The shuffle step changes the order in which batches are processed, but it does not necessarily reshuffle individual rows into new batch groups. This is still valid training, but it means the in-batch negatives for a given row may be relatively stable across epochs in this implementation.

Across training, all training rows are still visited.

Transactions outside the current batch are not used in **that one gradient update**, but they are not ignored. They appear in other batches, and the model trains for multiple epochs. Over time, every training transaction gets many chances to update the embeddings.

Think of it like studying a large book:

```text
You do not read all pages at the exact same second.
You read one page group at a time.
Pages outside the current group are not ignored.
They are just read in later groups.
```

Training works the same way.

---

## 7. Why Use In-Batch Negatives?

The training table only contains purchases. It does not contain explicit "customer did not buy article" rows.

For a retrieval model, we still need contrast. The model must learn not only:

```text
Alice should be close to the beanie she bought.
```

but also:

```text
Alice should score that beanie higher than random other articles.
```

In-batch negatives solve this cheaply.

Suppose a tiny batch has four purchase rows:

```text
Row 1: Alice bought beanie
Row 2: Alice bought scarf
Row 3: Bob bought beanie
Row 4: Carol bought sundress
```

The model computes query vectors:

```text
u_Alice
u_Alice
u_Bob
u_Carol
```

And item vectors:

```text
v_beanie
v_scarf
v_beanie
v_sundress
```

For Row 1, the positive pair is:

```text
Alice -> beanie
```

The other articles in the same batch act as negatives:

```text
Alice -> scarf
Alice -> sundress
```

But wait: Alice also bought scarf in Row 2. Does that mean scarf is incorrectly treated as a negative for Alice in Row 1?

In this tiny example, yes, that can happen. This is a known approximation with in-batch negatives. It is called a **false negative**: an item treated as negative in one batch context even though it is actually positive elsewhere.

This does not usually break training because:

1. The positive signal for `Alice -> scarf` appears in Row 2.
2. Training uses many batches and many epochs.
3. Gradient updates are small, so one false-negative push does not dominate all positive pulls.
4. Across the whole dataset, the model minimizes total loss over all rows, not just one row.
5. With a large batch size like 2048, each row still gets many comparison candidates in every update.

So the final geometry becomes a compromise that satisfies all repeated signals:

```text
Alice is close to beanie.
Alice is close to scarf.
Bob is close to beanie.
Carol is close to sundress.
Carol is far from winter accessories.
```

The model is not saying "Alice must dislike every other article in this batch forever." It is saying:

```text
For this training step, make the row's purchased article score higher
than the other available candidate articles in this batch.
```

Then future steps correct and refine that geometry using other rows.

---

## 8. A Bigger Batch Example

Now imagine a batch size of 2048.

Each row contributes:

- 1 positive article: the article actually bought in that row.
- Up to 2047 in-batch negatives: the other articles in the same batch.

So one batch gives a lot of comparison signal:

```text
2048 positive pairs
millions of possible customer-vs-other-article comparisons
```

That is why in-batch negatives are popular in retrieval systems. They avoid having to manually build a huge negative dataset.

Without in-batch negatives, we might need to generate rows like:

```text
Alice did not buy article A.
Alice did not buy article B.
Alice did not buy article C.
...
```

That explodes quickly. With 1 million customers and 100,000 articles, the full "did not buy" matrix is enormous. In-batch negatives give a practical approximation.

---

## 9. What Happens Across Epochs?

An epoch means one pass over the training data.

In this project, the two-tower model is configured for multiple epochs. If the training data has 200,000 rows and the model trains for 10 epochs, then the model sees about:

```text
200,000 purchase rows x 10 epochs = 2,000,000 row presentations
```

This does not mean 2,000,000 unique purchases. It means the same training purchases influence the model repeatedly.

Each time a row appears:

1. The query tower produces a customer vector.
2. The item tower produces an article vector.
3. The retrieval loss compares the positive article against current in-batch negatives.
4. Backpropagation nudges the model weights.

Across epochs, the same customer and article embeddings are repeatedly adjusted. The final vector positions are the accumulated result of all these small nudges.

So transactions outside the current batch are only "outside" temporarily. They still affect the model when their batch is processed.

---

## 10. Why This Still Makes Sense

At first, in-batch negatives can feel suspicious because the model compares only against articles in the current batch, not every article in the catalog.

But full-catalog comparison is often too expensive. If the catalog has 100,000 articles, comparing every customer in every batch against every article would be much slower.

The in-batch approach says:

```text
Use a manageable subset of other articles per step.
Repeat this over many batches and epochs.
The model can learn from many local comparisons instead of every possible comparison.
```

This is similar to how stochastic gradient descent works in general. Each batch is an imperfect sample of the full problem. No single batch tells the whole truth, but many batches together give a strong enough training signal.

One implementation note: if we wanted each row to see a more varied set of negatives across epochs, we would normally shuffle individual rows before batching. The current code batches before shuffling, so the training still works, but the negative set per row is less varied than it could be.

The model does not need every possible negative comparison in every step. It needs enough varied comparisons over time to learn useful geometry.

---

## 11. Does CatBoost Have Separate Weights Per Customer?

No. CatBoost has **one shared model**.

It does not learn a private vector or private set of tree weights for each customer. Customer `C42`, customer `C99`, and every other customer all pass through the same trained CatBoost trees.

What changes is the input row.

For example, the same CatBoost model may score:

```text
Customer A:
age=25, month=winter, article=beanie, garment_group=Accessories

Customer B:
age=60, month=summer, article=beanie, garment_group=Accessories
```

The model is the same, but the feature values are different. So the predicted score can be different.

In the current project, CatBoost's direct user signal is mostly:

```text
age
month_sin
month_cos
```

It also receives the article's structured metadata, such as product type, product group, color, department, section, and garment group. It learns shared rules like:

```text
younger customers + winter + knitwear -> higher probability
older customers + summer + light dresses -> higher probability
certain product groups work better in certain months
```

It does **not** deeply remember:

```text
Alice specifically likes scarves.
Bob specifically avoids sandals.
Carol specifically buys floral dresses.
```

That per-customer memory is the two-tower model's job. The `QueryTower` has a learned `customer_id` embedding, so it can retrieve a candidate set tailored to a specific customer. CatBoost then applies the same global ranking logic to those already-personalized candidates.

The responsibility split is:

```text
Two-tower retrieval:
    "For this exact customer, retrieve likely items."

CatBoost ranking:
    "Given these candidate items, the customer's age, and the season,
    which items should be ordered highest?"
```

This makes sense because a production recommender usually separates **personalized candidate generation** from **shared ranking logic**. The retriever handles the high-dimensional user taste memory. The ranker handles structured product/context scoring and final ordering.

If we wanted CatBoost itself to become more personalized, we could add richer user features:

- `customer_id` as a categorical feature
- recent purchased categories
- favorite garment groups
- average spend
- interaction counts
- customer segment
- two-tower query embedding dimensions

But in the current project, CatBoost is only lightly personalized directly. Its stronger personalization comes indirectly from the two-tower candidate set it receives.

---

## 12. Final Classification Of This Project's Recommender

The best label for the current architecture is:

```text
Hybrid two-stage recommender
```

More specifically:

```text
Stage 1: Two-tower retrieval
Type: mostly collaborative filtering
Signal: customer-article purchases
Main IDs: customer_id and article_id embeddings
Output: top candidate articles from vector search

Stage 2: CatBoost ranking
Type: content/context-aware ranking
Signal: purchase labels plus item metadata, customer age, month
Output: final ordered recommendation list
```

So if someone asks, "What kind of recommendation system is this?", the most accurate answer is:

> It is a hybrid recommender. The candidate generation stage is a two-tower collaborative filtering retriever, and the ranking stage adds content and context features to improve the final order.

That one sentence captures the architecture well.


---
⚠️ **END OF REFERENCE PROJECT FILE** ⚠️

Remember: This is archived code. Use `system-design/` for current implementation.

---
