---
⚠️ **REFERENCE PROJECT DISCLAIMER** ⚠️

**THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION**

- **DO NOT USE** unless explicitly asked to reference old code
- **CURRENT IMPLEMENTATION** is in `system-design/` directory
- This file is for **REFERENCE ONLY** to understand legacy approaches
- All new development should follow current system design specifications

---

# System Walkthrough: How The Fashion Recommendation System Actually Works

This document explains the full system end-to-end, in the exact order things happen. It clears up the most common point of confusion in this project: there are **two completely different "embeddings"** being computed, and they are used by different parts of the system. Once that is clear, the rest of the pipeline makes sense.

The reader should walk away knowing:

1. What the two-tower model does, and what it does not do.
2. What the ranking model does, and what it does not do.
3. What `article_description` embeddings are, where they live, and what actually uses them.
4. The exact sequence of everything that runs offline (before any user shows up).
5. The exact sequence of everything that runs online (when a user requests recommendations).
6. How the three models interact end-to-end during a real request.

---

## 1. The Most Important Idea: There Are Two Different "Embeddings"

This is the single biggest source of confusion. The project produces two embeddings that share the same name "embeddings" but mean very different things and are produced by different models.

### Embedding A: Semantic Article Embedding (Text-based)

- Produced by a **SentenceTransformer** (`all-MiniLM-L6-v2`), which is a pre-trained NLP model.
- Input: the synthetic `article_description` string built from product name, type, group, color, category, and `detail_desc`.
- Output: a **384-dimensional vector** describing the article's textual/semantic meaning.
- Stored in the `articles` feature group, in a column literally called `embeddings`.
- This embedding has nothing to do with users. It only describes the article's text.
- This is computed once during feature engineering and is the only place `detail_desc` is used.

### Embedding B: Two-Tower Item Embedding (Behavior-based)

- Produced by the **trained ItemTower** of the two-tower model.
- Input: `article_id`, `garment_group_name`, `index_group_name`.
- Output: a **16-dimensional vector** in the same space as the user/query embedding.
- Stored in a separate feature group called `candidate_embeddings`, in a column also literally called `embeddings`.
- This is computed once after the two-tower model is trained.

### How Embedding B Actually Learns "Who Buys What"

This is the part that often feels like magic. The ItemTower is just a small neural network; it does not "know" anything about customers. So how do its outputs end up reflecting buying behavior? It is entirely because of **how the model is trained, not what the ItemTower looks at**.

The training data is a list of real purchases from `transactions_train.csv`. Each purchase is one row: customer `C` bought article `A` on date `D`. Nothing else. There are no explicit labels like "good match: 0.9".

During training, the system does the following for each row (and many rows in parallel in a batch):

1. The **QueryTower** turns the customer side of the row (`customer_id`, `age`, `month_sin`, `month_cos`) into a 16-dim vector. Call it `u`.
2. The **ItemTower** turns the article side of the row (`article_id`, `garment_group_name`, `index_group_name`) into a 16-dim vector. Call it `v`.
3. The model computes the dot product `u · v`. This is one number, treated as a "compatibility score".
4. The loss function (`tfrs.tasks.Retrieval`, which is essentially contrastive softmax over the batch) does two things at once:
  - **Pull together:** Increase `u · v` for the real (customer, purchased article) pair. The customer's vector and the bought article's vector should become more aligned.
  - **Push apart:** Decrease `u · v'` where `v'` is the embedding of every *other* article in the same batch. Those articles are treated as "negatives" — articles this customer did not buy in this batch.
5. Backpropagation updates the weights of both towers so that next time, similar customers and similar articles will land closer in vector space, and dissimilar ones farther apart.

#### Aside: What Are `month_sin` and `month_cos`?

These two numbers are a **cyclical encoding of the month** of the transaction date — the standard way to tell a machine learning model "what time of year is it" without distortion.

**The problem with a raw month number.** If you encode month as a plain integer (Jan = 1, Feb = 2, ..., Dec = 12), the model thinks December and January are 11 units apart, when in reality they are just one month apart. Fashion is highly seasonal — a Christmas Eve buyer and a New Year's Day buyer behave almost identically — so this distortion would mislead the model.

**The fix.** Place the 12 months around a clock face, and represent each month by its (x, y) coordinates on that circle:

- `month_sin = sin(month * 2π / 12)`
- `month_cos = cos(month * 2π / 12)`

This wraps the year into a closed loop. December ends up right next to January on the circle. June ends up on the opposite side.


| Month         | `month_sin` | `month_cos` |
| ------------- | ----------- | ----------- |
| January (1)   | 0.50        | 0.87        |
| March (3)     | 1.00        | 0.00        |
| June (6)      | 0.00        | -1.00       |
| September (9) | -1.00       | 0.00        |
| December (12) | 0.00        | 1.00        |


The (sin, cos) coordinates for December and January are very close (Euclidean distance ≈ 0.52), while December and June are far apart (distance = 2.0). The model can now learn "winter buyers behave alike" automatically, because winter months are geometrically close in this 2-D plane.

**Why two numbers (sin AND cos)?** A single sine value is ambiguous: `sin(30°) = sin(150°)`, so January (angle 30°) and May (angle 150°) would share the same value and look identical to the model. Adding the cosine resolves the ambiguity, because `cos(30°) ≈ 0.87` while `cos(150°) ≈ -0.87`. Together, the pair uniquely identifies a point on the unit circle.

**Where the project uses them.**

- **QueryTower input.** Lets the customer's query vector shift seasonally — "Alice in summer" can land in a slightly different spot than "Alice in winter", so the candidates retrieved for her also shift with the season.
- **CatBoost ranking features.** Combined feature importance ≈ 93% in the trained model — fashion purchases in this dataset are dominated by seasonal patterns.
- **Computed live at request time** from the request's `transaction_date` (see `query_transformer.py`), so the model always uses the actual current month.

The values stay in [-1, 1], so they are already roughly on the same scale as the normalized age — no extra normalization needed.

#### A Worked Toy Example

Take a tiny world with three customers and three articles. Pretend everyone is shopping in November (so `month_sin` and `month_cos` are some fixed numbers).

- **Alice**, age 25. Bought the **beanie** and the **scarf**.
- **Bob**, age 28. Bought the **beanie**.
- **Carol**, age 60. Bought the **sundress**.

That gives us 4 training rows. Each row is just `(customer, article)`:

- Row 1: (Alice, beanie)
- Row 2: (Alice, scarf)
- Row 3: (Bob, beanie)
- Row 4: (Carol, sundress)

Before training, every weight in the model is random. Imagine the (simplified) 2-D vectors start at random positions:

- `u_alice = [0.1, 0.2]`, `u_bob = [-0.3, 0.4]`, `u_carol = [0.5, -0.1]`
- `v_beanie = [-0.2, 0.1]`, `v_scarf = [0.3, -0.4]`, `v_sundress = [0.0, 0.5]`

**Processing the batch (all 4 rows at once).** In practice the model never processes a single row in isolation. Here is what happens step by step before the score matrix exists.

**Step 1 — split the batch into two columns and feed both towers simultaneously.**

```
BATCH INPUT
───────────────────────────────────────────────────────
Row 1: (Alice, beanie)
Row 2: (Alice, scarf)
Row 3: (Bob,   beanie)
Row 4: (Carol, sundress)
```

```
CustomerTower                    ArticleTower
─────────────────                ─────────────────
Input: [alice_id, age=25, ...]   Input: [beanie_id,   color, ...]
Input: [alice_id, age=25, ...]   Input: [scarf_id,    color, ...]
Input: [bob_id,   age=28, ...]   Input: [beanie_id,   color, ...]
Input: [carol_id, age=60, ...]   Input: [sundress_id, color, ...]
```

**Step 2 — each tower produces one vector per row.**

```
CustomerTower output          ArticleTower output
──────────────────────        ──────────────────────
u_alice = [ 0.1,  0.2]       v_beanie   = [-0.2,  0.1]
u_alice = [ 0.1,  0.2]       v_scarf    = [ 0.3, -0.4]
u_bob   = [-0.3,  0.4]       v_beanie   = [-0.2,  0.1]
u_carol = [ 0.5, -0.1]       v_sundress = [ 0.0,  0.5]
```

`u_alice` appears twice because she has two purchases — the same weights run on the same input both times, so the vector is identical.

**Step 3 — deduplicate, then compute all dot products in one matrix multiply (`U · Vᵀ`).**

Unique customers: `u_alice`, `u_bob`, `u_carol` (3 vectors)  
Unique articles: `v_beanie`, `v_scarf`, `v_sundress` (3 vectors)

That gives a 3×3 score matrix of every customer-article dot product in the batch:


|             | v_beanie   | v_scarf | v_sundress |
| ----------- | ---------- | ------- | ---------- |
| **u_alice** | **0** ✓    | −0.06 ✓ | 0.10       |
| **u_bob**   | **0.10** ✓ | −0.25   | 0.02       |
| **u_carol** | −0.10      | 0.19    | **0.05** ✓ |


✓ marks the actual purchase (the positive pair) for each row. Every other entry in the same row is an **in-batch negative** — an article the customer did not buy, automatically available at no extra cost because its embedding is already in the batch.

**Row 1 loss (Alice, beanie).** The dot product `u_alice · v_beanie = 0.1·(−0.2) + 0.2·0.1 = 0`. The loss for this row says: "Alice's score for beanie (0) must be higher than her score for scarf (−0.06) and sundress (0.10)." Gradient descent nudges `u_alice` toward `v_beanie`, nudges `v_beanie` toward `u_alice`, and nudges `v_scarf` and `v_sundress` slightly away from `u_alice`.

**Processing Row 2: (Alice, scarf).** Now `u_alice` and `v_scarf` get pulled together; beanie and sundress are the negatives this time. After Rows 1 and 2, `u_alice` has been pulled toward both `v_beanie` and `v_scarf`. Her vector now sits somewhere between them — that "between" point is what we will later call "Alice's taste".

**Processing Row 3: (Bob, beanie).** `u_bob` gets pulled toward `v_beanie`. Now `v_beanie` has been pulled toward both `u_alice` and `u_bob`. The beanie's final position sits in the region of the space where "people who buy beanies" live.

**Processing Row 4: (Carol, sundress).** `u_carol` and `v_sundress` get pulled together; beanie and scarf are pushed away from `u_carol`. After this row, `u_carol` and the winter accessories end up far apart in the space.

After many epochs over the full purchase log, the geometry settles:

- `u_alice` lives near `v_beanie` and `v_scarf` — those are the items she actually bought.
- `u_bob` lives near `v_beanie`.
- `u_carol` lives near `v_sundress`, far from the beanies.
- `v_beanie` lives near both `u_alice` and `u_bob` (the two people who bought it).
- `v_scarf` lives mainly near `u_alice` (only she bought it).
- `v_sundress` lives near `u_carol`.

At serving time, if we encode Alice through the QueryTower, her query vector lands in the "winter accessories" region. A nearest-neighbor vector search returns beanies and scarves. If we encode Carol, her vector lands far from those and near sundresses. The geometry directly reflects who bought what.

If Alice had bought 100 items during training instead of 2, her vector would be the result of being tugged 100 times in different directions. It would represent the **center of mass of Alice's taste** across all those purchases.

**Why the competing forces don't cancel each other out.** A natural concern: in Row 1, `v_scarf` is pushed *away* from `u_alice` (scarf is an in-batch negative). Then in Row 2, `v_scarf` is pulled *toward* `u_alice` (scarf is the positive). Does this just undo itself?

No — and the reason matters. The gradient steps are tiny and the loss landscape is high-dimensional. After many epochs the system converges to the unique geometry where the total loss across *all* rows is minimized simultaneously. The only geometry that satisfies every constraint at once is:

- `u_alice` lands between `v_beanie` and `v_scarf` (she bought both, so she belongs near both).
- `v_beanie` lands near both `u_alice` and `u_bob` but is not identical to either.
- `v_scarf` lands near `u_alice` but farther from `u_bob` (he never bought it).

The "ending up in the middle" is not a bug — it is the correct answer. `u_alice` sitting between `v_beanie` and `v_scarf` is precisely what "Alice likes both winter accessories" should look like in vector space. The center of mass is the intended result.

#### Where Does The Purchase History Actually "Live" In The Model?

This is the part most people miss when they look at the model code. The QueryTower's literal inputs are just `customer_id`, `age`, `month_sin`, `month_cos`. There is no "list of last 10 items bought" feature anywhere. So where is Alice's purchase history?

**Answer: it lives inside Alice's learned `customer_id` embedding.**

The QueryTower contains a Keras `Embedding` layer with one slot per known customer. Conceptually, it is a lookup table:

```
customer_id        → 16-dim vector
"alice_hash..."    → [0.13, -0.42, 0.07, ...]
"bob_hash..."      → [-0.21, 0.18, 0.55, ...]
"carol_hash..."    → [0.66, 0.02, -0.31, ...]
...
```

During training, every time a row featuring Alice flows through the model, exactly four things happen for the `customer_id` part of her input:

1. The `StringLookup` layer finds Alice's row in the table.
2. The `Embedding` layer pulls her current 16-dim vector.
3. The retrieval loss is computed for the whole batch.
4. Gradients flow backward and update **only Alice's slot** in the embedding table (along with all other weights).

So each of Alice's 2 purchases nudged her slot once. If she had 100 purchases, her slot would be nudged 100 times. By the end of training, her slot is the **compressed summary of every purchase Alice ever made**. The model never stored "Alice bought beanie, then scarf, then..."; it just kept moving her single 16-dim vector around until that vector pointed toward the kinds of items she buys.

This is the entire mechanism by which "purchase history" is encoded. It is not a sequence, not a list, not a set. It is **one learned vector per customer**, shaped by all of that customer's training rows.

Two important consequences:

- **No real-time updates.** If Alice buys a new item *after* training finishes, her embedding does not move. The system will keep recommending based on her old vector until the model is retrained. (Some systems handle this by adding a "recent purchases" input feature — this project does not.)
- **No purchase order.** The model knows what Alice bought, but it does not know in what order. Sequence-aware recommenders use a different architecture for this.

#### Why The Output Is Personalized For Each Customer

Two customers with identical age and shopping in the same month will still get **different** query vectors. The reason is mechanical: the customer_id input goes through a per-customer embedding lookup, and every customer's slot holds a different vector.

Walking through what the QueryTower does for one customer:

1. **Look up `customer_id`** → retrieves that customer's personal 16-dim vector. **Different for every customer.**
2. **Look up `age`** (normalized) → one number. Same if two customers have the same age.
3. **Pass through `month_sin`, `month_cos`** → two numbers. Same if two requests are in the same month.
4. **Concatenate** → a vector of length 16 + 1 + 1 + 1 = 19.
5. **Two Dense layers** project it down to a 16-dim query vector.

Step 1 is where personalization is born. Because Alice's slot vector is different from Bob's slot vector, their concatenated 19-dim inputs differ, and therefore their final 16-dim query vectors differ. The vector search returns different candidates for each of them. The age and month components add small modulations (Alice in summer vs. Alice in winter can shift slightly), but the **dominant personalization signal is the customer_id embedding**.

If Alice and Carol both happen to be 25 and shopping in November, their `age` and month features are identical. The only thing that distinguishes their query vectors is the customer_id embedding. That is enough — because their embeddings were shaped by different training rows, the final query vectors point in different directions, and nearest-neighbor search returns different items.

One known limitation: a customer never seen during training (cold start) does not have a slot in the embedding table. The `StringLookup` routes them to a shared "out-of-vocabulary" embedding. So *all* unseen customers share the same fallback vector. If they also share the same age and month, they will get *identical* recommendations. Personalization for them only kicks in once they appear in the training data on the next retraining cycle.

Repeat this for thousands of real purchases and many epochs. The math has only one way to drive the loss down: **make the ItemTower's output for article A reflect the kinds of customers who actually buy A**, and make the QueryTower's output for customer C reflect the kinds of articles C actually buys. The geometry of the 16-dim space is shaped entirely by purchase co-occurrence patterns in the data.

A concrete intuition. Suppose:

- Customers in their 20s, in winter, frequently buy black knit beanies.
- Those same customers also buy black wool scarves.
- Customers in their 60s, in summer, frequently buy floral sundresses.
- Those customers rarely buy beanies.

After training, the beanie's vector and the scarf's vector will end up close to each other (because they get pulled toward similar customer vectors). The 25-year-old's vector and the beanie/scarf vectors will be close. The 65-year-old's vector will be far from the beanies and close to the sundresses. Nobody hand-coded any of this. The contrastive loss did it.

This is why Embedding B is called "behavior-based": its only training signal is **who bought what**. The ItemTower's literal inputs (`article_id`, `garment_group_name`, `index_group_name`) are just an interface that lets the model identify the article. The semantic meaning of the vector comes from the loss function and the purchase data, not from those three inputs.

Compare this with Embedding A: SentenceTransformer outputs are shaped by general English semantics from its pretraining corpus. It has never seen a single H&M transaction. So Embedding A says "this dress's text is similar to that dress's text," while Embedding B says "this dress is bought by people who also buy that dress." Very different things.

### Why It Matters

The main personalized recommendation pipeline uses **Embedding B**. The natural-language "describe what you want" search uses **Embedding A**. They live in different feature groups, have different dimensions, and serve different purposes. Mixing them up is what makes the architecture feel confusing.

A short table to keep them straight:


| Property               | Embedding A                                 | Embedding B                                            |
| ---------------------- | ------------------------------------------- | ------------------------------------------------------ |
| Producer               | SentenceTransformer (`all-MiniLM-L6-v2`)    | Trained `ItemTower` (Keras model)                      |
| Inputs                 | `article_description` text                  | `article_id`, `garment_group_name`, `index_group_name` |
| Dimension              | 384                                         | 16                                                     |
| Feature group          | `articles.embeddings`                       | `candidate_embeddings.embeddings`                      |
| Uses `detail_desc`     | Yes (indirectly, via `article_description`) | No                                                     |
| Used in main rec flow? | No, only for content-based / LLM search     | Yes, this is what vector search retrieves over         |
| Personalized?          | No                                          | Yes                                                    |


Hold onto this distinction. Everything else below depends on it.

---

## 2. The Three Models In One Sentence Each

The system has three independent ML/AI components. Knowing what each one does, alone, makes the integration easier to follow.

1. **Two-Tower Model.** A pair of neural networks that map customers and articles into the same 16-dim space so that "customer is close to article" means "customer is likely to buy article". Used to **retrieve candidates fast**.
2. **CatBoost Ranking Model.** A gradient boosting classifier that takes a small set of candidate articles, looks at customer age, time of year, and structured product attributes, and outputs a **purchase probability** for each pair. Used to **reorder candidates precisely**.
3. **SentenceTransformer.** A frozen pre-trained text model. Converts text into 384-dim vectors. Used to **embed article descriptions and free-text search queries** for content/LLM search. It is not trained in this project; it is just used as a tool.

Two-tower + CatBoost form the **main personalized recommendation pipeline**. SentenceTransformer powers a **secondary search/LLM pipeline**. Both pipelines coexist in the same app.

---

## 3. Why The Two-Tower Approach Exists At All

There are 105,542 articles in the catalog (about 11,820 in the sampled subset used here). Running a heavy ranking model against every single article for every customer is too slow. The standard fix in production recommenders is two stages:

- **Stage 1 - Retrieval (fast, approximate).** Quickly narrow 100k+ items down to ~100. Recall matters more than precision here. A two-tower model is well suited because its candidate embeddings can be precomputed and searched with vector ANN in milliseconds.
- **Stage 2 - Ranking (slow, precise).** Carefully score those ~100 with a stronger model that uses richer features. Precision matters more here.

The two-tower model is purely a Stage 1 retriever. The CatBoost model is purely a Stage 2 ranker. Each is optimized for its own job. Trying to make one model do both is what most beginner systems get wrong.

---

## 4. The Offline Pipeline (What Runs Before Any User Shows Up)

Offline = batch jobs that happen in advance. Their outputs (feature groups, trained models, vector indexes) are what the online system reads from at request time. The offline pipeline must run in a specific order because later stages depend on earlier ones.

Notebooks numbered `1_fp_`*, `2_tp_*`, `3_tp_*`, `4_ip_*`, `5_ip_*`, `7_ip_*` correspond to the stages below.

### Step 1: Raw Data Ingestion

Three CSVs from the H&M dataset are loaded:

- `articles.csv` (~105k product records)
- `customers.csv` (~1.37M customer records)
- `transactions_train.csv` (~31.8M purchase records)

A `DatasetSampler` reduces the customer pool to keep this manageable on a laptop (defaults to 1,000 customers in `SMALL` mode). Transactions are filtered to only those customers.

### Step 2: Feature Engineering (Notebook 1)

Each table gets transformed:

- **Articles.** Build `article_description` text by combining `prod_name`, `product_type_name`, `product_group_name`, color fields, category fields, and `detail_desc`. This text is then encoded with SentenceTransformer into the 384-dim `embeddings` column (**Embedding A**). Add `prod_name_length` and `image_url`. Drop `detail_desc` itself after it has been consumed.
- **Customers.** Add `age_group` bucket. Fill missing `club_member_status`. Drop nulls.
- **Transactions.** Extract `year`, `month`, `day`, `day_of_week`. Compute cyclical encodings `month_sin` and `month_cos` so the model sees December and January as close in time.
- **Interactions.** Synthesize click/ignore/purchase events around real purchases to mimic a browsing session. This produces multi-typed signals (score 0/1/2).
- **Ranking dataset.** Build positive pairs from transactions and negative pairs by sampling article IDs, customer IDs, and ages. The 10:1 negative-to-positive ratio means about 224k ranking rows from ~20k purchases.

All of these are written into Hopsworks as **feature groups**: `articles`, `customers`, `transactions`, `interactions`, `ranking`. The `articles` feature group is registered with an **embedding index** on the `embeddings` column, which lets it be searched by vector similarity later.

### Step 3: Two-Tower Model Training (Notebook 2)

This is where the system learns its personalization signal.

**What "the training input is the join of customers, articles, and transactions filtered through a feature view" actually means.**

The three feature groups (`customers`, `articles`, `transactions`) live in the feature store as separate tables. The two-tower model needs all three kinds of information *together* in each training row, so the system creates a **feature view** that defines the join. A feature view is essentially a saved SQL-style query that says: "start from `transactions`, then join the corresponding customer row from `customers` on `customer_id`, then join the corresponding article row from `articles` on `article_id`, and select these specific columns."

The driving table is `transactions`. Each row in `transactions` is one purchase: customer `C` bought article `A` on date `D`. After the join, that single row is enriched with the customer's `age` (from `customers`) and the article's `garment_group_name`, `index_group_name`, etc. (from `articles`). The `month_sin` and `month_cos` columns come from on-demand transformations applied to `t_dat` during the join. So one row of the joined dataset looks roughly like:

```
customer_id="C42", age=34, month_sin=-0.5, month_cos=-0.866,
article_id="A17", garment_group_name="Knitwear", index_group_name="Ladieswear"
```

That single row says: "Customer C42, age 34, in November, bought article A17, which is Knitwear in Ladieswear." There is no `label` column — the *existence* of the row is the positive signal. Every row in this joined dataset is a purchase that actually happened.

The "filtered through a feature view" part just means we don't materialize the full giant join on disk and read all of it. The feature view lazily produces only the columns we asked for (`customer_id`, `age`, `month_sin`, `month_cos`, `article_id`, `garment_group_name`, `index_group_name`) for the rows we need, and it handles the train/validation/test split for us via `train_validation_test_split(...)`. This is what `TwoTowerDataset.get_train_val_split()` in `recsys/training/two_tower.py` does.

**Why each row represents one purchase, and why that is the entire training signal.**

The two-tower model never sees a "didn't buy" example explicitly. There is no separate negative table here. The contrastive loss treats *other rows in the same training batch* as negatives. So when the batch contains 2048 real purchases, for each purchase, the 2047 other purchases' articles are implicitly "things this customer did not buy in this row." That gives the model both the positive signal (your row's customer-article pair) and the negative signal (every other article in the batch) without ever needing a synthesized negative dataset.

This is the entire reason the model can learn purchase-aware embeddings from a table that contains nothing but purchases. The training algorithm manufactures negatives on the fly from the batch.

**The rest of training.**

- The **QueryTower** consumes the customer-side columns (`customer_id`, `age`, `month_sin`, `month_cos`) of the row and outputs a 16-dim vector `u`.
- The **ItemTower** consumes the article-side columns (`article_id`, `garment_group_name`, `index_group_name`) of the row and outputs a 16-dim vector `v`.
- The retrieval loss pulls `u` and `v` closer for the real purchase row, and pushes `u` away from all other articles' `v'` vectors in the same batch.
- Backpropagation updates both towers together. They are trained jointly, not separately.
- Training output: a trained `QueryTower` and a trained `ItemTower`, saved to the Hopsworks Model Registry as `query_model` and `candidate_model`.

After this step, the **system knows how to encode a customer into a 16-dim vector and how to encode an article into a 16-dim vector** such that nearness in that space implies purchase likelihood. But there is no precomputed index of article vectors yet, and there is no scorer yet either.

### Step 4: CatBoost Ranking Model Training (Notebook 3)

- Reads the pre-built `ranking` feature group: ~224k rows of (customer_id, article_id, age, product attrs, month_sin, month_cos, label).
- Trains `CatBoostClassifier` with `scale_pos_weight=10` to handle the 10:1 negative imbalance, learning rate `0.2`, depth `10`, 100 iterations with early stopping.
- The output is a binary classifier: given a customer's age, current month, and an article's structured attributes, predict the probability of purchase.
- Saved to the Model Registry as `ranking_model`.

Importantly: the ranking model does **not** see embeddings (neither A nor B). It is purely tabular over structured features. It is also not aware of who the customer is by ID; it sees the customer only through `age` (and indirectly via `month_sin/cos`).

### Step 5: Precomputing Two-Tower Item Embeddings (Notebook 4)

- Load the trained `candidate_model` (the ItemTower) from the registry.
- For each article in the training set (~11,820 unique articles), run the ItemTower forward pass to produce a 16-dim vector (**Embedding B**).
- Store these as a new feature group `candidate_embeddings` with a vector index on the `embeddings` column.

This is the **vector index that retrieval will search at request time.** Without this step, there is no fast way to go from a customer vector to candidate article IDs. The trained model alone is not enough; we need its outputs cached and indexed.

### Step 6: Deploying The Online Services (Notebooks 5 and 7)

- Deploy `query_model_deployment`: the query tower wrapped in a transformer (`query_transformer.py`) that fetches customer features, computes `month_sin/month_cos` from the request timestamp, runs the QueryTower, and returns a query embedding.
- Deploy `ranking_deployment`: the CatBoost model wrapped in a transformer (`ranking_transformer.py`) that takes a query embedding, searches `candidate_embeddings` for the top-100 nearest articles, fetches their attributes, fetches customer features, calls `ranking_predictor.py`, and returns sorted predictions.
- Optionally deploy `llmranking_deployment`: same role as the CatBoost ranker but using GPT-4o-mini through LangChain (`llm_ranking_predictor.py`). Limited to 20 candidates for latency reasons.

After Step 6, three serving endpoints exist and are ready to be called by the Streamlit UI.

### What's In The Feature Store At The End Of Offline Work

- `articles` feature group: structured attributes + `article_description` + 384-dim **Embedding A** + `image_url`.
- `customers` feature group: structured customer features.
- `transactions` feature group: raw + temporal features, with `month_sin/month_cos` as on-demand transformations.
- `interactions` feature group: synthesized browsing events (currently not consumed by training).
- `ranking` feature group: precomputed positive/negative pairs for ranker training.
- `candidate_embeddings` feature group: per-article 16-dim **Embedding B**, indexed for vector search.
- Model registry: `query_model`, `candidate_model`, `ranking_model`, optionally an LLM ranker config.
- Deployments: `query`, `ranking`, optionally `llmranking`.

---

## 5. The Online Pipeline (What Happens When A User Asks For Recommendations)

### Inference Time: Two Phases

Before walking through the steps, it helps to understand the fundamental asymmetry in how the two towers are used at inference time.

**Phase 1 — Offline pre-compute (runs once, before any user shows up).**

After training, the ItemTower is run on every article in the catalog and its output vectors are stored in the `candidate_embeddings` feature group. Using the same toy numbers:

```
ArticleTower(beanie_id,   "Accessories",  "Ladieswear") → v_beanie   = [-0.18,  0.09]
ArticleTower(scarf_id,    "Accessories",  "Ladieswear") → v_scarf    = [ 0.28, -0.38]
ArticleTower(sundress_id, "Garment Upper","Ladieswear") → v_sundress = [ 0.02,  0.48]
ArticleTower(hat_id,      "Accessories",  "Ladieswear") → v_hat      = [-0.15,  0.12]
... (11,820 articles in the sampled dataset)
```

All vectors are stored with a vector index. **The ItemTower is never called again after this point.**

**Phase 2 — Online query (runs per user request, in milliseconds).**

Alice opens the app. The request arrives: `customer_id=alice`, `transaction_date=2022-11-15`.

Step 1 — run the QueryTower once:

```
QueryTower(alice_id, age=25, month_sin=−0.97, month_cos=−0.26)
    → q_alice = [−0.16, 0.10]
```

Step 2 — ANN search against all stored article vectors:

```
dot(q_alice, v_beanie)   = (−0.16)(−0.18) + (0.10)(0.09) = 0.038  ← high
dot(q_alice, v_scarf)    = (−0.16)(0.28)  + (0.10)(−0.38) = −0.083
dot(q_alice, v_sundress) = (−0.16)(0.02)  + (0.10)(0.48)  = 0.045  ← high
dot(q_alice, v_hat)      = (−0.16)(−0.15) + (0.10)(0.12)  = 0.036  ← high
```

Top candidates returned: `sundress (0.045)`, `beanie (0.038)`, `hat (0.036)`, ... These go to CatBoost for final scoring.

**The key asymmetry:**


|                      | ItemTower                      | QueryTower       |
| -------------------- | ------------------------------ | ---------------- |
| Runs at training     | Every batch                    | Every batch      |
| Runs offline         | Once per article (pre-compute) | Never            |
| Runs at request time | Never                          | Once per request |


This is why the architecture scales to 105K articles — article vectors are pre-computed and indexed once, and each user request is just one QueryTower forward pass plus a fast ANN lookup.

---

This is the runtime flow. Every step here is fast (target: 2-3 seconds end-to-end for CatBoost; 15-30s for LLM ranking). Nothing is trained or recomputed during online serving. Everything reads from what offline produced.

The Streamlit UI calls the `query` deployment with a JSON body like:

```json
[{"customer_id": "d327...ecf", "transaction_date": "2022-11-15T12:16:25.330916"}]
```

Then the following steps happen in order.

### Step 1: Query Transformer Preprocessing

File: `recsys/inference/query_transformer.py`.

- Extract `customer_id` and parse `transaction_date`.
- Look up that customer in the `customers` feature view to get their `age` (and any other customer features).
- Compute `month_sin` and `month_cos` for the request month using the `ranking` feature view's on-demand transformations.
- Build the input dict: `{customer_id, age, month_sin, month_cos}`.

This step exists because the QueryTower expects all four inputs, but the request only gives us two. The transformer enriches the request.

### Step 2: Query Tower Forward Pass

The TensorFlow `query_model` is called with that input dict. It outputs a 16-dim `query_emb` vector. This is essentially the customer's "taste fingerprint at this moment in time."

This vector is added to the request payload and the request is forwarded to the `ranking` deployment.

### Step 3: Candidate Retrieval (Vector Search)

File: `recsys/inference/ranking_transformer.py`.

- Take the `query_emb` and call `candidate_index.find_neighbors(query_emb, k=100)` on the `candidate_embeddings` feature view.
- Hopsworks returns the top-100 article IDs whose **Embedding B** vectors are closest to the query vector.
- These are the candidate articles. There is no scoring yet, just "these 100 items have item embeddings most similar to this user embedding."

This is the only place the two-tower retrieval is actually used at runtime. Both towers ran offline; the QueryTower runs per request; the ItemTower never runs at request time (its outputs were cached in Step 5 of offline).

### Step 4: Filter Already-Purchased Items

- Query the `transactions` feature group for all articles this customer already bought.
- Remove those from the candidate list.
- This typically leaves ~50-80 candidates.

The point: do not recommend something they already own.

### Step 5: Fetch Article Features For Each Remaining Candidate

- For each surviving article ID, call `articles_fv.get_feature_vector({"article_id": item_id})`.
- This pulls in structured attributes the ranking model needs: `product_type_name`, `product_group_name`, `graphical_appearance_name`, color fields, `department_name`, `index_name`, `index_group_name`, `section_name`, `garment_group_name`.
- Note: **Embedding A** (the 384-dim text vector) is technically in the articles feature group but it is excluded from the ranking feature view (see `select_except(["embeddings"])` in `feature_store.py`). So even at ranking time, Embedding A is not used.

### Step 6: Attach Customer And Temporal Features

- Add `age` (from the customer feature view) to every candidate row.
- Add the same `month_sin` and `month_cos` (the customer's request-time month) to every row.

Now every candidate row has the full feature set the CatBoost model expects: `[age, month_sin, month_cos, product_type_name, product_group_name, graphical_appearance_name, colour_group_name, perceived_colour_value_name, perceived_colour_master_name, department_name, index_name, index_group_name, section_name, garment_group_name]`.

### Step 7: Ranking Model Scoring

File: `recsys/inference/ranking_predictor.py`.

- The CatBoost classifier's `predict_proba` runs on all candidate rows at once.
- For each row, take the probability of class 1 (purchase) as the score.
- Pair each score back with its article ID.

### Step 8: Sort And Return

- Sort `(score, article_id)` tuples in descending order.
- Return the top-K to the UI.

### Step 9: Streamlit UI Displays Recommendations

- The UI calls `articles_fv.get_feature_vector(...)` for the top items to pull their `image_url`, `prod_name`, etc.
- Renders cards with the product images, scores, and interaction buttons.
- Tracks clicks/views in the `interactions` feature group.

### Summary Of The Online Sequence

Request → enrich with customer/time features → QueryTower → 16-dim query embedding → vector search over `candidate_embeddings` → top-100 articles → drop already-purchased → fetch their structured features → attach customer features → CatBoost scoring → sort → respond.

That is the entire main pipeline.

---

## 6. The Three Serving Pipelines

The system has three distinct serving pipelines. Confusing them is the most common source of misreading the code.


|                   | Pipeline 1              | Pipeline 2                 | Pipeline 3                              |
| ----------------- | ----------------------- | -------------------------- | --------------------------------------- |
| **Entry point**   | `customer_id`           | `customer_id`              | Free-text input                         |
| **Retrieval**     | Two-tower (Embedding B) | Two-tower (Embedding B)    | SentenceTransformer (Embedding A)       |
| **Re-ranking**    | CatBoost                | GPT-4o-mini                | None                                    |
| **Personalized?** | Yes                     | Yes                        | No                                      |
| **Key files**     | `ranking_predictor.py`  | `llm_ranking_predictor.py` | `recommenders.py` → `get_similar_items` |


---

### Pipeline 1: Main Personalized Recommendations (Two-Tower + CatBoost)

This is the primary flow described in Section 5. `customer_id` → QueryTower → ANN search over Embedding B → CatBoost re-ranking → sorted results.

---

### Pipeline 2: LLM Re-ranker (Two-Tower + GPT-4o-mini)

This is an alternative to Pipeline 1 — the **retrieval step is identical** (same two-tower model, same Embedding B, same ANN search). Only the re-ranking step differs: instead of CatBoost, GPT-4o-mini is used.

File: `recsys/inference/llm_ranking_predictor.py`.

The LLM receives each candidate article's structured features (age, month_sin, month_cos, product type, colour, garment group, etc.) formatted as a prompt and outputs a purchase probability. Capped at 20 candidates because one LLM call per candidate makes this slow (~15-30 seconds end-to-end).

**What `llm_ranking_predictor.py` is NOT:** it is not a text search tool and it does not use Embedding A. It is purely a drop-in replacement for CatBoost in the re-ranking step of the personalized pipeline.

---

### Pipeline 3: LLM Text Search (HyDE + Embedding A)

This is the part where **Embedding A** is used. A user types a natural-language request like "I'm going to the beach for a week-long vacation."

The actual flow (verified from `recsys/ui/recommenders.py`) is:

**Step 1 — LLM query expansion.** The raw user text is sent to GPT, which generates 3–5 specific item descriptions in H&M product language:

```
User: "I'm going to the beach for a week-long vacation"
GPT →
  "Floral print wrap dress with flutter sleeves"
  "Strappy nude block heel sandals"
  "Woven straw tote bag with leather handles"
  "Cropped denim jacket with raw hem"
```

**Step 2 — Encode each description and search.** For each generated description, `get_similar_items()` runs:

```python
description_embedding = embedding_model.encode(description)   # 384-dim
articles_fv.find_neighbors(description_embedding, k=25)       # Embedding A index
```

**Step 3 — Display by category, no re-ranking.** The top 5 matches per category are shown directly. There is no CatBoost or LLM scoring step after retrieval.

**Why this is not standard RAG.** In RAG the order is *retrieve → generate* (retrieved documents are injected into the LLM prompt to ground the answer). Here the order is *generate → retrieve* (the LLM generates hypothetical item descriptions first, and those are used as search queries). This pattern is known as **HyDE (Hypothetical Document Embeddings)**.

The reason HyDE works better than encoding the raw user query directly: the user query ("I'm going to the beach") lives in a different part of embedding space than article descriptions ("Floral print wrap dress..."). The LLM bridges that gap by producing text that is phrased the same way the catalog is phrased. Searching with the hypothetical description finds much closer neighbors in Embedding A than searching with the original request.

**What this pipeline does not do:** it does not know who the user is, does not use purchase history, and does not use the two-tower model or CatBoost.

This is also the answer to "why does `create_article_description` exist when we have `detail_desc`": it exists to feed this text search path with a richer, more normalized text string. `detail_desc` alone has gaps (some articles have no `detail_desc`) and lacks the product hierarchy context (color, group, section) that helps the SentenceTransformer make good neighbors in Embedding A space.

---

## 7. Putting It All Together: A Concrete End-to-End Example

Imagine customer `C42` opens the app on November 15, 2022.

**Offline state at that moment:**

- `C42` is in the `customers` feature group with `age=34`.
- The articles feature group has 11,820 articles, each with structured fields, an `article_description` string, a 384-dim text embedding, an image URL.
- The `candidate_embeddings` feature group has 11,820 16-dim vectors produced by the trained ItemTower.
- The trained `query_model`, `candidate_model`, and `ranking_model` are all in the registry.
- The `query` and `ranking` deployments are running.

**Runtime sequence:**

1. UI sends `{customer_id: "C42", transaction_date: "2022-11-15T12:16:25"}` to the `query` deployment.
2. `query_transformer.py` looks up `C42` and finds `age=34`, computes `month_sin=sin(11 * 2π/12) ≈ -0.5`, `month_cos=cos(...) ≈ -0.866`.
3. `query_model` (the QueryTower) takes `{customer_id: "C42", age: 34, month_sin: -0.5, month_cos: -0.866}` and outputs a 16-dim vector, call it `q = [0.12, -0.34, ...]`.
4. The request continues to the `ranking` deployment with `q` attached.
5. `ranking_transformer.py` calls `candidate_index.find_neighbors(q, k=100)`. Hopsworks searches the `candidate_embeddings` index and returns 100 article IDs.
6. The transformer queries the `transactions` feature group for `customer_id == "C42"` and removes any of those 100 IDs that `C42` already bought. Say 80 remain.
7. For each of those 80 articles, the transformer pulls structured attributes from the `articles` feature view (without the heavy text/embedding columns).
8. It attaches `age=34`, `month_sin=-0.5`, `month_cos=-0.866` to every row.
9. It sends the 80x14 feature matrix and the 80 article IDs to `ranking_predictor.py`, which runs `model.predict_proba`.
10. The CatBoost model returns 80 purchase probabilities.
11. The transformer pairs them with article IDs and sorts descending.
12. UI receives `[[0.94, "592846001"], [0.91, "536139006"], [0.88, "408554004"], ...]` and renders product cards.

The whole thing takes ~2-3 seconds.

If instead `C42` types "warm wool sweater for winter" into the LLM search box, none of the above runs. Instead (Pipeline 3):

1. GPT receives "warm wool sweater for winter" and generates specific item descriptions:
  - "Oversized chunky cable-knit wool sweater in cream"
  - "Slim-fit ribbed turtleneck in dark grey merino wool"
  - "Relaxed-fit longline knit cardigan with button front"
2. For each description, Streamlit calls `embedding_model.encode(description)` → 384-dim vector.
3. Calls `articles_fv.find_neighbors(description_embedding, k=25)` against the **Embedding A** index.
4. Renders the top 5 matches per category. No CatBoost scoring, no customer ID involved.

Three pipelines, two embedding spaces, different uses for each.

---

## 8. Things That Look Wrong But Are Actually By Design

These tend to confuse readers of the code. None of them are bugs.

- **The ItemTower is "thinner" than the QueryTower features-wise.** The QueryTower takes 4 inputs, the ItemTower takes 3. This is okay because both towers' job is to map their respective entity into the *same 16-dim space*. The number of input features per tower does not need to match; what matters is the output dimension matches.
- **The ranking model does not use either embedding.** Many readers expect Embedding A or B to be a feature in the ranker. They are not. The ranker is purely tabular on structured fields. This is a known limitation; see "Improvements Worth Considering" below.
- **The `interactions` feature group is built but unused for training.** The synthetic click/ignore/purchase events are generated in feature engineering but the ranker is trained from `transactions` + random negatives, not from `interactions`. The interactions data is mostly groundwork for future improvements.
- `**detail_desc` is dropped from the final articles feature group.** It is consumed once to build `article_description` and then removed because (a) it has nulls (the code drops any column with nulls) and (b) the system stores the embedding of the text, not the raw text. The information lives on in the 384-dim vector.
- **There are two embedding indexes.** `articles.embeddings` (384-dim text) and `candidate_embeddings.embeddings` (16-dim behavior). They serve different search paths and are queried by different things (SentenceTransformer-encoded text vs. QueryTower-encoded customers).
- **The CatBoost ranker reaches 98% F1 but the two-tower validation metrics are all 0.0%.** The training analysis flags this as a real issue in the two-tower training run, not a feature. The CatBoost numbers are also somewhat optimistic because the negatives are random, which makes the classification task easier than real ranking.

---

## 9. Improvements Worth Considering

These are concrete next steps a future iteration could take, ordered roughly by impact-to-effort ratio:

1. **Feed Embedding A into the ItemTower.** Add `article_description` text embeddings as an input feature to the ItemTower (projected down to a manageable size). This would let the retrieval model use semantic article meaning, not just `garment_group_name` and `index_group_name`. It would also help cold-start items.
2. **Feed Embedding A into the CatBoost ranker.** Either include the raw vector as numeric features or include a small PCA/projection of it. This brings text signal into the ranker.
3. **Replace random negatives with harder negatives.** Use items retrieved by the two-tower model that the customer did not buy. This better matches the ranking task and prevents the ranker from learning trivial shortcuts.
4. **Use the `interactions` feature group for ranker training.** Treat clicks as weak positives and ignores as informed negatives. This better reflects exposure.
5. **Bump embedding dimensions.** 16-dim is small. 32 or 64 would preserve more signal, especially if text embeddings are added as input.
6. **Add ranking metrics.** Recall@K, NDCG@K, MAP@K give a much better picture than binary precision/recall on a 10:1 imbalance.
7. **Use the `customer_id` in the ranker.** Currently the ranker only knows the customer through `age`. A per-customer feature like recent purchase categories or two-tower customer embedding would inject true personalization into the ranker.

---

## 10. Quick Reference Cheat Sheet


| You want to know...                      | Look at...                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------- |
| How an article's text gets embedded      | `recsys/features/articles.py` (`create_article_description`, `generate_embeddings_for_dataframe`) |
| How customer/article towers are built    | `recsys/training/two_tower.py`                                                                    |
| How the ranker is trained                | `recsys/training/ranking.py`                                                                      |
| How item embeddings get precomputed      | `recsys/features/embeddings.py` + Notebook 4                                                      |
| How a user request flows through serving | `recsys/inference/query_transformer.py` → `ranking_transformer.py` → `ranking_predictor.py`       |
| How text search works                    | `recsys/ui/recommenders.py` (`get_similar_items`), uses `articles_fv.find_neighbors`              |
| Where feature groups are defined         | `recsys/hopsworks_integration/feature_store.py`                                                   |
| Where the article schema is declared     | `recsys/hopsworks_integration/constants.py`                                                       |


If you remember only one thing from this doc: **Embedding A is text, Embedding B is behavior, the main pipeline uses B, the search pipeline uses A, and the CatBoost ranker uses neither.** The rest follows from there.
---
⚠️ **END OF REFERENCE PROJECT FILE** ⚠️

Remember: This is archived code. Use `system-design/` for current implementation.

---
