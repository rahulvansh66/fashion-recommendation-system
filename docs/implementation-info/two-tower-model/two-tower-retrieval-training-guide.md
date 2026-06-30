# Two-Tower Retrieval Model — Training Guide

**Source:** `tmp/notebooks/2_tp_training_retrieval_model.ipynb` and `tmp/recsys/`  
**Implementation (this repo):** [`two-tower-retrieval-implementation-guide.md`](./two-tower-retrieval-implementation-guide.md) — code layout, pipelines, SageMaker, MLflow/Optuna, runbook  
**Purpose:** Document how the Stage-1 retrieval model is trained — input features, preprocessing, sampling, architecture, hyperparameters, and evaluation. 

The retrieval model narrows the full H&M catalog (~~105k articles) to a small candidate set (~~100 items) using a **two-tower architecture**. Each tower embeds its side of a user–item interaction into a shared low-dimensional vector space. At inference, nearest-neighbor search over item embeddings finds candidates whose vectors are closest to the query (user) embedding.

---

## 1. Input Features

Training data comes from each purchase row in `transactions`:

Each training row represents one real purchase: customer `C` bought article `A` at time `T`. There is no explicit `label` column — the row itself is the positive signal.

### 1.1 Features consumed by the model

Only a subset of columns from the joined feature view are passed into the two towers.

**Query tower (user + context)**


| Feature       | Type        | Description                                     |
| ------------- | ----------- | ----------------------------------------------- |
| `customer_id` | string (ID) | Unique customer identifier                      |
| `age`         | float       | Customer age at time of purchase                |
| `month_sin`   | float       | Sine encoding of purchase month (seasonality)   |
| `month_cos`   | float       | Cosine encoding of purchase month (seasonality) |


**Candidate tower (item)**


| Feature              | Type        | Description                                             |
| -------------------- | ----------- | ------------------------------------------------------- |
| `article_id`         | string (ID) | Unique article identifier                               |
| `garment_group_name` | categorical | Garment type (e.g. Trousers, Blouses, Dresses Ladies)   |
| `index_group_name`   | categorical | Top-level category (e.g. Ladieswear, Menswear, Divided) |


### 2 In-model preprocessing (training time)

Applied inside `QueryTower` and `ItemTower` (`recsys/training/two_tower.py`):

**Query tower**


| Input                    | Preprocessing                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `customer_id`            | `StringLookup` vocabulary built from unique training customer IDs (+1 slot for unknown tokens) → `Embedding(emb_dim=16)` |
| `age`                    | `Normalization` layer adapted on the training set (zero mean, unit variance)                                             |
| `month_sin`, `month_cos` | Passed through as-is (already scaled to [-1, 1])                                                                         |


Concatenated vector → `Dense(16, relu)` → `Dense(16)` → query embedding.

**Candidate tower**


| Input                | Preprocessing                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| `article_id`         | `StringLookup` vocabulary from unique training article IDs (+1 unknown slot) → `Embedding(emb_dim=16)` |
| `garment_group_name` | `StringLookup` → `tf.one_hot` over garment-group vocabulary                                            |
| `index_group_name`   | `StringLookup` → `tf.one_hot` over index-group vocabulary                                              |


Concatenated vector → `Dense(16, relu)` → `Dense(16)` → item embedding.

**Dataset pipeline**

- Train/validation split via Hopsworks `train_validation_test_split`.
- Convert DataFrame to `tf.data.Dataset.from_tensor_slices`.
- Training set: `.batch(2048).cache().shuffle(batch_size x 10)`.
- Validation set: `.batch(2048).cache()`.

---

## 3. Positive and Negative Sample Strategy

The retrieval model uses **implicit feedback** from purchase history. There is no separate negative dataset.

### 3.1 Positive samples

- **Source:** Every row in the joined `transactions` feature view.
- **Definition:** Each row is a `(customer_id, article_id)` pair that represents a confirmed purchase.
- **Context:** User-side features (`age`, `month_sin`, `month_cos`) and item-side features (`garment_group_name`, `index_group_name`) come from the same row.

Example training row:

```text
customer_id="C42", age=34, month_sin=-0.5, month_cos=-0.866,
article_id="A17", garment_group_name="Knitwear", index_group_name="Ladieswear"
```

### 3.2 Negative samples — in-batch contrastive learning

Negatives are **not precomputed**. The TensorFlow Recommenders `Retrieval` task generates them implicitly:

- For each batch of 2,048 purchase rows, every row's `(user_embedding, item_embedding)` pair is the **positive**.
- All **other items in the same batch** (2,047 per row) act as **in-batch negatives**.
- The loss is a softmax over batch items: pull the true pair together, push all other batch items apart.

This contrastive setup avoids building an explicit negative table while still teaching the model what a customer did *not* buy in the context of co-occurring batch items.

---

## 4. Model Architecture

The model is a **two-tower (dual-encoder) retrieval** system built with TensorFlow + TensorFlow Recommenders (`tfrs`). The two towers are trained jointly to project users and items into the same 16-dimensional vector space, where proximity means purchase likelihood.

### 4.1 Full architecture diagram

```
 Training batch (B rows)
 customer_id, age, month_sin, month_cos,
 article_id, garment_group_name, index_group_name
              |
    __________|__________
   |                     |
   v                     v
QUERY TOWER          CANDIDATE TOWER
(tf.keras.Model)     (tf.keras.Model)

customer_id          article_id
  StringLookup         StringLookup
  Embedding            Embedding
  [16-d]               [16-d]

age                  garment_group_name
  Normalization        StringLookup
  reshape [1-d]        tf.one_hot
                       [~26-d]

month_sin  [1-d]     index_group_name
month_cos  [1-d]       StringLookup
                       tf.one_hot
                       [~6-d]

tf.concat [19-d]     tf.concat [~48-d]
  Dense(16, relu)      Dense(16, relu)
  Dense(16)            Dense(16)
  [16-d]               [16-d]
       |                     |
       u (query vec)         v (item vec)
       |_____________________|
                 |
                 v
    tfrs.tasks.Retrieval
    loss = -log( exp(u.v_pos) / sum_j exp(u.v_j) )
                 j in batch
    + AdamW weight decay regularization
```

---

### 4.2 Query tower — every component explained

The query tower produces a 16-d vector that answers: *"what kind of articles does this customer want, in this month?"*

#### A. `StringLookup` for `customer_id`

**What it does:** Maintains a vocabulary of all unique training customer IDs. Converts a raw string ID (e.g. `"f7048acb…"`) into an integer index. Any ID not seen during vocabulary construction maps to the reserved `+1` unknown index.

**Why it is needed:** Neural networks require numeric inputs. Customer IDs are opaque hash strings with no ordinal meaning — you cannot treat `"f70…"` as a number. `StringLookup` creates a stable string-to-integer mapping so the downstream embedding layer can address the correct row in its weight matrix.

**Why train-only vocabulary:** Using only training IDs prevents validation/test customer identities from leaking into the vocabulary and influencing learned representations.

---

#### B. `Embedding(num_users + 1, emb_dim=16)` for `customer_id`

**What it does:** A trainable weight matrix of shape `(num_users+1, 16)`. The integer index from `StringLookup` selects one row — that customer's 16-d representation. Weights are randomly initialized and updated by backpropagation.

**Why embeddings instead of one-hot:** A one-hot for 966 users is a 966-d sparse vector. Embedding compresses this to 16 dense trainable dimensions, which drastically reduces parameter count and allows the model to generalize by learning to place similar users near each other in the 16-d space.

**The +1 slot:** Reserves one embedding row for any user not in the training vocabulary (cold-start users at inference). Without this the model would crash on an out-of-bounds index.

**Collaborative signal:** Because the embedding is trained end-to-end on purchase data, users who bought similar articles will have similar embedding vectors — even without explicit user-similarity features. This is the core collaborative filtering signal the tower learns.

---

#### C. `Normalization` for `age`

**What it does:** A stateful Keras layer. During `adapt(train_data)` it computes the training-set mean and variance of `age`. At call time it outputs `(age - mean) / std` — standard z-score normalization.

**Why normalization matters:** The concatenated input before the Dense layers combines a 16-d embedding vector (values roughly in `[-1, 1]`) with raw `age` (values `[15, 90]`). Without scaling, the large magnitude of raw age dominates the gradients, making the optimizer spend most of its budget fitting the age dimension and effectively ignoring the embedding dimensions. Z-score normalization puts all inputs on the same scale so every dimension contributes comparably to the dot products inside the Dense layer.

**Why adapted on train only:** Using the training distribution prevents the test mean/variance from leaking into the normalization statistics.

---

#### D. `month_sin` and `month_cos` — pass-through scalar inputs

**What they are:** Reshaped from scalars to column vectors `(batch, 1)` so they can be concatenated with the 16-d embedding and 1-d normalized age.

**Why no further preprocessing:** Both values are in `[-1, 1]` by construction (sine and cosine always are). They are already on the same scale as the other inputs after normalization.

**Why two values and not raw month integer:** Treating month as an integer `1..12` would mislead the model: December (12) and January (1) are numerically far apart but seasonally adjacent. Sine + cosine encoding wraps the month onto a unit circle so that Euclidean distance in (sin, cos) space equals angular distance in calendar time. The model can learn "winter buyers tend to prefer X" without needing to special-case the year boundary.

---

#### E. `tf.concat` — fusing all query inputs

Stacks the four vectors along axis 1:

```
[customer_embedding (16-d)  |  normalized_age (1-d)  |  month_sin (1-d)  |  month_cos (1-d)]
= concatenated vector (batch, 19)
```

**Why concatenation and not addition or multiplication:** The three input types live in different representation spaces — a learned collaborative embedding vs. a scalar context feature. Addition would force them to share the same directional meaning in the 16-d space. Concatenation preserves each signal in its own dimensions and lets the subsequent Dense layers learn how to weight and combine them freely.

---

#### F. `Dense(16, activation="relu")` — first feedforward layer

**What it does:** `output = relu(W * x + b)` where `W` has shape `(16, 19)`.

**Purpose:** Projects the 19-d concatenated input into 16-d and applies a **non-linearity**. Without this layer the tower would be entirely linear: just a dot product of the embedding and a projection matrix. A linear model cannot capture cross-feature interactions — for example, "a 20-year-old buying in summer" is different from "a 60-year-old buying in summer," and that difference requires non-linear mixing of the age and month features with the user ID embedding.

**Why ReLU specifically:**

- Does not suffer from vanishing gradients for positive activations (gradient is exactly 1 when input > 0).
- Computationally cheap — just a max(0, x).
- Produces sparse activations, which reduces co-adaptation between neurons and can act as implicit regularization.
- Standard choice for hidden layers in recommendation models.

**Why 16 units:** Matches the final embedding dimension. Keeping the hidden layer the same width as the output avoids an information bottleneck at an intermediate step while keeping the total parameter count small (19×16 + 16 = 320 parameters for this layer).

---

#### G. `Dense(16)` — output projection layer (no activation)

**What it does:** A second linear transformation: `output = W * x + b`, producing the final 16-d query embedding `u`.

**Why no activation:** The retrieval loss compares `u` and `v` via dot product. If ReLU were applied here, all output components would be non-negative, which would restrict the embedding vectors to one orthant of the 16-d space. This would prevent the model from learning negative correlations (e.g., "this user tends *not* to buy sportswear, which should be far from their embedding in one direction"). A linear output allows vectors to point in any direction in the full 16-d space.

**Why a second Dense layer at all:** The first Dense introduces non-linearity; the second gives the model a final linear re-mixing pass to shape the geometry of the output space before the contrastive loss is applied. This is the standard two-layer MLP "encoder" pattern in retrieval models.

---

### 4.3 Candidate tower — every component explained

The candidate tower produces a 16-d vector that answers: *"what kind of customer would buy this article?"*

#### A. `StringLookup` + `Embedding(num_items + 1, 16)` for `article_id`

Same mechanism as the user embedding. Learns a collaborative item representation: articles frequently purchased by similar customers will cluster together in the embedding space. The `+1` handles cold-start articles at inference. This is the primary identity-based item signal.

---

#### B. `StringLookup` + `tf.one_hot` for `garment_group_name`

**What it does:** `StringLookup` maps a garment group string (e.g. `"Trousers"`) to an integer index. `tf.one_hot` then expands that integer into a binary vector of length equal to the garment-group vocabulary size (~26 in the reference run).

**Why one-hot instead of learned embedding for this categorical:**

The garment group vocabulary is small (~26 values). A learned embedding for 26 categories would need a 26×16 weight matrix — not much signal to train on, and easy to overfit or collapse with a small dataset (966 customers). A one-hot vector is a lossless, parameter-free encoding: every category is equidistant from every other as input to the Dense layer, and the Dense layer weights then learn which directions in the 16-d output space each garment group should push towards. With a tiny vocabulary, one-hot + Dense is functionally equivalent to an embedding but without the risk of embedding collapse.

**Why include garment group at all:** The article ID embedding alone captures identity-based collaborative signal but cannot generalize to articles not seen during training. Garment group provides a "type-level" content signal: a customer who bought several dresses is likely to want more dresses even if the specific new dress was never in the training data. This is the content-based fallback when collaborative signal is absent.

---

#### C. `StringLookup` + `tf.one_hot` for `index_group_name`

Same logic as garment group. `index_group_name` (e.g. `Ladieswear`, `Menswear`, `Divided`) captures the broadest catalog category. The vocabulary is even smaller (~6 values). Together with garment group it gives the tower a two-level item taxonomy — broad category plus fine garment type — enabling cross-item generalization at both coarse and fine granularity.

---

#### D. `tf.concat` of article embedding + garment one-hot + index one-hot

```
[article_embedding (16-d)  |  garment_one_hot (~26-d)  |  index_one_hot (~6-d)]
= concatenated vector (batch, ~48)
```

Combines the collaborative ID signal (who else bought this item) with content-level category signals (what kind of item it is), enabling the tower to generalize both within seen articles and to unseen ones.

---

#### E. `Dense(16, relu)` + `Dense(16)` — same structure as query tower

The two-layer MLP projects the ~48-d concatenated item input down to the 16-d shared embedding space. Using the same architecture on both towers keeps the embedding spaces structurally symmetric, which is important for the dot-product comparison in the retrieval loss to be geometrically meaningful — both towers map into the same ℝ¹⁶ space with the same "units."

---

### 4.4 Loss function — `tfrs.tasks.Retrieval` (softmax cross-entropy)

This is the core learning objective.

**What it computes:**

For a batch of `B` purchase rows, the model produces:

- `U` of shape `(B, 16)` — all query embeddings
- `V` of shape `(B, 16)` — all candidate embeddings

The score matrix is `S = U * V^T` of shape `(B, B)`. Entry `S[i,j]` is the dot product of the i-th user embedding with the j-th item embedding.

The loss for row `i`:

```
L_i = -log( exp(S[i,i]) / sum over j of exp(S[i,j]) )
```

This is categorical cross-entropy where the "correct class" for the i-th user is the i-th item (the one they actually bought). All other `B-1` items in the batch are treated as implicit negatives.

Total loss = mean of `L_i` over all B rows.

**Why softmax cross-entropy and not pairwise (BPR) or margin loss:**


| Alternative               | Issue                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Pairwise BPR              | Requires explicit (positive, negative) pairs; no precomputed negatives exist here                                  |
| Hinge / margin            | Requires careful margin hyperparameter tuning; harder to optimize at scale                                         |
| Softmax (sampled softmax) | Naturally handles many negatives from the batch; widely proven in retrieval (YouTube DNN, Google two-tower papers) |


Softmax treats the problem as "which of the B items did this user buy?" — a B-class classification per row. It is simple, numerically stable, and scales well with batch size.

**Why large batch size (2048) is critical for this loss:**

More items in the batch = more in-batch negatives per user = harder classification problem = richer gradient signal per step = better embedding geometry. Batch size is the single most important hyperparameter for retrieval quality with in-batch negatives. A batch of 2048 gives each user 2047 negatives per training step.

---

### 4.5 Regularization — AdamW weight decay

**What it does:** AdamW applies L2 regularization directly in the weight update rule rather than adding a penalty term to the loss. With weight decay `wd = 0.001` and learning rate `lr = 0.01`, each weight is shrunk before the gradient step:

```
w <- w * (1 - lr * wd) - lr * gradient
```

**Why regularization is needed:** Without it, embedding weights can grow unboundedly. Very large embedding vectors produce extremely high dot product scores, which saturate the softmax denominator — all probability mass concentrates on a few entries, gradients become near-zero, and training stalls. Weight decay keeps embedding norms bounded and gradients informative throughout training.

**Why AdamW over Adam + L2 loss penalty:** When L2 is added to the loss in standard Adam, the regularization is effectively scaled by the per-parameter adaptive learning rate. This makes the actual regularization strength uneven across parameters — parameters with small gradient variance (rare users/items) get weaker regularization than they should. AdamW decouples weight decay from the adaptive step size, applying it uniformly across all parameters regardless of their gradient history. This is better-calibrated regularization and is the standard in large embedding models.

---

### 4.6 Optimizer — AdamW

**What it does:** Maintains per-parameter running estimates of:

- First moment (exponential moving average of gradients) → `m`
- Second moment (exponential moving average of squared gradients) → `v`

Update: `theta <- theta - lr * m_hat / (sqrt(v_hat) + epsilon)` where `m_hat` and `v_hat` are bias-corrected estimates.

**Why Adam over SGD for embedding models:**

Embedding tables are sparse: in a batch of 2048 rows, only 2048 out of ~12k item embedding rows receive non-zero gradients. SGD with momentum would either under-update rare items (low learning rate) or over-update frequent items (high learning rate) — there is no single learning rate that works well for all items simultaneously.

Adam's per-parameter adaptive rates automatically solve this: rare items accumulate small gradient variance estimates early on, which gives them a high effective learning rate; frequent items accumulate large variance estimates, which gives them a lower effective learning rate. This self-calibration is exactly what sparse embedding training needs.

---

### 4.7 Custom `train_step` and `test_step`

The `TwoTowerModel` overrides Keras's default training step with an explicit `tf.GradientTape`:

```python
with tf.GradientTape() as tape:
    user_embeddings = self.query_model(batch)
    item_embeddings = self.item_model(batch)
    loss = self.task(user_embeddings, item_embeddings, compute_metrics=False)
    regularization_loss = sum(self.losses)
    total_loss = loss + regularization_loss

gradients = tape.gradient(total_loss, self.trainable_variables)
self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
```

**Why a custom step:**

- The same batch dict must be routed to **both towers** simultaneously. The default Keras `model(inputs)` call assumes one model processes inputs once; here two separate sub-models process the same batch in parallel.
- `compute_metrics=False` during training skips the expensive `FactorizedTopK` computation (which scores every query against the full ~11k item corpus) on every training batch — that computation is reserved for validation-time evaluation only.
- Tracking `regularization_loss` separately from `loss` in the returned metrics dict makes it easy to inspect whether weight decay is dominating or the retrieval loss is driving training — useful for debugging stability.

---

### 4.8 Evaluation layer — `tfrs.metrics.FactorizedTopK`

**What it does:** During validation, builds a brute-force nearest-neighbor index over the full candidate corpus (all unique training articles passed through the candidate tower once). For each validation query embedding, computes dot products against every item embedding and identifies the top-K highest-scoring items.

**Why "factorized":** The metric pre-embeds all candidate items once per epoch, producing a fixed matrix `V_all` of shape `(N_items, 16)`. Per-batch scoring is then just a matrix multiply `U_batch * V_all^T` — O(batch × N_items × 16) instead of passing every (user, item) pair through the full model.

**Why top-100 is the headline metric:**

The retrieval model is Stage 1 in a two-stage pipeline. The XGBoost ranker (Stage 2) will only score the top-K candidates returned by retrieval. If the truly purchased article is not in the retrieved top-100, the ranker has zero chance of surfacing it. Top-100 recall is therefore the direct business-relevant metric: it measures whether Stage 1 keeps the correct answer in the candidate set for Stage 2 to rank.

---

### 4.9 Summary table — all components and their purpose


| Component                           | Tower      | Purpose                                                                                                       |
| ----------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| `StringLookup` (IDs)                | Both       | Convert opaque string IDs to integer indices for embedding lookup; handles unknown IDs at inference           |
| `Embedding` (ID)                    | Both       | Trainable dense lookup; learns collaborative signal — similar users/items cluster together                    |
| `Normalization`                     | Query      | Z-score scaling of `age` so its magnitude does not dominate embedding values during training                  |
| `tf.one_hot` (categoricals)         | Candidate  | Lossless, parameter-free encoding of small garment/index vocabularies; avoids embedding collapse risk         |
| `tf.concat`                         | Both       | Fuses heterogeneous inputs (collaborative ID signal + content metadata + temporal context) into one vector    |
| `Dense(16, activation="relu")`      | Both       | Non-linear mixing of concatenated inputs; enables cross-feature interactions that a linear model cannot learn |
| `Dense(16)` (no activation)         | Both       | Linear output projection; preserves all directions in the shared embedding space for dot-product comparison   |
| `tfrs.tasks.Retrieval` (softmax CE) | Combined   | In-batch contrastive loss: treats true purchase pairs as positives and all other batch items as negatives     |
| AdamW optimizer                     | Training   | Adaptive per-parameter learning rates; handles sparse embedding gradients well                                |
| Weight decay `0.001`                | AdamW      | Keeps embedding norms bounded; prevents softmax saturation and gradient stalling                              |
| `FactorizedTopK`                    | Evaluation | Top-K recall over full item corpus; directly measures Stage-1 pipeline value for the downstream ranker        |


---

## 5. Finalized Training Parameters

All hyperparameters are defined in `recsys/config.py` (`Settings` class):


| Parameter                                | Value                     | Description                                    |
| ---------------------------------------- | ------------------------- | ---------------------------------------------- |
| `TWO_TOWER_MODEL_EMBEDDING_SIZE`         | `16`                      | Output dimension for both towers               |
| `TWO_TOWER_MODEL_BATCH_SIZE`             | `2048`                    | Batch size for training and in-batch negatives |
| `TWO_TOWER_NUM_EPOCHS`                   | `10`                      | Training epochs                                |
| `TWO_TOWER_LEARNING_RATE`                | `0.01`                    | AdamW learning rate                            |
| `TWO_TOWER_WEIGHT_DECAY`                 | `0.001`                   | AdamW L2 weight decay                          |
| `TWO_TOWER_DATASET_VALIDATON_SPLIT_SIZE` | *deprecated*              | Replaced by snap-date val windows (FR-BATCH-02) |
| `TWO_TOWER_DATASET_TEST_SPLIT_SIZE`      | *deprecated*              | Replaced by snap-date test window (FR-BATCH-02) |
| `CUSTOMER_DATA_SIZE`                     | `SMALL` (1,000 customers) | Upstream dataset sampling size                 |


**Optimizer:** `torch.optim.AdamW` with the learning rate and weight decay above.

**Initialization:**

- Age `Normalization` layer is adapted on the training dataset before the first epoch.
- Query tower is warm-started with a single batch from `query_df` to build lookup tables.

**Framework:** PyTorch (`torch.nn`, manual training loop, Recall@K in `evaluate.py`).

---

## 6. Evaluation Strategy

### 6.1 Primary metric — Top-K categorical accuracy

Evaluation uses `tfrs.metrics.FactorizedTopK` with the **full candidate corpus** (all unique training articles embedded through the candidate tower):


| Metric                         | Meaning                                                       |
| ------------------------------ | ------------------------------------------------------------- |
| `top_1_categorical_accuracy`   | Fraction of validation purchases where the true item ranks #1 |
| `top_5_categorical_accuracy`   | True item in top 5                                            |
| `top_10_categorical_accuracy`  | True item in top 10                                           |
| `top_50_categorical_accuracy`  | True item in top 50                                           |
| `top_100_categorical_accuracy` | True item in top 100                                          |


The notebook highlights **top-100 accuracy** as the headline retrieval metric: for each validation purchase, embed the query, retrieve the 100 nearest items, and check whether the actually purchased article appears in that set.

### 6.2 Evaluation procedure

1. Build candidate index from **deduplicated train-split articles** (encoded through the candidate tower once per epoch).
2. For each validation batch, compute query and item embeddings.
3. Compute dot products of each query embedding against the full candidate matrix (`Recall@K` in `evaluate.py`).
4. Report `val_recall_at_100` (headline metric) and per-epoch val loss.

### 6.3 Training vs validation monitoring

During `model.fit()`:

- **Training loss** (`loss`, `total_loss`): retrieval softmax loss on training batches.
- **Validation loss** (`val_loss`): same loss on held-out purchases.
- **Validation top-K metrics**: computed each epoch on the validation set.

Loss curves (training vs validation) are plotted at the end of the notebook to check for overfitting.

### 6.4 What evaluation does not cover

- **Cold-start users/items** not seen in training vocabularies fall back to the `+1` unknown embedding slot; performance on truly new IDs is not separately reported.
- **Ranking quality** (precision of ordering within top-100) is deferred to the XGBoost ranking model in notebook 3.
- **Test split** (10%) is created by the feature view but not evaluated in notebook 2.

---

## 7. Training Outputs

After training, two models are registered in the Hopsworks Model Registry:


| Registry name     | Component    | Purpose                                                                         |
| ----------------- | ------------ | ------------------------------------------------------------------------------- |
| `query_model`     | `QueryTower` | Encode `(customer_id, age, month_sin, month_cos)` → 16-d vector at serving time |
| `candidate_model` | `ItemTower`  | Encode `(article_id, garment_group_name, index_group_name)` → 16-d vector       |


Notebook 4 precomputes item embeddings from `candidate_model` into the `candidate_embeddings` feature group for ANN retrieval at inference.

---

---

## References


| Resource                            | Location                                                             |
| ----------------------------------- | -------------------------------------------------------------------- |
| Training notebook                   | `tmp/notebooks/2_tp_training_retrieval_model.ipynb`                  |
| Feature engineering notebook        | `tmp/notebooks/1_fp_computing_features.ipynb`                        |
| Two-tower implementation            | `tmp/recsys/training/two_tower.py`                                   |
| Retrieval feature view              | `tmp/recsys/hopsworks_integration/feature_store.py`                  |
| Hyperparameters                     | `tmp/recsys/config.py`                                               |
| Transaction / customer / article FE | `tmp/recsys/features/transactions.py`, `customers.py`, `articles.py` |


