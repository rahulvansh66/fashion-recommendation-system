# Quick-and-Easy XGBoost Ranking Guide

**Source notebook:** `[notebooks/example-quick-and-easy-model-build.ipynb](../../../notebooks/example-quick-and-easy-model-build.ipynb)`  
**Origin:** H&M Personalized Fashion Recommendations Kaggle competition reference solution (author workflow: Snowflake datamart → XGBoost pair classifier → candidate generation → batch scoring)  
**Related:** `[ranking-model-training-guide.md](./ranking-model-training-guide.md)` · `[features-eng.md](./features-eng.md)` · `[v1-requirements.md](../../system-design/v1/v1-requirements.md)`  
**Purpose:** Document the notebook end-to-end so you can reproduce the approach locally, understand why it worked on Kaggle, and see how it differs from v1.

This is a **reference / learning** path, not the v1 production pipeline. v1 uses temporal splits, window-aware negatives, two-tower retrieval + XGBoost ranker, and SageMaker — see [§12](#12-reference-vs-v1-summary).

---

## 1. Problem framing

The competition asks: for each customer, predict **12 article IDs** they are most likely to purchase in the next week.

The notebook treats this as **binary classification on `(customer_id, article_id)` pairs**:


| Label      | Meaning                                                                    |
| ---------- | -------------------------------------------------------------------------- |
| `SOLD = 1` | Customer purchased the article in the label week                           |
| `SOLD = 0` | Random `(customer_id, article_id)` pair with no purchase in the label week |


At inference, score many candidate pairs per customer, sort by `P(SOLD=1)`, and take the **top 12** article IDs.

```text
Feature datamart (Snowflake → CSV)
    → train XGBoost on pair rows (positives + random negatives)
    → score candidate pairs per customer
    → top-12 by XGB_SCORE → submission.csv
```

---

## 2. How the training dataset is built

**Scope of this section:** The notebook does **not** contain the Snowflake SQL that builds `model_build_base.csv`. That work is described only in the notebook’s opening markdown cell. Everything below is taken from that cell plus the notebook cells that **read and filter** the exported file. Where the author does not specify a detail, it is called out explicitly.

### 2.1 Where the dataset comes from

The author loaded the full H&M competition data into **Snowflake**, built a **datamart** there, and exported one flat file:


| Artifact                                    | Stated role                                                                                    |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `model_build_base.csv`                      | All training and test `(customer, article)` rows with labels and joined features               |
| `articles_predictions.csv`                  | Article-level features used at inference (not for training-table construction in the notebook) |
| `customers_prediction.csv`                  | Customer-level features used at inference                                                      |
| `model_suggested_items_enriched_sorted.csv` | Pre-built candidate pairs for submission scoring                                               |


The training notebook **starts from the pre-built CSV** on Kaggle (`/kaggle/input/hm-model-build-base/model_build_base.csv`). There is no dataset-build script in this repository.

### 2.2 Core row shape

Every row in `model_build_base.csv` represents one **(customer, article) pair** at one **snap date**. The author documents these key columns:


| Column        | Role                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------- |
| `SNAP_DATE`   | Which temporal snapshot this row belongs to (`2020-09-01`, `2020-09-08`, or `2020-09-15`) |
| `CUSTOMER_ID` | Customer identifier                                                                       |
| `ARTICLE_ID`  | Article identifier                                                                        |
| `SOLD`        | Target: `1` if the customer bought the article in the row’s label period; `0` otherwise   |
| `CUST_`*      | Customer-level aggregated predictors                                                      |
| `ART_*`       | Article-level aggregated predictors                                                       |
| `CUSTART_*`   | Customer–article cross predictors                                                         |


The author states that **article information**, **customer information**, and **customer transaction history** are all joined onto each row in Snowflake before export. The notebook does not list every joined column name; it only references a subset by name when imputing or plotting (see §3).

### 2.3 Snap dates and label windows

Each row answers one question: **“Given what we know on date X, will this customer buy this article in the next ~7 days?”**  
The datamart builds **three independent copies** of that question (one per snap date) and stacks the first two for training.

#### Big picture


| Concept            | Meaning                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| **Snap date**      | The “as of” date — all features use history **up to** this date only          |
| **Label window**   | The calendar week **after** the snap — purchases here set `SOLD = 1`          |
| **One row**        | One `(customer_id, article_id)` pair at one snap date, with `SOLD = 1` or `0` |
| **Multiple snaps** | More training examples — same prediction task, different weeks                |
| **Test snap**      | Same pattern, but the label week is the one used to evaluate                  |


This is **not** a chain where “week 1 features predict week 3.” Each snap date has its **own** aligned pair: `features@snap → label in the next week after that snap`.

#### What is a snap date?

Think of a **photo taken on a specific date**. On that date you freeze the world:

- How often has this customer shopped? → `CUST_`*
- How popular is this article? → `ART_*`
- Did this customer buy this article before? → `CUSTART_*`

All of that uses transactions **on or before the snap date only** — nothing from the future goes into features. Then you look **forward** and ask: did this customer buy this article in the **label window** after the snap? That answer is `SOLD`.

```text
Past history          SNAP DATE          Label window (future)
|---------------------|●|------------------|
  used for features   "as of"            SOLD = 1 if purchase here
```

**Example:** Snap `2020-09-08`, customer Alice, article Red Dress. Features use Alice’s history through Sept 8 and dress sales through Sept 8. Label window: `2020-09-09`–`2020-09-15`. If Alice bought the dress on Sept 12 → `SOLD = 1`.

#### Timeline for all three snaps

The author assigns three weekly snap dates:


| Snap date    | Author’s stated purpose | Features use history up to | Label window (`SOLD = 1` if purchase here) |
| ------------ | ----------------------- | -------------------------- | ------------------------------------------ |
| `2020-09-01` | Training                | Sept 1                     | `2020-09-02`–`2020-09-08`                  |
| `2020-09-08` | Training                | Sept 8                     | `2020-09-09`–`2020-09-15`                  |
| `2020-09-15` | Testing                 | Sept 15                    | `2020-09-15`–`2020-09-22`                  |


```text
Aug         Sep 1    Sep 8    Sep 15   Sep 22
|-----------|--------|--------|--------|
            ●        ●        ●
            snap1    snap2    snap3
            |←label1→|        |
                     |←label2→|
                              |←label3 (test)→|
```

The author does **not** document in the notebook which snap date maps to which training label week. The table above is the reasonable reading — consistent with the author’s snap-date list and combined training label ranges (`2020-09-02`–`2020-09-08` **and** `2020-09-09`–`2020-09-15`). The notebook does not show SQL or row-level checks that confirm it.

flowchart LR
  subgraph T1["Training snap 1 (Sept 1)"]
    F1["Features ≤ Sept 1"] --> L1["Label: Sept 2–8"]
  end
  subgraph T2["Training snap 2 (Sept 8)"]
    F2["Features ≤ Sept 8"] --> L2["Label: Sept 9–15"]
  end
  subgraph Test["Test snap (Sept 15)"]
    F3["Features ≤ Sept 15"] --> L3["Label: Sept 15–22"]
  end
  T1 --> Model["One XGBoost model"]
  T2 --> Model
  Test --> Metrics["Validation metrics"]


#### How labels are assigned

**Positive row (`SOLD = 1`)**

1. Pick a snap date (e.g. `2020-09-08`).
2. Find real purchases in **that snap’s** label window (`2020-09-09`–`2020-09-15`).
3. Each `(CUSTOMER_ID, ARTICLE_ID)` from those purchases → one row with `SOLD = 1`.
4. Attach features computed **as of that snap date**.

**Negative row (`SOLD = 0`)**

1. Pick random `(customer_id, article_id)`.
2. Confirm they **did not** buy together in **that snap’s** label window.
3. Same snap date, same feature cutoff, `SOLD = 0`.

**Two training snaps = two batches of rows**

Batch A — snap `2020-09-01`:


| CUSTOMER | ARTICLE | SNAP_DATE    | SOLD | Features from… | Label checks purchases in… |
| -------- | ------- | ------------ | ---- | -------------- | -------------------------- |
| Alice    | Dress   | `2020-09-01` | 1    | ≤ Sept 1       | Sept 2–8                   |
| Bob      | Shoes   | `2020-09-01` | 0    | ≤ Sept 1       | Sept 2–8                   |


Batch B — snap `2020-09-08`:


| CUSTOMER | ARTICLE | SNAP_DATE    | SOLD | Features from… | Label checks purchases in… |
| -------- | ------- | ------------ | ---- | -------------- | -------------------------- |
| Alice    | Dress   | `2020-09-08` | 1    | ≤ Sept 8       | Sept 9–15                  |
| Carol    | Hat     | `2020-09-08` | 0    | ≤ Sept 8       | Sept 9–15                  |


Both batches go into **one** XGBoost training set. The model learns one rule: `P(purchase in next week | current features)`. Each row already has the correct “next week” for **its** snap.

#### Why multiple snap dates?


| Reason                        | Explanation                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------- |
| **More examples**             | Two weeks of real purchases → ~~2× positive rows (~~496k total stated)          |
| **Same task, different time** | Each row is “predict next week from today’s features” — not a different problem |
| **Richer patterns**           | Features and seasonality differ between early Sept vs mid Sept                  |
| **Kaggle-style shortcut**     | Pre-build everything in Snowflake, export one CSV, train one model              |


**Wrong mental model:** three sequential calendar chunks where train week 1 must eventually predict week 3.  
**Correct mental model:** three independent “predict the upcoming week from today’s features” datasets; the first two are stacked for training; the third evaluates the same pattern one week later.

**Weather analogy:** a Monday photo predicts rain Tue–Mon; the next Monday photo predicts the **following** Tue–Mon. You do not ask “why didn’t Monday week 1 predict week 3?” — each Monday snapshot predicts **the week after that Monday**.

#### Validation set (notebook behavior)

There **is** a validation set, but the notebook uses it in a leaky way (see §2.8 for masks):


| Mask           | Snap dates                      | Used for                      |
| -------------- | ------------------------------- | ----------------------------- |
| **Training**   | Sept 1, Sept 8, **and Sept 15** | XGBoost `fit`                 |
| **Validation** | **Sept 15 only**                | Lift, Gini, precision@12, ROC |


The author calls `2020-09-15` the test snap in the datamart description, but the notebook **includes it in training** while also using it for validation. Treat reported validation metrics accordingly.

#### End-to-end story (Alice)

```text
Sept 1 snap:
  Features: Alice bought 5 items in August
  Label window: Sept 2–8
  Alice buys Dress on Sept 5 → row: (Alice, Dress, snap=Sept1, SOLD=1)

Sept 8 snap:
  Features: Alice bought 7 items total (includes Sept 5 dress)
  Label window: Sept 9–15
  Alice buys Dress again on Sept 12 → row: (Alice, Dress, snap=Sept8, SOLD=1)

Sept 15 snap (test):
  Features: Alice’s history through Sept 15
  Label window: Sept 15–22
  Used to score: “Will Alice buy X in the next week?”
```

The same customer and article can appear in **multiple rows** with **different snap dates**, **different features**, and **different label windows**. That is expected.

#### Quick reference


| Question                    | Answer                                                       |
| --------------------------- | ------------------------------------------------------------ |
| What is snap date?          | Feature cutoff — “what we knew on this date”                 |
| What is `SOLD`?             | Did they buy this pair in **this row’s** forward label week? |
| Why 2 training snaps?       | More labeled rows; same “predict next week” task             |
| Does week 1 predict week 3? | **No** — week 1 snap predicts Sept 2–8 only                  |
| What predicts “week 3”?     | Only **Sept 15 snap** rows (Sept 15–22 labels)               |


**Features vs labels:** Predictors (`CUST_*`, `ART_*`, `CUSTART_*`) are aggregations such as “number bought last month” and “days since last purchase” computed from history **up to the snap date**. The `SOLD` flag looks **forward** into the label week after that snap. The notebook never prints feature cutoff logic; the above follows the author’s prose in the opening cell.

### 2.4 Positive rows (`SOLD = 1`)

For each snap date, the author builds **positive rows from real transactions**:

1. Take transactions that occurred in that row’s label window.
2. Each resulting `(CUSTOMER_ID, ARTICLE_ID)` pair becomes one row with `SOLD = 1`.
3. Attach customer, article, and cross features computed as of the snap date.

**Training positive count (author-stated):** **495,774** rows with `SOLD = 1` across the two training label periods (`2020-09-02`–`2020-09-08` and `2020-09-09`–`2020-09-15`). The notebook does not break this total down by snap date or verify it after load.

The test split is described as **similar to train, shifted forward** for `2020-09-15`–`2020-09-22`. The author does not give a test positive count.

### 2.5 Negative rows (`SOLD = 0`)

The author adds **random non-purchase pairs**:

- **4,000,000** rows where `customer_id` and `article_id` are chosen at random.
- The pair **was not sold** in the relevant label window (author’s words: “combinations … that were not sold”).
- Same feature joins as positives: article info, customer info, and transaction-history aggregates on the row.

The notebook does **not** specify:

- Whether negatives are drawn per snap date or pooled across training snaps.
- The exact sampling procedure (uniform over all customers and articles, rejection sampling, fixed ratio per positive, etc.).
- Whether the test snap (`2020-09-15`) has its own negative sample or a separate count.

So the only hard facts are: **4,000,000** random `(customer, article)` pairs labeled `0` for training, with features joined the same way as positives.

### 2.6 Feature groups attached in Snowflake

The author groups predictors into three prefixes (computed in Snowflake, not in the notebook):


| Prefix     | Level              | What the author says is included                                                                                                          |
| ---------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `CUST_`    | Customer           | Aggregations over the customer’s history; includes **counts of purchases across product groups**                                          |
| `ART_`     | Article            | Aggregations over the article’s sales history (e.g. quantity sold, average price, channel counts — see §3 for names used in the notebook) |
| `CUSTART_` | Customer + article | Pair history: **last purchase**, **quantity purchased in the last month**, and similar cross metrics                                      |


Predictor types mentioned: **counts over time windows** (e.g. last month) and **recency** (e.g. days since last purchase). The notebook later treats some categoricals (e.g. `AGE`, and categoricals selected by dtype) as object columns for target encoding; it does not enumerate every `CUST_` / `ART_` column name in the build description.

### 2.7 Worked example (illustrative pair rows)

See §2.3 for the conceptual walkthrough (snap dates, label windows, Alice story). Below is a **concrete notebook-style example** with illustrative IDs.

**Setup:** Customer `0123456789`, article `0706016001`, snap date `2020-09-08`.

**Positive row**

1. Check whether customer `0123456789` bought article `0706016001` between `**2020-09-09` and `2020-09-15`** (label week for snap `2020-09-08`).
2. A purchase exists on `2020-09-12` → emit one row: `SNAP_DATE = 2020-09-08`, `CUSTOMER_ID = 0123456789`, `ARTICLE_ID = 0706016001`, `**SOLD = 1**`.
3. Attach features computed using transactions **on or before `2020-09-08`** only, for example:
  - `CUST_*`: how often this customer shops, including counts by product group.
  - `ART_*`: e.g. `ART_QUANTITY_SOLD_1M` (article’s store-wide sales in the last month), `ART_AVERAGE_PRICE`, `ART_NUM_CHANNEL_2`.
  - `CUSTART_*`: e.g. if they bought this article before, `CUSTART_QUANTITY_SOLD_1M` might be `2`; if never, those columns may be null in the CSV (the notebook later fills several CUSTART nulls with `0` — see §4).

**Negative row**

1. Draw random customer `0987654321` and random article `0198765003`.
2. Confirm they **did not** purchase together in the label window for that snap (`2020-09-09`–`2020-09-15` for snap `2020-09-08`).
3. Emit: same `SNAP_DATE`, `**SOLD = 0`**, with the same three feature groups joined. `CUSTART_*` fields are often **null or zero** because the pair has no history.

For snap `2020-09-01` and test snap `2020-09-15`, apply the same pattern with their respective label windows (§2.3 table). The notebook still includes `2020-09-15` rows in `**train_mask`** when fitting XGBoost (§2.8).

**After export — notebook-only steps**

The CSV is not used raw. Before modeling, the notebook:

1. Sets nulls to **0** on seven named columns (`CUSTART_QUANTITY_SOLD_1M`, `CUSTART_QUANTITY_SOLD_3M`, `CUSTART_QUANTITY_SOLD_12M`, `CUSTART_QUANTITY_SOLD_OVERALL`, `CUSTART_NUM_CHANNEL_2`, `ART_AVERAGE_PRICE`, `ART_NUM_CHANNEL_2`).
2. **Drops every row** where `ART_QUANTITY_SOLD_1M` is not greater than zero (articles with no store sales in the last month).
3. Applies train/validation masks by `SNAP_DATE`.

The notebook has **no saved outputs** for row counts after step 2, so the post-filter size is unknown from the checked-in file.

### 2.8 Training vs test rows in one file

All snaps live in **one** CSV. The notebook separates them only when fitting and evaluating (see §2.3 for why each snap exists):


| Mask                      | `SNAP_DATE` values included                  | Use                           |
| ------------------------- | -------------------------------------------- | ----------------------------- |
| Training (`train_mask`)   | `2020-09-01`, `2020-09-08`, `**2020-09-15`** | XGBoost `fit`                 |
| Validation (`valid_mask`) | `**2020-09-15` only**                        | Lift, Gini, precision@12, ROC |


So `**2020-09-15` rows are both trained on and used as validation**. The author calls `2020-09-15` the test snap in the datamart description, but the notebook does not hold it out of training. Treat reported validation metrics accordingly.

### 2.9 Author-stated class balance (training description)


| Label      | Author-stated count | Meaning                                          |
| ---------- | ------------------- | ------------------------------------------------ |
| `SOLD = 1` | 495,774             | Real purchases in the two training label windows |
| `SOLD = 0` | 4,000,000           | Random non-purchase pairs                        |


Rough ratio **1 : 8** positives to negatives in the author’s training description. The notebook fits XGBoost **without** `scale_pos_weight` (§6). After the `ART_QUANTITY_SOLD_1M > 0` filter and inclusion of `2020-09-15` in `train_mask`, the actual ratio in `fit` may differ; the notebook does not print `value_counts` in saved outputs.

### 2.10 What is not documented in the notebook


| Topic                                                             | Status                                                                                      |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Snowflake SQL or table names                                      | Not in repo                                                                                 |
| Full column list for `model_build_base.csv`                       | Not printed (`df.head()` has no saved output)                                               |
| Negative sampling algorithm                                       | Only “randomly chosen combinations … not sold”                                              |
| Per-snap row counts                                               | Not given except training totals above                                                      |
| Test-set negative count                                           | Not given                                                                                   |
| Whether one purchase generates one row or one row per transaction | Not stated (495,774 described as “rows of SOLD=1” tied to transactions sold in the windows) |


For v1’s window-aware negative sampling and temporal splits, see `[ranking-model-training-guide.md](./ranking-model-training-guide.md)` §4.

---

## 3. Feature columns referenced in the notebook

The Snowflake build is only summarized in prose (§2). The notebook **names** these columns when imputing, filtering, excluding, or plotting:

### 3.1 Customer level (`CUST_*`)

The author states that customer predictors include purchase counts and **purchases across product groups**. No individual `CUST_*` column names appear in the notebook source cells.

### 3.2 Article level (`ART_*`)


| Column                          | How the notebook uses it                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| `ART_QUANTITY_SOLD_1M`          | **Filter:** rows kept only if this value is greater than zero                          |
| `ART_AVERAGE_PRICE`             | Nulls filled with `0` before modeling                                                  |
| `ART_NUM_CHANNEL_2`             | Nulls filled with `0` before modeling                                                  |
| `ART_DAYS_SINCE_FIRST_PURCHASE` | Present in `col_exclude` list as a commented-out entry (not excluded in the run shown) |
| `ART_DAYS_SINCE_LAST_PURCHASE`  | Same as above                                                                          |


### 3.3 Customer–article cross level (`CUSTART_*`)

The author credits these with the largest competition score gain (~0.007 → ~0.0247 MAP@12). Columns **explicitly named** in the notebook:


| Column                          | Notebook treatment |
| ------------------------------- | ------------------ |
| `CUSTART_QUANTITY_SOLD_1M`      | Null → `0`         |
| `CUSTART_QUANTITY_SOLD_3M`      | Null → `0`         |
| `CUSTART_QUANTITY_SOLD_12M`     | Null → `0`         |
| `CUSTART_QUANTITY_SOLD_OVERALL` | Null → `0`         |
| `CUSTART_NUM_CHANNEL_2`         | Null → `0`         |


The author’s build description also mentions **last purchase** and **number purchased last month** at the CUSTART level; those map to the quantity/recency family above but exact column names beyond this list are not shown in the notebook.

### 3.4 Other columns used later


| Column               | Context                                                                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `AGE`                | SHAP dependence plot with `ART_AVERAGE_PRICE`                                                                                             |
| `PRODUCT_TYPE_NAME`  | Mentioned in candidate-generation prose for inference (§8)                                                                                |
| Object-dtype columns | Selected automatically as categorical predictors (`cat_preds`) for target encoding; the notebook does not print the list in saved outputs |


CUSTART columns are **included** in the model. A commented line in the notebook shows an optional experiment to exclude all `CUSTART` columns; that path is **not** used in the saved workflow.

---

## 4. Notebook steps after the CSV is loaded

These steps happen **in Python** after Snowflake export. They are the only training-data transformations visible in the repository.

### 4.1 Load

The notebook reads `model_build_base.csv` from the Kaggle dataset path `/kaggle/input/hm-model-build-base/model_build_base.csv`. A commented line shows an optional `sample(100000)` that is **not** enabled.

### 4.2 Null imputation

Seven columns with missing values are filled with **zero** before any split or model code runs:

- `CUSTART_QUANTITY_SOLD_1M`, `CUSTART_QUANTITY_SOLD_3M`, `CUSTART_QUANTITY_SOLD_12M`, `CUSTART_QUANTITY_SOLD_OVERALL`, `CUSTART_NUM_CHANNEL_2`
- `ART_AVERAGE_PRICE`, `ART_NUM_CHANNEL_2`

This matches pairs or articles with no history in the source aggregates (see §2.7 negative example).

### 4.3 Active-article filter

The notebook keeps only rows where `**ART_QUANTITY_SOLD_1M` is greater than zero**. Rows for articles with no store-wide sales in the last month are removed entirely. The checked-in notebook has **no saved output** for `len(df)` after this step, so the remaining row count is unknown.

### 4.4 Train and validation masks

Rows are subset by `SNAP_DATE`:


| Mask           | Included snap dates                      |
| -------------- | ---------------------------------------- |
| **Training**   | `2020-09-01`, `2020-09-08`, `2020-09-15` |
| **Validation** | `2020-09-15` only                        |


XGBoost is fit on the training mask. Lift, Gini, accuracy, precision@12, and ROC use the validation mask. Because `2020-09-15` appears in both, validation rows are **not** held out of training.

The notebook calls `value_counts` on `SOLD` for each mask but does **not** save those outputs in the checked-in file.

### 4.5 Columns excluded from model inputs

The target column is `**SOLD`**. These columns are excluded from predictors (`col_exclude`):

- `SNAP_DATE`
- `CUSTOMER_ID`
- `ARTICLE_ID`
- `SOLD`

`ART_DAYS_SINCE_FIRST_PURCHASE` and `ART_DAYS_SINCE_LAST_PURCHASE` appear in the exclude list only as **commented** entries, so they remain available as numeric predictors in the run shown. A similar commented block would have excluded all `CUSTART` columns; that exclusion is **not** applied.

Identifiers and snap date never enter XGBoost. All numeric columns not in `col_exclude` (including CUSTART fields) are eligible predictors after encoding (§5).

---

## 5. Categorical encoding

Object columns (excluding `col_exclude`) are **Bayesian target-encoded** with `[category_encoders](https://contrib.scikit-learn.org/category_encoders/)`:

```python
import category_encoders as ce

cat_preds = [c for c in df.select_dtypes("object") if c not in col_exclude]

encoder = ce.TargetEncoder(min_samples_leaf=1, smoothing=1.0)
encoder.fit_transform(df[train_mask][cat_preds], df[train_mask][col_target])

df = pd.concat(
    [df, encoder.transform(df[cat_preds]).add_prefix("BAYES_")],
    axis=1,
)
```


| Parameter          | Value    | Effect                                                 |
| ------------------ | -------- | ------------------------------------------------------ |
| `min_samples_leaf` | `1`      | Allow encoding for rare categories                     |
| `smoothing`        | `1.0`    | Pull estimates toward global mean for low-count levels |
| Output prefix      | `BAYES_` | Encoded columns appended alongside originals           |


Final predictors are **all int/float columns** not in `col_exclude`:

```python
col_preds = [c for c in df.select_dtypes(["int", "float"]) if c not in col_exclude]
```

v1 passes high-cardinality categoricals to XGBoost as native `cat_features` instead of target encoding — see `[ranking-model-training-guide.md](./ranking-model-training-guide.md)` §5.

---

## 6. Model training

### 6.1 Algorithm and hyperparameters

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    max_depth=4,
    seed=1234,
    colsample_bytree=0.5,
    gamma=1,
    min_child_weight=5,
    n_estimators=100,
)

xgb_model.fit(
    df[train_mask][col_preds],
    df[train_mask][col_target],
    verbose=0,
    eval_metric="logloss",
)
```


| Parameter          | Value     | Role                                                  |
| ------------------ | --------- | ----------------------------------------------------- |
| `max_depth`        | 4         | Shallow trees — limits overfit on noisy negatives     |
| `colsample_bytree` | 0.5       | Feature subsampling per tree                          |
| `gamma`            | 1         | Minimum loss reduction to split                       |
| `min_child_weight` | 5         | Minimum sum of instance weight in a child             |
| `n_estimators`     | 100       | Fixed boosting rounds (no early stopping in notebook) |
| `eval_metric`      | `logloss` | Binary cross-entropy                                  |


**Class imbalance:** ~~495k positives vs 4M negatives (~~1:8). The notebook does **not** set `scale_pos_weight`; v1 explicitly sets `scale_pos_weight = 5` for its 1:5 sampling ratio.

### 6.2 Scoring

```python
df["XGB_SCORE"] = xgb_model.predict_proba(df[col_preds])[:, 1]
```

`XGB_SCORE` is the estimated `P(SOLD=1)` used for ranking candidates.

### 6.3 Saved artifacts

```python
pickle.dump(xgb_model, open("hm_xgb_model.pkl", "wb"))
pickle.dump(encoder, open("hm_encoder.pkl", "wb"))
```

The inference section loads a **full** variant trained with all CUSTART features:

- `hm_xgb_model_custart_full.pkl`
- `hm_encoder_custart_full.pkl`

---

## 7. Evaluation (notebook metrics)

The notebook uses the **[mofr](https://github.com/Vrboska/mofr)** library (`pip install git+https://github.com/Vrboska/mofr@master`) for ranking-style diagnostics.

### 7.1 Pair-level metrics (train and valid)

```python
import mofr

mofr.metrics.lift(y_true, y_score)
mofr.metrics.gini(y_true, y_score)
mofr.metrics.accuracy_score(y_true, (y_score > 0.5).astype(int))
```


| Metric             | Interpretation                                        |
| ------------------ | ----------------------------------------------------- |
| **Lift**           | How much better than random the scores rank positives |
| **Gini**           | Normalized rank correlation between score and label   |
| **Accuracy @ 0.5** | Threshold classification — weak for heavy imbalance   |


### 7.2 List-level precision @ 12 (valid)

Simulates competition serving on a sample of customers:

```python
random_customers = np.random.choice(df[valid_mask]["CUSTOMER_ID"].unique(), size=1000)
top12 = (
    df[valid_mask]
    .merge(random_customers)
    .groupby("CUSTOMER_ID")
    .apply(lambda x: x.sort_values("XGB_SCORE", ascending=False).head(12))
)
mean_precision = top12.groupby("CUSTOMER_ID")["SOLD"].mean().mean()
```

This approximates **MAP@12** behavior: for each customer, precision of the top-12 ranked list.

### 7.3 ROC curve

```python
from mofr.basic_evaluators.ROCCurve import ROCCurveEvaluator

rce = ROCCurveEvaluator()
rce.d(df[valid_mask]).t([(col_target, "one")]).s(["XGB_SCORE"])
rce.get_graph()
```

### 7.4 Feature importance and SHAP

**Gain-based importance** from XGBoost:

```python
sorted_idx = xgb_model.feature_importances_.argsort()
# horizontal bar chart of col_preds[sorted_idx]
```

**SHAP (TreeExplainer)** for local and global explanations:

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(df[train_mask][col_preds])
shap.summary_plot(shap_values, df[train_mask][col_preds])
shap.dependence_plot("ART_AVERAGE_PRICE", shap_values, ..., interaction_index="AGE")
```

Use SHAP to verify that CUSTART and recency features dominate, and to inspect price × age interactions.

---

## 8. Inference and submission pipeline

Training rows are dropped (`del df`) before inference. A **pre-trained full model** and **pre-built candidate file** are loaded to score at scale.

### 8.1 Side tables

```python
articles = pd.read_csv("articles_predictions.csv")
customers = pd.read_csv("customers_prediction.csv").fillna(0)
```

Join keys are normalized:

```python
customers["CUSTOMER_ID10"] = customers["CUSTOMER_ID"].str[:10]
articles["ARTICLE_ID6"] = articles["ARTICLE_ID"].astype(str).str[:6].astype(int)
```

### 8.2 Candidate generation (upstream of the notebook)

For each customer, the author builds up to **500 candidate articles** from a **blend** of lists (computed outside this notebook, stored in `model_suggested_items_enriched_sorted.csv`):


| Source                    | Size    | Description                                                  |
| ------------------------- | ------- | ------------------------------------------------------------ |
| Customer last month       | Top 100 | Articles the customer bought most in the last month          |
| Customer overall          | Top 100 | Customer lifetime top articles                               |
| Global last month         | Top 100 | Most popular articles store-wide last month                  |
| Customer top product type | Top 100 | Popular items in the customer's dominant `PRODUCT_TYPE_NAME` |


The candidate file is **enriched with CUSTART predictors** before scoring. Only a sample of that file ships with the Kaggle dataset; the full file is built in Snowflake.

### 8.3 Batch scoring function

```python
def process_chunk(chunk):
    chunk = chunk.rename(columns={"CUSTOMER_ID": "CUSTOMER_ID10"}).drop_duplicates()
    chunk = chunk.merge(articles, how="left", on="ARTICLE_ID")
    chunk = chunk.merge(customers, how="left", on="CUSTOMER_ID10")
    chunk = pd.concat(
        [chunk, encoder.transform(chunk[cat_preds]).add_prefix("BAYES_")],
        axis=1,
    )
    chunk["XGB_SCORE"] = xgb_model.predict_proba(
        chunk[xgb_model.feature_names_in_]
    )[:, 1]
    chunk["ARTICLE_ID"] = chunk["ARTICLE_ID"].astype(str).str.zfill(10)

    top12 = (
        chunk.groupby("CUSTOMER_ID")
        .apply(lambda x: x.sort_values("XGB_SCORE", ascending=False).head(12))
        .reset_index(drop=True)
    )
    return (
        top12.groupby("CUSTOMER_ID")["ARTICLE_ID"]
        .apply(list)
        .apply(" ".join)
        .reset_index(name="PREDICTION")
    )
```

### 8.4 Chunked read and submission

```python
submission = pd.DataFrame()
with pd.read_csv("model_suggested_items_enriched_sorted.csv", chunksize=10**6) as reader:
    for chunk in reader:
        submission = pd.concat([submission, process_chunk(chunk)])

submission.drop_duplicates(subset="CUSTOMER_ID", keep="first", inplace=True)
submission.to_csv("submission.csv", index=False)
```

Output schema: `CUSTOMER_ID`, `PREDICTION` (space-separated 12 zero-padded article IDs).

### 8.5 Reported competition result


| Stage                                   | Approx. MAP@12                             |
| --------------------------------------- | ------------------------------------------ |
| Without CUSTART features                | ~0.007                                     |
| With CUSTART features + candidate blend | **~0.0247** (bronze-medal range on Kaggle) |


Achieved without GPU or a dedicated training cluster — tabular features + XGBoost + sensible candidate generation.

---

## 9. Dependencies


| Package                 | Purpose                    |
| ----------------------- | -------------------------- |
| `pandas`, `numpy`       | Data frames                |
| `xgboost`               | `XGBClassifier`            |
| `category_encoders`     | Bayesian target encoding   |
| `mofr`                  | Lift, Gini, ROC evaluators |
| `shap`                  | Explainability             |
| `matplotlib`, `seaborn` | Plots                      |


Install mofr as in the notebook:

```bash
pip install git+https://github.com/Vrboska/mofr@master
```

---

## 10. Reproducing locally

1. **Training table** — Obtain `model_build_base.csv` from the author’s Kaggle dataset (`hm-model-build-base`). The Snowflake build steps are **not** in this repo; §2 documents them from the notebook’s opening cell only.
2. **Open notebook** — `[notebooks/example-quick-and-easy-model-build.ipynb](../../../notebooks/example-quick-and-easy-model-build.ipynb)`.
3. **Replace paths** — Point the read path at your local copy of `model_build_base.csv` instead of `/kaggle/input/hm-model-build-base/...`.
4. **Run training cells** — Load, impute, filter, encode, fit, and metrics (§4–§7). SHAP on the full training mask can be slow.
5. **Inference** — Requires `model_suggested_items_enriched_sorted.csv`, `articles_predictions.csv`, and `customers_prediction.csv`; skip if studying training only.

For v1-aligned ranker training, use `[ranking-model-training-guide.md](./ranking-model-training-guide.md)` instead.

---

## 11. Design lessons (why it works)


| Idea                                            | Benefit                                                                        |
| ----------------------------------------------- | ------------------------------------------------------------------------------ |
| **Three-level features (CUST / ART / CUSTART)** | Separates user taste, item popularity, and repeat-purchase signal              |
| **CUSTART cross features**                      | Strong signal for fashion repurchase; largest score jump in this solution      |
| **Candidate generation before scoring**         | Keeps inference tractable — score hundreds of pairs per user, not full catalog |
| **Shallow XGBoost (`max_depth=4`)**             | Robust to label noise from random negatives                                    |
| **Target encoding**                             | Handles high-cardinality categoricals in a single pass                         |
| **Batch chunked scoring**                       | Scales to millions of candidate rows on CPU                                    |


**Caveats for production:**

- Random global negatives ≠ realistic "did not buy" distribution.
- Snap-date split overlap weakens valid-set claims.
- Target encoding on full data at inference requires careful leakage control (fit encoder on train only — the notebook does this correctly).
- No retrieval stage — candidate rules encode most of the recall; the model mainly re-ranks.

---

## 12. Reference vs v1 summary


| Topic                 | This notebook (Kaggle reference)              | v1 (`[ranking-model-training-guide.md](./ranking-model-training-guide.md)`) |
| --------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| **Architecture**      | Single XGBoost ranker + rule-based candidates | Two-tower retrieval (~100) → filter → XGBoost ranker → diversity → top-10   |
| **Train dates**       | Snap dates Sep 2020 (competition-tail)        | FR-BATCH-02: train → 2020-03-31, val/test through 2020-06-30                |
| **Positives**         | Sold rows in competition windows              | Purchases in split label window                                             |
| **Negatives**         | 4M random cross-pairs (~1:8)                  | 5 window-aware negatives per positive (`scale_pos_weight=5`)                |
| **Categoricals**      | Bayesian target encoding (`BAYES_`*)          | Native XGBoost `cat_features`                                               |
| **Class weight**      | Not set                                       | `scale_pos_weight = 5`                                                      |
| **Serving list size** | Top **12** (competition)                      | Top **10** (v1 requirements)                                                |
| **Eval gate**         | Lift, Gini, precision@12 sample               | AUC-PR, `hit_rate@10`, `recall@100`                                         |
| **Platform**          | Snowflake + Kaggle + pickle                   | Glue features → SageMaker Pipeline → Model Registry                         |
| **Explainability**    | SHAP + mofr ROC                               | Feature importance + Model Monitor                                          |


Use this notebook to learn **pairwise tabular ranking** and **candidate + rerank** patterns. Implement v1 per the contract documents, not by copying snap dates or negative sampling from the competition notebook.

---

## 13. Source file map


| Topic                  | Location                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| End-to-end notebook    | `[notebooks/example-quick-and-easy-model-build.ipynb](../../../notebooks/example-quick-and-easy-model-build.ipynb)` |
| v1 ranker contract     | `[ranking-model-training-guide.md](./ranking-model-training-guide.md)`                                              |
| v1 feature definitions | `[features-eng.md](./features-eng.md)`                                                                              |
| H&M schema             | `[schema-info.md](../../system-design/schema-info.md)`                                                              |
| v1 requirements        | `[v1-requirements.md](../../system-design/v1/v1-requirements.md)`                                                   |


