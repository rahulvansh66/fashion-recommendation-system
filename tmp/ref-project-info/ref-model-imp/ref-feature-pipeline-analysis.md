---
⚠️ **REFERENCE PROJECT DISCLAIMER** ⚠️

**THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION**

- **DO NOT USE** unless explicitly asked to reference old code
- **CURRENT IMPLEMENTATION** is in `system-design/` directory
- This file is for **REFERENCE ONLY** to understand legacy approaches
- All new development should follow current system design specifications

---

# Feature Pipeline Analysis

## Overview
This document analyzes the feature pipeline implementation for the modern fashion recommendation system, based on code from the Jupyter notebook `1_fp_computing_features.ipynb` and supporting Python modules in `recsys/features/`.

## Data Preprocessing Workflow

### Core Libraries and Framework
The feature pipeline primarily uses **Polars** for high-performance data processing, with some Pandas integration for specific operations:

```python
import polars as pl
import pandas as pd
from sentence_transformers import SentenceTransformer
```

### Data Loading and Initial Processing

The pipeline processes three main datasets from the H&M fashion dataset:

1. **Articles Dataset** (105,542 records, 25 columns)
   - Product catalog with hierarchical classification
   - Contains product metadata, descriptions, and categorical attributes

2. **Customers Dataset** (1,371,980 records, 7 columns)  
   - Customer demographics and preferences
   - Includes age, club membership status, and postal codes

3. **Transactions Dataset** (31,788,324 records, 5 columns)
   - Purchase transaction history
   - Links customers to articles with timestamps and prices

### Dataset Sampling Strategy

For performance optimization, the pipeline implements customer-based sampling:

```python
class DatasetSampler:
    _SIZES = {
        CustomerDatasetSize.LARGE: 50_000,
        CustomerDatasetSize.MEDIUM: 5_000,
        CustomerDatasetSize.SMALL: 1_000,
    }
```

The sampling process:
1. Randomly selects N customers from the full dataset
2. Filters transactions to only include selected customers
3. Reduces dataset from 31M+ transactions to manageable size (e.g., 23,799 for SMALL)

## Feature Engineering Techniques

### 1. Articles Feature Engineering

**Text Feature Creation**: Generates rich article descriptions by combining multiple categorical fields:

```python
def create_article_description(row):
    description = f"{row['prod_name']} - {row['product_type_name']} in {row['product_group_name']}"
    description += f"\nAppearance: {row['graphical_appearance_name']}"
    description += f"\nColor: {row['perceived_colour_value_name']} {row['perceived_colour_master_name']} ({row['colour_group_name']})"
    description += f"\nCategory: {row['index_group_name']} - {row['section_name']} - {row['garment_group_name']}"
    
    if row["detail_desc"]:
        description += f"\nDetails: {row['detail_desc']}"
    
    return description
```

**Semantic Embeddings**: Uses SentenceTransformer to create vector representations:

```python
def generate_embeddings_for_dataframe(df, text_column, model, batch_size=32):
    # Batch processing with progress tracking
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts, device=model.device)
        all_embeddings.extend(batch_embeddings.tolist())
```

**Additional Features**:
- Product name length calculation
- Image URL construction from article IDs
- Data type conversions (article_id to string)

### 2. Customer Feature Engineering

**Age Group Categorization**: Creates categorical age buckets for better model performance:

```python
def create_age_group():
    return (
        pl.when(pl.col("age").is_between(0, 18)).then(pl.lit("0-18"))
        .when(pl.col("age").is_between(19, 25)).then(pl.lit("19-25"))
        .when(pl.col("age").is_between(26, 35)).then(pl.lit("26-35"))
        .when(pl.col("age").is_between(36, 45)).then(pl.lit("36-45"))
        .when(pl.col("age").is_between(46, 55)).then(pl.lit("46-55"))
        .when(pl.col("age").is_between(56, 65)).then(pl.lit("56-65"))
        .otherwise(pl.lit("66+"))
    ).alias("age_group")
```

**Data Cleaning**:
- Fills missing club member status with 'ABSENT'
- Drops rows with null age values
- Validates required columns presence

### 3. Transaction Feature Engineering

**Temporal Feature Extraction**: Derives multiple time-based features:

```python
def compute_features_transactions(df):
    return df.with_columns([
        pl.col("t_dat").dt.year().alias("year"),
        pl.col("t_dat").dt.month().alias("month"),
        pl.col("t_dat").dt.day().alias("day"),
        pl.col("t_dat").dt.weekday().alias("day_of_week"),
    ])
```

**Cyclical Time Encoding**: Implements sine/cosine encoding for months to capture seasonality:

```python
@udf(return_type=float, mode="pandas")
def month_sin(month: pd.Series):
    return np.sin(month * (2 * np.pi / 12))

@udf(return_type=float, mode="pandas") 
def month_cos(month: pd.Series):
    return np.cos(month * (2 * np.pi / 12))
```

**Timestamp Conversion**: Converts datetime to epoch milliseconds for model consumption:

```python
(pl.col("t_dat").cast(pl.Int64) // 1_000_000).alias("t_dat")
```

### 4. Interaction Data Generation

**Synthetic Interaction Creation**: Generates comprehensive user-item interactions with multiple interaction types:

```python
def generate_interaction_data(trans_df):
    # Interaction scores:
    # 0: No interaction (ignored items)
    # 1: Click interaction 
    # 2: Purchase interaction
```

**Realistic Interaction Patterns**:

Ignore Events (40–60 per customer)

- For each customer, 40–60 random articles are drawn from the full catalog (not just items they bought).
- Each ignored article gets 1 or 2 separate ignore events at slightly different times (1–12 hours apart), making the data look like a user scrolled past an item in two separate browsing sessions.
- Timestamps are placed 1–96 hours before the customer's last purchase — ensuring ignores are chronologically plausible browsing activity.
- This is the largest signal category and produces the 0 (73,710) distribution seen in the final output.

Step 2 — Pre-Purchase Clicks (90% probability)

- For 90% of purchased items, 1–2 click events are inserted 1–48 hours before the actual purchase timestamp.
- This models realistic shopping behavior: users browse and click on a product detail page before committing to buy.
- The 10% without pre-clicks represent impulse purchases — also realistic.
- The real purchase event is always recorded as score 2 regardless.

Step 3 — Extra Exploratory Clicks (95% probability, 5–8 per customer)

- 5–8 additional clicks are added on items the customer did not buy and has not yet interacted with.
- The exclusion set (purchased ∪ clicked ∪ ignored) prevents duplicate interactions on the same article.
- These simulate "window shopping" — browsing items with some interest but not enough to buy.
- Timestamps are 1–72 hours before the last purchase.
- 95% probability means nearly every customer gets these extra exploratory signals.

Step 4 — Time-Based Sequencing & prev_article_id

- All interactions are sorted by (customer_id, timestamp), establishing a chronological browsing session per user.
- A prev_article_id column is created using a window shift — each row records the article the customer interacted with immediately before. The first interaction gets "START".
- This sequential context is fed into the two-tower model so it can learn session-aware patterns (e.g., "after clicking a blue jacket, this customer tends to click trousers").

## Configuration Management

The pipeline uses a Pydantic-based configuration system defined in `recsys/config.py`:

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `CUSTOMER_DATA_SIZE` | CustomerDatasetSize | SMALL | Dataset sampling size |
| `FEATURES_EMBEDDING_MODEL_ID` | str | "all-MiniLM-L6-v2" | SentenceTransformer model |
| `TWO_TOWER_MODEL_EMBEDDING_SIZE` | int | 16 | Embedding dimension |
| `TWO_TOWER_MODEL_BATCH_SIZE` | int | 2048 | Training batch size |
| `TWO_TOWER_NUM_EPOCHS` | int | 10 | Training epochs |
| `TWO_TOWER_LEARNING_RATE` | float | 0.01 | Learning rate |
| `RANKING_LEARNING_RATE` | float | 0.2 | Ranking model learning rate |
| `RANKING_ITERATIONS` | int | 100 | Ranking model iterations |
| `RANKING_SCALE_POS_WEIGHT` | int | 10 | Positive class weight |

## Data Validation and Quality

### Null Value Handling
- **Articles**: Removes columns with any null values, specifically drops 'detail_desc' column (416 null values)
- **Customers**: Fills missing club_member_status, optionally drops null ages
- **Transactions**: No explicit null handling (clean dataset)

### Data Type Conversions
- Article IDs: Integer to String for consistency
- Timestamps: Date strings to datetime objects, then to epoch milliseconds
- Age: Cast to Float64 for numerical processing

### Schema Validation
Customer processing includes required column validation:

```python
required_columns = ["customer_id", "club_member_status", "age", "postal_code"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Columns {', '.join(missing_columns)} not found")
```

## Performance Optimizations

### Polars Usage
- The feature pipeline standardizes on Polars DataFrames for the heavy tabular stages (`customers`, `articles`, `transactions`, `interactions`, and `ranking`) because the H&M source tables are large enough that Pandas-only processing would create avoidable memory pressure.
- Most feature transforms are expressed as Polars expressions inside `.with_columns()`, `.select()`, `.join()`, `.unique()`, `.sample()`, `.sort()`, and window operations. This keeps column operations vectorized and avoids Python row loops for deterministic transformations such as ID casting, age-group creation, date feature extraction, and previous-article lookup.
- Joins are kept narrow whenever possible:
  - Customer sampling joins only `customer_id` back into the transactions table, reducing the transaction set before downstream interaction generation.
  - Ranking data construction first joins transaction pairs with a reduced article/customer view, then joins the full item feature set only after positive and negative training pairs have been created.
  - Article feature retrieval for ranking uses `select_except(["article_description", "embeddings", "image_url"])` to avoid carrying large text, vector, and URL columns through joins that do not need them.
- Column projection is used as a memory optimization, not just schema cleanup. Customer features are reduced to `customer_id`, `club_member_status`, `age`, `postal_code`, and `age_group`; ranking query features are reduced to `customer_id`, `age`, and `article_id`; item ranking features are reduced to categorical product attributes.
- The interaction feature stage uses Polars filtering per customer chunk instead of repeatedly scanning the full transaction table for each customer. For each chunk, transactions are filtered once with `pl.col("customer_id").is_in(chunk_customers)`, then customer-level work happens against that smaller in-memory subset.
- Sequential recommendation context is computed with a Polars window expression:
  - The synthetic interaction table is sorted by `customer_id` and timestamp.
  - `article_id` is shifted within each `customer_id` group using `.shift(1).over("customer_id")`.
  - Missing first-event values are filled with `"START"`.
  This avoids a manual per-customer sequence-building pass after the interactions have already been generated.
- Data type conversions are performed before joins or feature-store writes to prevent expensive implicit casting later. In particular, `article_id` is consistently cast to `Utf8`, timestamps are converted to epoch milliseconds, and `age` is cast to `Float64` for model compatibility.

### Batch Processing
- Article text embeddings are generated in batches inside `generate_embeddings_for_dataframe()`. The implementation converts the selected text column to a Python list once, slices it by `batch_size`, and calls `SentenceTransformer.encode()` on each slice rather than encoding one article at a time.
- The embedding batch loop is designed to balance throughput and memory:
  - Larger batches improve model utilization by reducing per-call overhead.
  - Smaller batches reduce peak memory when embedding all 105K+ articles.
  - The function default is `batch_size=32`, while notebook/pipeline callers can override it for the available CPU/GPU memory. The project documentation references `128` as the typical configured batch size for faster full-catalog runs.
- Embedding progress is tracked with `tqdm(total=total_rows, desc="Generating embeddings")`, and the progress bar is updated by the actual number of texts processed in each batch. This makes long catalog embedding jobs observable without enabling the SentenceTransformer internal progress bar for every batch.
- SentenceTransformer stdout is temporarily suppressed during each encode call. This keeps notebook and terminal logs readable during long embedding jobs while preserving the outer progress bar as the primary status signal.
- Customer interaction generation processes customers in fixed chunks of `1000`. For each chunk:
  - The pipeline creates `chunk_customers` from the unique customer list.
  - It filters transactions once for those customers.
  - It iterates through the chunk to generate ignores, clicks, purchases, and previous-article context.
  This reduces repeated full-table scans and keeps intermediate transaction slices bounded.
- Training data creation also uses batching downstream:
  - Two-tower training defaults to `TWO_TOWER_MODEL_BATCH_SIZE=2048`.
  - Candidate datasets are batched before retrieval metric computation.
  - Validation/test datasets are batched and cached where appropriate.
  These batches are separate from feature-generation batches but serve the same goal: keep model input pipelines predictable and memory bounded.
- Ranking dataset negative sampling is vectorized in bulk. The pipeline creates `10x` as many negative pairs as positive pairs by sampling article IDs, customer IDs, and ages with replacement using deterministic seeds, then concatenates the positive and negative frames. This avoids generating negatives through a nested customer-item loop.

### Memory Management
- The pipeline deliberately reduces data volume early. `DatasetSampler` supports three dataset sizes:
  - `SMALL`: 1,000 customers
  - `MEDIUM`: 5,000 customers
  - `LARGE`: 50,000 customers
  After sampling customers, the transaction table is immediately inner-joined to the sampled customer IDs so all downstream stages operate only on relevant transactions.
- Article processing drops null-bearing columns after feature creation. This removes sparse fields such as `detail_desc` from the final feature group after the richer `article_description` text has already been constructed from it.
- Large columns are excluded when they are not needed:
  - `article_description` is needed for embeddings but not for ranking joins.
  - `embeddings` are needed for retrieval/item representation but not for classical ranking feature joins.
  - `image_url` is useful for UI display but unnecessary during ranking dataset construction.
- The ranking dataset is assembled in stages to control peak memory:
  - Read only transaction `article_id` and `customer_id`.
  - Read only customer `age`.
  - Read article metadata without embedding/text-heavy columns.
  - Create positive/negative query pairs.
  - Join the compact item feature frame at the end.
- Intermediate arrays are kept simple in the embedding path. The implementation accumulates `batch_embeddings.tolist()` into `all_embeddings`, then attaches the final list as a single Polars `Series`. This avoids repeatedly mutating the DataFrame inside the batch loop.
- Interaction generation stores synthetic events in a Python list and converts to a Polars DataFrame once at the end. That approach is faster than appending to a DataFrame repeatedly, but it also means interaction volume is controlled by customer sampling, bounded ignore/click counts, and the 1000-customer chunking strategy.
- Reproducibility also supports performance debugging. Customer sampling uses `random.seed(27)`, and ranking negative sampling uses fixed seeds (`2`, `3`, and `4`), so memory and runtime profiles are comparable across repeated runs with the same configuration.
- The main memory trade-off is that some stages still materialize full intermediate outputs:
  - Article embeddings materialize all article vectors before writing the feature group.
  - Interaction generation materializes all synthetic interactions before final sorting and previous-item calculation.
  - Ranking training data materializes a 10:1 negative-to-positive dataset.
  These choices are acceptable for the documented sampled runs, but full-scale production runs should consider streaming writes, partitioned feature generation, or larger external compute resources.

## Pipeline Integration with H&M Dataset

### Schema Alignment
The feature pipeline specifically handles H&M dataset characteristics:
- **Hierarchical Product Categories**: product_group_name → section_name → garment_group_name
- **Color Classification**: Multiple color attributes (perceived_colour_value_name, colour_group_name)
- **Hashed Identifiers**: Customer IDs are pre-hashed for privacy
- **Temporal Range**: Transactions span multiple years (2018-2020)

### Business Logic
- **Seasonal Fashion Patterns**: Month sine/cosine encoding captures fashion seasonality
- **Product Variants**: Article-level granularity handles size/color variations
- **Customer Segmentation**: Age group categorization aligns with fashion demographics

## Output Datasets

### Feature Groups Created
1. **Customers** (1,000 records) - Demographics with age groups
2. **Articles** (105,542 records) - Products with embeddings and metadata
3. **Transactions** (23,799 records) - Temporal purchase data
4. **Interactions** (135,813 records) - Multi-type user-item interactions
5. **Ranking** (224,136 records) - Training data for ranking model

### Final Data Distribution
- **Interaction Scores**: 0 (73,710), 1 (38,304), 2 (23,799)
- **Ranking Labels**: 0 (203,760), 1 (20,376) - 10:1 negative to positive ratio

This feature pipeline demonstrates sophisticated data preprocessing combining traditional feature engineering with modern embedding techniques, specifically tailored for fashion recommendation systems.
---
⚠️ **END OF REFERENCE PROJECT FILE** ⚠️

Remember: This is archived code. Use `system-design/` for current implementation.

---
