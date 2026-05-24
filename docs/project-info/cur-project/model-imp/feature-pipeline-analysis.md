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
- 90% probability of clicks before purchases
- 40-60 random ignore interactions per customer
- 5-8 additional clicks on non-purchased items
- Time-based interaction sequencing

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
- Lazy evaluation for memory efficiency
- Vectorized operations for speed
- Efficient joins and aggregations

### Batch Processing
- Embedding generation in configurable batches (default: 128)
- Customer processing in chunks (1000 customers)
- Progress tracking with tqdm

### Memory Management
- Selective column retention
- Early data filtering and sampling
- Efficient data type usage

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