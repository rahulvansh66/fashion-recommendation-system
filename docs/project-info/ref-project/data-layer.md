# Data Layer: AWS-Native Data Lake Architecture

## Overview

**Philosophy:** Build a serverless, cost-efficient data lake leveraging AWS managed services for reliable data processing without infrastructure overhead.

**Core Strategy:** S3-based data lake with Glue ETL pipelines, Parquet optimization, and Athena querying for end-to-end data lifecycle from raw ingestion to ML-ready features.

**Key Advantage:** Production-grade data architecture with managed services eliminates traditional data warehouse operational complexity while maintaining scalability.

## Storage Architecture

### S3 Data Lake Structure

**Design Principle:** Organize data by processing stage, enabling clear separation of concerns and efficient access patterns.

```
fashion-recommender-data-[environment]/
├── raw/                          # Original H&M CSV files (immutable)
│   ├── articles/
│   │   └── articles.csv
│   ├── customers/
│   │   └── customers.csv
│   └── transactions/
│       └── transactions_train.csv
│
├── processed/                     # Cleaned, validated data (Parquet format)
│   ├── articles/
│   │   └── year=2024/month=05/day=24/part-00000.parquet
│   ├── customers/
│   │   └── year=2024/month=05/day=24/part-00000.parquet
│   └── transactions/
│       └── year=2024/month=05/day=24/part-00000.parquet
│
├── features/                      # Engineered features for ML models
│   ├── user_features/
│   │   ├── year=2024/month=05/day=24/
│   │   │   └── part-00000.parquet
│   │   └── _metadata
│   ├── item_features/
│   │   ├── year=2024/month=05/day=24/
│   │   │   └── part-00000.parquet
│   │   └── _metadata
│   └── interaction_matrix/        # Sparse user-item interactions
│       ├── year=2024/month=05/day=24/
│       │   └── part-00000.parquet
│       └── _metadata
│
├── embeddings/                    # Pre-computed ML embeddings
│   ├── user_embeddings/
│   │   ├── version_001/
│   │   │   ├── year=2024/month=05/day=24/
│   │   │   │   └── part-00000.parquet
│   │   │   └── metadata.json
│   │   └── current/ → symlink to latest version
│   └── item_embeddings/
│       ├── version_001/
│       │   ├── year=2024/month=05/day=24/
│       │   │   └── part-00000.parquet
│       │   └── metadata.json
│       └── current/ → symlink to latest version
│
└── recommendations/               # Final recommendation outputs
    ├── batch_recommendations/
    │   └── year=2024/month=05/day=24/
    │       └── part-00000.parquet
    └── metadata/
        └── generation_logs.json
```

### S3 Configuration Best Practices

**Bucket Settings:**
```yaml
Versioning: Enabled
  - Enables rollback for corrupted data
  - Required for production compliance
  - Learning: Keep only last 3 versions to control costs

Lifecycle Policies:
  - Keep raw/ data indefinitely (cost-sensitive: tier to IA after 90 days)
  - Move processed/ to S3 IA after 30 days (historical analysis only)
  - Archive embeddings/version_* to Glacier after 6 months
  - Delete temporary/ data after 7 days automatically

Server-Side Encryption:
  - Learning: Default S3-managed encryption (SSE-S3)
  - Production: Customer-managed keys (SSE-KMS) for regulatory compliance
  - Enable encryption in transit (HTTPS only)

Block Public Access:
  - Enable all 4 options to prevent accidental exposure
  - No public read/write permissions under any circumstances
```

**Access Control:**
```yaml
Bucket Policy:
  - Allow only specific IAM roles (Glue, Lambda, SageMaker)
  - Deny unencrypted uploads (PutObject only with aws:x-amz-server-side-encryption)
  - Restrict access by source IP for sensitive environments

CORS Configuration:
  - Not needed for backend processing
  - Add only if frontend browser access required for dashboards
```

### Partitioning Strategy

**Design Rationale:** Time-based partitioning enables efficient queries on date ranges while improving query performance and reducing costs.

**Partitioning Scheme:**
```
year=YYYY/month=MM/day=DD/hour=HH/
  - Supports daily batch jobs with daily partitions
  - Enables monthly aggregations for reporting
  - Allows efficient date-range filtering
  - Reduces query scans by 90% when filtering to specific dates
```

**Partition Key Selection:**
- **Primary:** `year`, `month`, `day` (daily granularity for batch jobs)
- **Secondary:** `hour` (optional, for sub-daily processing)
- **Avoid:** User IDs or item IDs as partition keys (creates too many partitions)

**Query Performance Impact:**
```
Without Partitioning:
- Query: SELECT * FROM transactions WHERE t_dat = '2024-05-15'
- Scans: 31.8M rows entire dataset
- Cost: ~$0.15-0.25 per query

With Date Partitioning:
- Query: SELECT * FROM transactions WHERE year=2024 AND month=05 AND day=15
- Scans: ~100K rows only that partition
- Cost: ~$0.001-0.01 per query (95% reduction)
```

**Learning Project:** Use daily partitions only. Skip hourly for simplification.

**Production Scaling:** Add hourly partitions if processing frequency exceeds once daily.

## Data Processing Pipeline

### Glue ETL Jobs

**AWS Glue Overview:**
- Managed Apache Spark service (serverless ETL)
- Python/PySpark support for data transformations
- Native integration with S3, Redshift, DynamoDB
- Automatic Catalog schema discovery and metadata management

#### Job 1: Data Validation and Ingestion

**Purpose:** Validate source data, handle missing values, convert formats

**Configuration:**
```yaml
Job Name: hm-data-validation-and-ingestion
Trigger: Manual (learning) or EventBridge (production)
Worker Type: G.1X (4 vCPU, 16GB RAM)
Number of Workers: 2 (for learning), scale to 10 for production
Max Capacity: 20 DPU (learning), 100 DPU (production)
Timeout: 1800 seconds (30 minutes)
IAM Role: GlueServiceRole with S3 and Catalog permissions
```

**Processing Steps:**

1. **Read CSV from S3 raw/**
   ```python
   import awswrangler as wr
   from pyspark.sql import SparkSession
   
   spark = SparkSession.builder.appName("DataValidation").getOrCreate()
   
   # Read with schema inference
   articles = wr.s3.read_csv(
       "s3://fashion-recommender-data-dev/raw/articles/articles.csv",
       dtype_backend="pyarrow",
       on_bad_lines="skip"  # Handle malformed rows
   )
   ```

2. **Schema Validation**
   ```python
   # Expected schema
   expected_schema = {
       'article_id': 'string',
       'product_code': 'string', 
       'prod_name': 'string',
       'product_type_no': 'int',
       # ... complete schema
   }
   
   # Validate
   for col, dtype in expected_schema.items():
       if col not in articles.columns:
           raise ValueError(f"Missing column: {col}")
   ```

3. **Data Quality Checks**
   ```python
   # Completeness
   assert articles['article_id'].isna().sum() == 0, "Null article_ids found"
   assert customers['customer_id'].isna().sum() == 0, "Null customer_ids found"
   
   # Uniqueness
   assert articles['article_id'].duplicated().sum() == 0, "Duplicate article_ids"
   
   # Value ranges
   assert transactions['price'].between(0, 1000).all(), "Price out of range"
   assert transactions['sales_channel_id'].isin([1, 2]).all(), "Invalid channel"
   ```

4. **Convert to Parquet with Partitioning**
   ```python
   from datetime import datetime
   
   # Add processing metadata
   articles['processing_date'] = datetime.now().date()
   articles['processing_version'] = '1.0'
   
   # Write as Parquet with date partitions
   wr.s3.to_parquet(
       df=articles,
       path="s3://fashion-recommender-data-dev/processed/articles/",
       dataset=True,
       partition_cols=['processing_date'],
       compression='snappy'
   )
   ```

5. **Register in Glue Catalog**
   ```python
   wr.catalog.create_parquet_table(
       database='fashion_recommendation',
       table='articles_processed',
       path='s3://fashion-recommender-data-dev/processed/articles/',
       columns_types={
           'article_id': 'string',
           'prod_name': 'string',
           # ... all columns
       }
   )
   ```

#### Job 2: Feature Engineering

**Purpose:** Transform raw data into ML-ready features

**Configuration:**
```yaml
Job Name: hm-feature-engineering
Trigger: EventBridge after validation job completes
Worker Type: G.2X (8 vCPU, 32GB RAM)  
Number of Workers: 5
Max Capacity: 50 DPU
Timeout: 3600 seconds (1 hour)
```

**Feature Groups:**

1. **User Features** (from customers + transaction history)
   ```python
   # Aggregate transaction history per user
   user_stats = transactions.groupby('customer_id').agg({
       'article_id': ['count', 'nunique'],  # total purchases, unique items
       'price': ['sum', 'mean', 'std'],     # spending patterns
       't_dat': ['min', 'max']              # first/last purchase dates
   }).reset_index()
   
   # Merge with customer demographics
   user_features = customers.merge(user_stats, on='customer_id', how='left')
   user_features['purchase_frequency'] = (
       user_stats['article_id_count'] / 
       ((user_stats['t_dat_max'] - user_stats['t_dat_min']).dt.days + 1)
   )
   
   # Bucket age for categorical feature
   user_features['age_bucket'] = pd.cut(
       user_features['age'],
       bins=[0, 20, 30, 40, 50, 100],
       labels=['teen', '20s', '30s', '40s', '50+']
   )
   ```

2. **Item Features** (from articles + transaction history)
   ```python
   # Popularity metrics
   item_stats = transactions.groupby('article_id').agg({
       'customer_id': ['nunique'],          # unique buyers
       'price': ['mean'],                   # avg price point
       't_dat': ['count', 'min', 'max']     # purchase frequency
   }).reset_index()
   
   # Merge with article attributes
   item_features = articles.merge(item_stats, on='article_id', how='left')
   item_features['popularity_score'] = (
       item_stats['customer_id_nunique'] / len(customers.unique())
   )
   item_features['recent_activity'] = (
       datetime.now().date() - item_stats['t_dat_max']
   ).dt.days
   ```

3. **Interaction Matrix** (sparse user-item pairs)
   ```python
   # Create explicit feedback signal
   interaction_matrix = transactions[[
       'customer_id', 'article_id', 't_dat'
   ]].copy()
   
   # Count purchases per user-item pair
   interaction_matrix = interaction_matrix.groupby(
       ['customer_id', 'article_id']
   ).agg({
       't_dat': 'count'  # number of times user bought item
   }).reset_index()
   interaction_matrix.rename(
       columns={'t_dat': 'interaction_strength'},
       inplace=True
   )
   
   # Normalize to 0-1 scale
   max_interactions = interaction_matrix['interaction_strength'].max()
   interaction_matrix['normalized_strength'] = (
       interaction_matrix['interaction_strength'] / max_interactions
   )
   ```

4. **Temporal Features** (for seasonal modeling)
   ```python
   from datetime import datetime
   import numpy as np
   
   # Day of week, month, quarter
   transactions['dow'] = transactions['t_dat'].dt.dayofweek
   transactions['month'] = transactions['t_dat'].dt.month
   transactions['quarter'] = transactions['t_dat'].dt.quarter
   transactions['days_since_epoch'] = (
       (transactions['t_dat'] - datetime(2000, 1, 1)).dt.days
   )
   
   # Seasonal indicators (for fashion seasonality)
   transactions['is_holiday_season'] = transactions['month'].isin([11, 12])
   transactions['is_summer'] = transactions['month'].isin([6, 7, 8])
   ```

**Output:**
```
s3://fashion-recommender-data-dev/features/
├── user_features/year=2024/month=05/day=24/part-*.parquet
├── item_features/year=2024/month=05/day=24/part-*.parquet
└── interaction_matrix/year=2024/month=05/day=24/part-*.parquet
```

#### Job 3: Data Quality Monitoring

**Purpose:** Continuous validation and anomaly detection

**Configuration:**
```yaml
Job Name: hm-data-quality-checks
Trigger: EventBridge after feature engineering completes
Worker Type: G.1X
Number of Workers: 1
Timeout: 600 seconds (10 minutes)
```

**Quality Rules:**
```python
from awsglue.dataquality import DataQualityBuilder

quality_checks = DataQualityBuilder(spark) \
    .add_check("Row count increases or stays same", 
               "row_count > 1000") \
    .add_check("No null customer_ids",
               "NULL_COUNT(customer_id) == 0") \
    .add_check("No null article_ids",
               "NULL_COUNT(article_id) == 0") \
    .add_check("Price values in valid range",
               "AVG(price) BETWEEN 0 AND 1000") \
    .add_check("Transaction dates recent",
               "MAX(t_dat) >= CURRENT_DATE - INTERVAL 7 DAYS") \
    .add_check("No duplicate customer_id-article_id pairs per day",
               "DUPLICATE_COUNT(customer_id, article_id) == 0") \
    .build()

quality_checks.run()
```

**Actions on Failure:**
- Log to CloudWatch
- Publish to SNS for alerting
- Write to S3 quality_reports/ bucket
- Stop downstream jobs if critical checks fail

### Athena Query Patterns

**AWS Athena:** Interactive SQL query engine on S3 data (pay-per-byte-scanned model)

**Setup:**
```sql
-- Create database
CREATE DATABASE fashion_recommendation
COMMENT 'H&M recommendation system data lake'
LOCATION 's3://fashion-recommender-data-dev/processed/';

-- Register processed tables
CREATE EXTERNAL TABLE IF NOT EXISTS articles_processed (
    article_id STRING,
    product_code STRING,
    prod_name STRING,
    product_type_no INT,
    product_type_name STRING,
    product_group_name STRING,
    graphical_appearance_no INT,
    graphical_appearance_name STRING,
    colour_group_code STRING,
    colour_group_name STRING,
    perceived_colour_value_id INT,
    perceived_colour_value_name STRING,
    perceived_colour_master_id INT,
    perceived_colour_master_name STRING,
    department_no INT,
    department_name STRING,
    index_code STRING,
    index_name STRING,
    index_group_no INT,
    index_group_name STRING,
    section_no INT,
    section_name STRING,
    garment_group_no INT,
    garment_group_name STRING,
    detail_desc STRING
)
PARTITIONED BY (
    year INT,
    month INT,
    day INT
)
STORED AS PARQUET
LOCATION 's3://fashion-recommender-data-dev/processed/articles/';
```

**Common Query Patterns:**

1. **Customer Purchase Patterns**
   ```sql
   -- Find customers with highest spending
   SELECT 
       customer_id,
       COUNT(DISTINCT article_id) as unique_items_purchased,
       SUM(price) as total_spent,
       AVG(price) as avg_price_per_item,
       DATE_DIFF('day', MIN(t_dat), MAX(t_dat)) as days_active
   FROM transactions_processed
   WHERE year = 2024 AND month = 5  -- Partition pruning
   GROUP BY customer_id
   ORDER BY total_spent DESC
   LIMIT 100;
   ```

2. **Item Popularity Analysis**
   ```sql
   -- Top-selling items by category
   SELECT 
       a.product_group_name,
       a.article_id,
       a.prod_name,
       COUNT(DISTINCT t.customer_id) as unique_buyers,
       SUM(t.price) as total_revenue
   FROM transactions_processed t
   JOIN articles_processed a ON t.article_id = a.article_id
   WHERE t.year = 2024 AND t.month = 5
   GROUP BY a.product_group_name, a.article_id, a.prod_name
   ORDER BY total_revenue DESC;
   ```

3. **Temporal Trends**
   ```sql
   -- Daily transaction volume
   SELECT 
       year,
       month, 
       day,
       COUNT(*) as transaction_count,
       SUM(price) as daily_revenue,
       COUNT(DISTINCT customer_id) as active_customers
   FROM transactions_processed
   WHERE year = 2024
   GROUP BY year, month, day
   ORDER BY year, month, day;
   ```

4. **User Segmentation**
   ```sql
   -- Segment users by purchase frequency
   SELECT 
       customer_id,
       COUNT(*) as purchase_count,
       CASE 
           WHEN COUNT(*) >= 10 THEN 'High Frequency'
           WHEN COUNT(*) >= 5 THEN 'Medium Frequency'
           ELSE 'Low Frequency'
       END as segment
   FROM transactions_processed
   WHERE year IN (2023, 2024)
   GROUP BY customer_id;
   ```

**Cost Optimization:**
```
Query Plan Analysis:
- Always filter by date partitions (year/month/day)
- Use column pruning (SELECT specific columns, not *)
- Aggregate before joining large tables
- Use LIMIT to cap expensive operations

Typical Costs:
- Simple partition-filtered query: $0.001-0.01
- Full table scan: $0.10-0.25
- Complex join with aggregation: $0.05-0.15
```

**Learning vs Production Considerations:**
- Learning: Use Athena for ad-hoc analysis only
- Production: Set up Athena Workgroups with cost controls
  - Per-query cost limits
  - Result caching
  - Reserved capacity pricing

## Data Catalog Integration

### Glue Data Catalog

**Purpose:** Centralized metadata repository for all data assets

**Table Registration:**
```python
# Auto-register Parquet tables from S3
import boto3
from awsglue.catalog.dynamic_frame import DynamicFrame

glue_client = boto3.client('glue')

# Create table in catalog
glue_client.create_table(
    CatalogId='123456789012',
    DatabaseName='fashion_recommendation',
    TableInput={
        'Name': 'transactions_processed',
        'StorageDescriptor': {
            'Columns': [
                {'Name': 'customer_id', 'Type': 'string'},
                {'Name': 'article_id', 'Type': 'string'},
                {'Name': 't_dat', 'Type': 'date'},
                {'Name': 'price', 'Type': 'double'},
                {'Name': 'sales_channel_id', 'Type': 'int'}
            ],
            'Location': 's3://fashion-recommender-data-dev/processed/transactions/',
            'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
            'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
            'SerdeInfo': {
                'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
            }
        },
        'PartitionKeys': [
            {'Name': 'year', 'Type': 'int'},
            {'Name': 'month', 'Type': 'int'},
            {'Name': 'day', 'Type': 'int'}
        ]
    }
)

# Add partition projection for automatic partition discovery
glue_client.update_table(
    DatabaseName='fashion_recommendation',
    TableInput={
        'Name': 'transactions_processed',
        'TableType': 'EXTERNAL_TABLE',
        'Parameters': {
            'projection.enabled': 'true',
            'projection.year.type': 'integer',
            'projection.year.range': '2020,2026',
            'projection.month.type': 'integer',
            'projection.month.range': '1,12',
            'projection.month.digits': '2',
            'projection.day.type': 'integer',
            'projection.day.range': '1,31',
            'projection.day.digits': '2',
            'storage.location.template': (
                's3://fashion-recommender-data-dev/processed/transactions/'
                'year=${year}/month=${month}/day=${day}'
            )
        }
    }
)
```

**Lineage Tracking:**
- Glue automatically tracks job dependencies
- Catalog shows which data was created by which ETL job
- Enables impact analysis for data changes

### Schema Versioning

**Strategy:**
```
Version 1.0: Initial schema from H&M dataset
├── Breaking changes: Create new table version (v1_1)
└── Non-breaking: Add new columns to existing table

S3 Structure:
features/user_features/
├── schema_v1/        # Deprecated
├── schema_v2/        # Active
└── schema_v2_staging/  # For testing
```

## Data Flow: From Raw to Features

```
Step 1: Data Ingestion
├─ Manual upload: CSV files → s3://raw/
└─ Automated: S3 event notification → Lambda → Glue trigger

Step 2: Data Validation & Conversion
├─ Trigger: EventBridge rule on S3 PutObject event
├─ Job: hm-data-validation-and-ingestion
├─ Tasks:
│  ├─ Read CSV with schema inference
│  ├─ Validate data quality
│  ├─ Handle missing values
│  └─ Convert to Parquet format
└─ Output: s3://processed/ (partitioned by date)

Step 3: Feature Engineering
├─ Trigger: EventBridge after validation completes
├─ Job: hm-feature-engineering
├─ Tasks:
│  ├─ Aggregate user statistics
│  ├─ Calculate item popularity
│  ├─ Build interaction matrix
│  └─ Engineer temporal features
└─ Output: s3://features/user_features/, item_features/, etc.

Step 4: Quality Monitoring
├─ Trigger: EventBridge after feature engineering
├─ Job: hm-data-quality-checks
├─ Tasks:
│  ├─ Row count validation
│  ├─ Column completeness checks
│  ├─ Value range validation
│  └─ Anomaly detection
└─ Output: CloudWatch metrics + SNS alerts

Step 5: Ready for ML
├─ Features available in S3 for SageMaker training
├─ Metadata in Glue Catalog for easy discovery
└─ Athena available for exploratory analysis
```

## Performance Optimization

### Parquet Format Advantages

**Why Parquet Over CSV:**

| Aspect | CSV | Parquet |
|--------|-----|---------|
| **Compression** | Minimal (~10-20%) | Excellent (~70-80%) |
| **Query Performance** | Scans entire file | Reads only needed columns |
| **Storage Cost** | High per GB | 5-10x lower |
| **Schema Inference** | Required at read time | Embedded in file |
| **Nested Structures** | Not supported | Full support |
| **Type Safety** | String only | Strong typing |

**Size Comparison (H&M Dataset):**
```
CSV Format:
- articles.csv: ~50 MB
- customers.csv: ~150 MB
- transactions.csv: ~2.8 GB
- Total: ~3.0 GB

Parquet Format (with snappy compression):
- articles/: ~5 MB (90% reduction)
- customers/: ~15 MB (90% reduction)
- transactions/: ~400 MB (85% reduction)
- Total: ~420 MB (86% reduction)

Storage Savings: $150-200/month for large-scale deployments
Query Performance: 10-20x faster for columnar operations
```

### Parquet Optimization Configuration

**Compression Settings:**
```python
# Snappy: Fast compression, good ratio (recommended for Athena)
wr.s3.to_parquet(
    df=data,
    path="s3://bucket/path/",
    compression='snappy',  # Default, good balance
)

# Gzip: Higher compression ratio, slower (for storage optimization)
wr.s3.to_parquet(
    df=data,
    path="s3://bucket/path/",
    compression='gzip',    # 20-30% more compression, 2-3x slower
)

# Uncompressed: Fastest, only for high-throughput scenarios
wr.s3.to_parquet(
    df=data,
    path="s3://bucket/path/",
    compression=None,
)
```

**Row Group Configuration:**
```python
# Optimize for Athena (128MB row groups)
wr.s3.to_parquet(
    df=data,
    path="s3://bucket/path/",
    compression='snappy',
    dataset=True,
    mode='overwrite_partitions',
    glue_partitions=True  # Auto-register in Glue Catalog
)

# Parquet File Layout:
# Row Group 1: Rows 0-100K (optimized for Athena scans)
# Row Group 2: Rows 100K-200K
# ...
# Footer: Schema, compression codec, statistics
```

### Partitioning Strategy Performance

**Partition Pruning Impact:**
```sql
-- Without partitioning
Query: SELECT COUNT(*) FROM transactions WHERE DATE(t_dat) = '2024-05-15'
- Scans: 2.8 GB entire file
- Query Time: ~5 seconds
- Cost: $0.20

-- With year/month/day partitioning
Query: SELECT COUNT(*) FROM transactions 
       WHERE year=2024 AND month=5 AND day=15
- Scans: ~10 MB specific partition only
- Query Time: ~0.1 seconds
- Cost: $0.001

Improvement: 200x query speedup, 99.5% cost reduction
```

### Columnar Query Optimization

**Column Projection:**
```sql
-- Bad: Unnecessary columns slow query
SELECT * FROM transactions
WHERE year = 2024 AND month = 5;
-- Scans ALL columns: ~2.8 GB

-- Good: Select only needed columns
SELECT customer_id, article_id, price, t_dat
FROM transactions
WHERE year = 2024 AND month = 5;
-- Scans 4 columns: ~500 MB (80% reduction)
```

## Data Governance and Monitoring

### Quality Monitoring Workflow

**Automated Quality Checks:**

1. **Completeness Check**
   ```python
   # Verify required fields are populated
   null_counts = {
       'customer_id': df['customer_id'].isna().sum(),
       'article_id': df['article_id'].isna().sum(),
       'price': df['price'].isna().sum()
   }
   
   for field, count in null_counts.items():
       if count > 0:
           raise DataQualityException(f"Nulls in {field}: {count}")
   ```

2. **Consistency Check**
   ```python
   # Verify referential integrity
   valid_articles = set(articles_df['article_id'])
   invalid_transactions = (
       ~transactions_df['article_id'].isin(valid_articles)
   ).sum()
   
   if invalid_transactions > 0:
       raise DataQualityException(
           f"Invalid article_ids in {invalid_transactions} rows"
       )
   ```

3. **Freshness Check**
   ```python
   from datetime import datetime, timedelta
   
   max_date = transactions_df['t_dat'].max()
   days_old = (datetime.now().date() - max_date.date()).days
   
   if days_old > 7:
       logger.warning(f"Data is {days_old} days old")
   ```

4. **Distribution Check**
   ```python
   # Detect anomalies in data distribution
   price_mean = df['price'].mean()
   price_std = df['price'].std()
   
   outliers = (
       (df['price'] > price_mean + 3 * price_std) |
       (df['price'] < price_mean - 3 * price_std)
   ).sum()
   
   if outliers > len(df) * 0.05:  # > 5% outliers
       logger.warning(f"Unusual price distribution: {outliers} outliers")
   ```

### CloudWatch Monitoring

**Custom Metrics:**
```python
import boto3
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

# Log data quality metrics
cloudwatch.put_metric_data(
    Namespace='FashionRecommender/DataQuality',
    MetricData=[
        {
            'MetricName': 'ProcessedRecordCount',
            'Value': len(processed_df),
            'Unit': 'Count',
            'Timestamp': datetime.utcnow()
        },
        {
            'MetricName': 'DataFreshnessDays',
            'Value': days_old,
            'Unit': 'None',
            'Timestamp': datetime.utcnow()
        },
        {
            'MetricName': 'FeatureEngineeringDurationSeconds',
            'Value': elapsed_time,
            'Unit': 'Seconds',
            'Timestamp': datetime.utcnow()
        }
    ]
)
```

**Dashboard Setup:**
```yaml
Dashboard: FashionRecommender-DataQuality
Widgets:
  - ProcessedRecordCount: Line chart (trend over time)
  - FeatureEngineeringDuration: Bar chart (per-job timing)
  - DataFreshness: Gauge (max age of data)
  - QualityChecksPassed: Stacked area (pass/fail rate)
```

## Learning vs Production Considerations

### Learning Project Simplifications

**Scope Reduction:**
```
Production Scale:
- Continuous data ingestion (hourly)
- 1.3M users, 105K items, 31M transactions
- Multiple data sources (API feeds, catalogs)
- Complex transformation workflows

Learning Project:
- Batch ingestion (daily)
- 10K users, 5K items, 100K transactions (sample)
- Single H&M CSV source
- Simplified feature engineering
```

**Infrastructure:**
```yaml
Learning Glue Configuration:
  Worker Type: G.1X (4 vCPU)
  Number of Workers: 2
  Max Capacity: 20 DPU
  Job Timeout: 30-60 minutes

Production Glue Configuration:
  Worker Type: G.2X (8 vCPU)
  Number of Workers: 10-50
  Max Capacity: 100-500 DPU
  Job Timeout: 2-4 hours
  Auto-scaling enabled
```

**Feature Engineering:**
```
Learning: Calculate basic features
- User: purchase count, avg price, unique items
- Item: popularity, avg rating
- Interaction: simple user-item pairs

Production: Advanced feature engineering
- User: behavioral segmentation, seasonal preferences, RFM
- Item: category affinities, visual embeddings, inventory velocity
- Interaction: time decay, contextual weighting, cross-category patterns
```

### Scaling to Production

**Step 1: Data Volume Increase**
- Increase Glue workers from 2 to 10+
- Enable partitioned data updates (incremental processing)
- Add data validation automation
- Set up Glue Data Quality jobs

**Step 2: Real-time Ingestion**
- Replace batch CSV uploads with Kinesis Data Firehose
- Implement micro-batch processing (5-minute windows)
- Add data enrichment streams
- Set up Kinesis monitoring

**Step 3: Advanced Governance**
- Implement Glue data catalog governance
- Add column-level encryption for sensitive data
- Set up automatic data lineage tracking
- Create data governance dashboards

**Step 4: Cost Optimization**
- Reserved capacity for Glue jobs
- S3 Intelligent-Tiering for automatic cost optimization
- Athena reserved capacity (500 DPU)
- Spot instances for non-critical batch jobs

## Integration with Other Layers

### Data Layer → ML Layer

**Feature Consumption:**
```python
# ML training pipeline reads features from data lake
import pandas as pd
import sagemaker

# Read features from S3
user_features = pd.read_parquet(
    's3://fashion-recommender-data-dev/features/user_features/year=2024/month=05/day=24/'
)
item_features = pd.read_parquet(
    's3://fashion-recommender-data-dev/features/item_features/year=2024/month=05/day=24/'
)

# Train embedding model
training_job = sagemaker.estimator.Estimator(
    image_uri='training-image:latest',
    role='SageMaker-Role-ARN',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    output_path='s3://fashion-recommender-data-dev/models/'
)

training_job.fit({
    'training': 's3://fashion-recommender-data-dev/features/'
})
```

### Data Lake → Application Layer

**Feature Serving:**
```python
# Lambda function reads features for real-time recommendations
import boto3
import pandas as pd

s3 = boto3.client('s3')

def lambda_handler(event, context):
    user_id = event['user_id']
    
    # Load user features from S3
    obj = s3.get_object(
        Bucket='fashion-recommender-data-dev',
        Key=f'features/user_features/current/user_{user_id}.parquet'
    )
    user_features = pd.read_parquet(obj['Body'])
    
    # Use in recommendation logic
    return generate_recommendations(user_features)
```

## Summary

The data layer provides:
- **Scalable Storage:** S3 data lake with clear organization
- **Efficient Processing:** Glue ETL jobs with Parquet optimization
- **Query Performance:** 95% cost reduction through partitioning
- **Data Governance:** Glue Catalog with lineage tracking
- **Production Ready:** Monitoring, quality checks, and operational patterns

This architecture supports both learning project simplifications and production-scale requirements, enabling seamless progression from POC to production deployment.
