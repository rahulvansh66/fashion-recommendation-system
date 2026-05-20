# H&M Dataset Schema Documentation

## Database Overview

The H&M fashion recommendation dataset consists of three core tables forming a complete e-commerce transaction system. This schema supports personalized fashion recommendation systems with comprehensive product, customer, and behavioral data.

### Technical Summary
- **Storage Format**: CSV files (production: PostgreSQL recommended)
- **Total Records**: 
  - Articles: 105,542 unique products
  - Customers: 1,371,980 unique customers
  - Transactions: 31,788,324 transaction records
- **Relationships**: Star schema with transactions as fact table
- **Recommended Engine**: PostgreSQL with pgvector extension for embedding storage
- **Total Data Volume**: ~33.3M lines across all files

### Scale and Performance Considerations
- **Transaction Table**: 31.8M rows - requires indexing on customer_id, article_id, and t_dat for efficient querying
- **Customer Base**: 1.37M unique customers - manageable in-memory for user embeddings
- **Product Catalog**: 105K articles - suitable for collaborative filtering and content-based recommendations
- **Temporal Data**: Transactions span multiple years enabling time-series analysis and seasonal pattern detection
- **Indexing Strategy**: Multi-column indexes essential for transaction filtering and customer-article joins

## Table Schemas

### 1. Articles Table
**File**: `data/full/articles.csv`
**Records**: 105,542
**Purpose**: Master product catalog with hierarchical classification and descriptive attributes

| Column | Type | Description |
|--------|------|-------------|
| article_id | STRING | Unique product identifier (12 digits) |
| product_code | STRING | Product code for grouping variants |
| prod_name | STRING | Product name/title |
| product_type_no | INTEGER | Product category code |
| product_type_name | STRING | Product category name (e.g., "Vest top") |
| product_group_name | STRING | High-level product group (e.g., "Garment Upper body") |
| graphical_appearance_no | INTEGER | Visual appearance code |
| graphical_appearance_name | STRING | Visual appearance description (e.g., "Solid") |
| colour_group_code | STRING | Color identifier |
| colour_group_name | STRING | Color name (e.g., "Black") |
| perceived_colour_value_id | INTEGER | Perceived color value code |
| perceived_colour_value_name | STRING | Perceived color value (e.g., "Dark") |
| perceived_colour_master_id | INTEGER | Master color category |
| perceived_colour_master_name | STRING | Master color name |
| department_no | INTEGER | Department code |
| department_name | STRING | Department name (e.g., "Jersey Basic") |
| index_code | STRING | Index code for classification |
| index_name | STRING | Index name (e.g., "Ladieswear") |
| index_group_no | INTEGER | Index group code |
| index_group_name | STRING | Index group name (e.g., "Ladieswear") |
| section_no | INTEGER | Section code |
| section_name | STRING | Section name (e.g., "Womens Everyday Basics") |
| garment_group_no | INTEGER | Garment group code |
| garment_group_name | STRING | Garment group name (e.g., "Jersey Basic") |
| detail_desc | STRING | Detailed product description |

**Key Indexes**:
- Primary: article_id
- Secondary: product_code, product_type_name, colour_group_name

### 2. Customers Table
**File**: `data/full/customers.csv`
**Records**: 1,371,980
**Purpose**: Customer demographics and engagement preferences

| Column | Type | Description |
|--------|------|-------------|
| customer_id | STRING | Unique customer identifier (hashed for privacy) |
| FN | STRING | Possibly "First Name" indicator (mostly empty) |
| Active | STRING | Customer activity status (mostly empty) |
| club_member_status | STRING | Club membership status (values: ACTIVE, INACTIVE, PRE-CREATE) |
| fashion_news_frequency | STRING | Frequency of fashion news subscription (NONE, Regularly, Monthly, Weekly) |
| age | INTEGER | Customer age in years |
| postal_code | STRING | Customer postal code (hashed for privacy) |

**Key Indexes**:
- Primary: customer_id
- Secondary: club_member_status, age

**Data Quality Notes**:
- FN and Active columns are largely empty
- Customer IDs and postal codes are hashed for privacy
- Age data ranges from teenagers to elderly customers
- Fashion news frequency indicates engagement level

### 3. Transactions Table
**File**: `data/full/transactions_train.csv`
**Records**: 31,788,324
**Purpose**: Fact table recording all customer purchase interactions (training set)

| Column | Type | Description |
|--------|------|-------------|
| t_dat | DATE | Transaction date (YYYY-MM-DD format) |
| customer_id | STRING | Foreign key to customers.customer_id (hashed) |
| article_id | STRING | Foreign key to articles.article_id |
| price | FLOAT | Transaction price (normalized/scaled value) |
| sales_channel_id | INTEGER | Sales channel identifier (1 or 2) |

**Key Indexes**:
- Primary: (t_dat, customer_id, article_id) composite
- Secondary: customer_id, article_id, t_dat
- Consider: Partitioning by t_dat for temporal queries

**Data Characteristics**:
- Prices are normalized (range: 0-1 approximate)
- Sales channels represent different distribution methods (online/offline)
- Temporal range spans multiple years
- High cardinality on customer_id enables personalization

## Relationship Model

```
CUSTOMERS (1) ─────< (M) TRANSACTIONS >(M)─── (1) ARTICLES
  - customer_id         - customer_id            - article_id
  - demographics        - article_id      
  - preferences         - price, date, channel
```

**Join Patterns**:
1. Customer Purchase History: `TRANSACTIONS JOIN CUSTOMERS ON transactions.customer_id = customers.customer_id`
2. Product Details: `TRANSACTIONS JOIN ARTICLES ON transactions.article_id = articles.article_id`
3. Customer-Product Matrix: `TRANSACTIONS GROUP BY customer_id, article_id`
4. Temporal Analysis: Filter `TRANSACTIONS WHERE t_dat BETWEEN date1 AND date2`

## Data Quality Observations

### Articles Table
- Complete article coverage with hierarchical classifications
- Rich descriptive data including visual attributes and product categorization
- No null values in key identifier fields

### Customers Table
- Privacy-protected identifiers (hashed customer_id and postal_code)
- Sparse demographic data (FN and Active columns largely empty)
- Meaningful engagement metric (fashion_news_frequency)
- Age data available for segmentation

### Transactions Table
- Normalized pricing structure
- Comprehensive temporal coverage
- Clean foreign key references to both customers and articles
- Sales channel differentiation supports multi-channel analysis

## Recommended Queries for Analysis

### 1. Customer Segmentation
```sql
SELECT 
  age,
  club_member_status,
  COUNT(DISTINCT customer_id) as customer_count,
  COUNT(*) as total_transactions,
  AVG(price) as avg_price
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
GROUP BY age, club_member_status
ORDER BY customer_count DESC;
```

### 2. Product Performance
```sql
SELECT 
  pa.product_group_name,
  pa.colour_group_name,
  COUNT(*) as purchase_count,
  AVG(t.price) as avg_price,
  COUNT(DISTINCT t.customer_id) as unique_customers
FROM transactions t
JOIN articles pa ON t.article_id = pa.article_id
GROUP BY pa.product_group_name, pa.colour_group_name
ORDER BY purchase_count DESC;
```

### 3. Temporal Trends
```sql
SELECT 
  DATE_TRUNC('month', t.t_dat) as month,
  COUNT(*) as transaction_count,
  COUNT(DISTINCT t.customer_id) as active_customers,
  SUM(t.price) as total_revenue
FROM transactions t
GROUP BY DATE_TRUNC('month', t.t_dat)
ORDER BY month;
```

## Next Steps

- [ ] Create table indexing strategy for PostgreSQL
- [ ] Define data validation rules and constraints
- [ ] Document embedding generation for recommendation models
- [ ] Define ETL pipeline for production deployment
- [ ] Create data lineage documentation
