# H&M Schema Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create comprehensive schema documentation for H&M dataset with technical-first structure and embedded business context.

**Architecture:** Single markdown document analyzing CSV file structures to produce database schema documentation with dual data typing, relationship mapping, and performance recommendations.

**Tech Stack:** Data analysis of CSV files, markdown documentation, SQL schema specifications

---

## File Structure

**Files to create:**
- `docs/project-info/schema-info.md` - Main schema documentation

**Files to analyze:**
- `data/full/articles.csv` - Product catalog data
- `data/full/customers.csv` - Customer profile data  
- `data/full/transactions_train.csv` - Transaction history data

---

### Task 1: Analyze Data Structure and Create Document Foundation

**Files:**
- Create: `docs/project-info/schema-info.md`
- Analyze: `data/full/articles.csv`, `data/full/customers.csv`, `data/full/transactions_train.csv`

- [ ] **Step 1: Examine CSV file structures**

```bash
# Check file sizes and basic structure
wc -l data/full/*.csv
head -2 data/full/articles.csv
head -2 data/full/customers.csv  
head -2 data/full/transactions_train.csv
```

Expected: View column headers and sample data for all three files

- [ ] **Step 2: Create document foundation**

```markdown
# H&M Dataset Schema Documentation

## Database Overview

The H&M fashion recommendation dataset consists of three core tables forming a complete e-commerce transaction system. This schema supports personalized fashion recommendation systems with comprehensive product, customer, and behavioral data.

### Technical Summary
- **Storage Format**: CSV files (production: PostgreSQL recommended)
- **Total Records**: ~31M transactions, ~105K articles, ~137K customers  
- **Relationships**: Star schema with transactions as fact table
- **Recommended Engine**: PostgreSQL with pgvector extension for embedding storage

### Scale and Performance Considerations
```

- [ ] **Step 3: Add database overview metrics**

```bash
# Get exact record counts
wc -l data/full/articles.csv data/full/customers.csv data/full/transactions_train.csv
```

Complete the Technical Summary section with actual counts and file sizes.

- [ ] **Step 4: Commit foundation**

```bash
git add docs/project-info/schema-info.md
git commit -m "docs: add H&M schema documentation foundation

- Database overview with technical summary
- Scale and performance considerations
- CSV file analysis baseline"
```

---

### Task 2: Document Articles Table Schema

**Files:**
- Modify: `docs/project-info/schema-info.md`
- Analyze: `data/full/articles.csv`

- [ ] **Step 1: Analyze articles table structure**

```bash
# Examine articles structure in detail
head -1 data/full/articles.csv | tr ',' '\n' | nl
head -5 data/full/articles.csv
```

Expected: Complete column listing and sample data

- [ ] **Step 2: Add articles table documentation**

```markdown
## Table Schemas

### articles - Product Catalog

Core product information table containing fashion article details, categorization, and metadata for recommendation algorithms.

**Primary Key**: `article_id`
**Indexes Recommended**: 
- `idx_articles_product_type` on `product_type_no`
- `idx_articles_department` on `department_no` 
- `idx_articles_color` on `colour_group_code`

#### Column Specifications

| Column | Basic Type | Detailed Type | Description | Relationships |
|--------|------------|---------------|-------------|---------------|
| `article_id` | string | varchar(20) | Unique product identifier, primary key for catalog lookups | → Referenced by transactions_train.article_id |
| `product_code` | string | varchar(15) | Product family identifier, groups article variants | |
| `prod_name` | string | varchar(100) | Product display name for customer interface | |
| `product_type_no` | integer | smallint | Numeric category code for product type classification | |
| `product_type_name` | string | varchar(50) | Human-readable product type (e.g., "Vest top", "Bra") | |
| `product_group_name` | string | varchar(30) | High-level category (e.g., "Garment Upper body", "Underwear") | |
| `graphical_appearance_no` | integer | smallint | Visual pattern classification code | |
| `graphical_appearance_name` | string | varchar(20) | Pattern description (e.g., "Solid", "Stripe") | |
| `colour_group_code` | integer | tinyint | Numeric color classification | |
| `colour_group_name` | string | varchar(20) | Color name (e.g., "Black", "White", "Off White") | |
| `perceived_colour_value_id` | integer | tinyint | Color brightness/saturation level | |
| `perceived_colour_value_name` | string | varchar(15) | Brightness description (e.g., "Dark", "Light", "Dusty Light") | |
| `perceived_colour_master_id` | integer | tinyint | Master color grouping identifier | |
| `perceived_colour_master_name` | string | varchar(15) | Master color name for broad categorization | |
| `department_no` | integer | smallint | Department classification number | |
| `department_name` | string | varchar(30) | Department name (e.g., "Jersey Basic", "Clean Lingerie") | |
| `index_code` | string | char(1) | Index classification code | |
| `index_name` | string | varchar(15) | Index category (e.g., "Ladieswear", "Lingeries/Tights") | |
| `index_group_no` | integer | tinyint | Index group classification | |
| `index_group_name` | string | varchar(15) | Index group name | |
| `section_no` | integer | tinyint | Section classification number | |
| `section_name` | string | varchar(30) | Section description | |
| `garment_group_no` | integer | smallint | Garment category grouping | |
| `garment_group_name` | string | varchar(20) | Garment category name | |
| `detail_desc` | string | text | Detailed product description for recommendations and search | |
```

- [ ] **Step 3: Commit articles table documentation**

```bash
git add docs/project-info/schema-info.md
git commit -m "docs: add articles table schema documentation

- Complete column specifications with dual typing
- Primary key and index recommendations
- Relationship indicators to transactions table"
```

---

### Task 3: Document Customers Table Schema

**Files:**
- Modify: `docs/project-info/schema-info.md`
- Analyze: `data/full/customers.csv`

- [ ] **Step 1: Analyze customers table structure**

```bash
# Examine customers structure
head -1 data/full/customers.csv | tr ',' '\n' | nl
head -5 data/full/customers.csv
```

- [ ] **Step 2: Add customers table documentation**

```markdown
### customers - Customer Profiles

Customer demographic and preference data enabling personalized recommendations and segmentation analysis.

**Primary Key**: `customer_id`
**Indexes Recommended**:
- `idx_customers_age` on `age`
- `idx_customers_club_status` on `club_member_status`
- `idx_customers_postal` on `postal_code`

#### Column Specifications

| Column | Basic Type | Detailed Type | Description | Relationships |
|--------|------------|---------------|-------------|---------------|
| `customer_id` | string | varchar(64) | Unique customer identifier, hashed for privacy | → Referenced by transactions_train.customer_id |
| `FN` | string | varchar(10) | Customer first name indicator (often null for privacy) | |
| `Active` | string | varchar(10) | Customer account status flag | |
| `club_member_status` | string | varchar(10) | Loyalty program membership level (e.g., "ACTIVE") | |
| `fashion_news_frequency` | string | varchar(15) | Email marketing preference (e.g., "NONE", "Regularly") | |
| `age` | integer | tinyint | Customer age for demographic targeting and recommendations | |
| `postal_code` | string | varchar(64) | Hashed postal code for geographic analysis while preserving privacy | |
```

- [ ] **Step 3: Commit customers table documentation**

```bash
git add docs/project-info/schema-info.md
git commit -m "docs: add customers table schema documentation

- Customer profile columns with privacy considerations
- Index recommendations for demographic queries
- Relationship mapping to transactions"
```

---

### Task 4: Document Transactions Table Schema

**Files:**
- Modify: `docs/project-info/schema-info.md`
- Analyze: `data/full/transactions_train.csv`

- [ ] **Step 1: Analyze transactions table structure**

```bash
# Examine transactions structure
head -1 data/full/transactions_train.csv | tr ',' '\n' | nl  
head -5 data/full/transactions_train.csv
tail -5 data/full/transactions_train.csv
```

- [ ] **Step 2: Add transactions table documentation**

```markdown
### transactions_train - Purchase History

Fact table containing all customer purchase transactions, forming the core behavioral data for recommendation algorithms.

**Primary Key**: Composite (`t_dat`, `customer_id`, `article_id`)
**Indexes Recommended**:
- `idx_transactions_customer_date` on (`customer_id`, `t_dat`)
- `idx_transactions_article_date` on (`article_id`, `t_dat`)
- `idx_transactions_date` on `t_dat`
- `idx_transactions_channel` on `sales_channel_id`

#### Column Specifications

| Column | Basic Type | Detailed Type | Description | Relationships |
|--------|------------|---------------|-------------|---------------|
| `t_dat` | date | date | Transaction date, critical for temporal recommendation patterns | |
| `customer_id` | string | varchar(64) | Customer identifier linking to profile data | → Links to customers.customer_id |
| `article_id` | string | varchar(20) | Product identifier for purchased item | → Links to articles.article_id |
| `price` | float | decimal(10,8) | Normalized transaction price for revenue analysis | |
| `sales_channel_id` | integer | tinyint | Sales channel (1=online, 2=store) for omnichannel analysis | |

**Data Characteristics**:
- **Volume**: ~31M transaction records
- **Date Range**: 2018-2020 (training data)
- **Price Normalization**: Values between 0-1, requires denormalization for business analysis
```

- [ ] **Step 3: Commit transactions table documentation**

```bash
git add docs/project-info/schema-info.md
git commit -m "docs: add transactions table schema documentation

- Fact table structure with composite primary key
- Performance indexes for recommendation queries
- Data characteristics and normalization notes"
```

---

### Task 5: Document Relationships and Complete Schema

**Files:**
- Modify: `docs/project-info/schema-info.md`

- [ ] **Step 1: Add relationship mapping section**

```markdown
## Entity Relationships

### Schema Architecture
The H&M dataset follows a **star schema** pattern optimized for analytical queries and recommendation algorithms:

```
    customers (137K records)
         |
         | customer_id
         |
         v
transactions_train (31M records) ← article_id → articles (105K records)
    [Fact Table]                                   [Product Dimension]
```

### Relationship Details

#### customers → transactions_train
- **Type**: One-to-Many
- **Key**: `customers.customer_id` → `transactions_train.customer_id`
- **Cardinality**: One customer can have multiple transactions
- **Business Rule**: All transactions must have valid customer_id
- **Query Pattern**: Customer purchase history, behavioral analysis

#### articles → transactions_train  
- **Type**: One-to-Many
- **Key**: `articles.article_id` → `transactions_train.article_id`
- **Cardinality**: One article can appear in multiple transactions
- **Business Rule**: All transactions must reference existing articles
- **Query Pattern**: Product popularity, inventory analysis

### Common Join Patterns

#### Customer Purchase History
```sql
SELECT c.customer_id, c.age, a.prod_name, t.t_dat, t.price
FROM customers c
JOIN transactions_train t ON c.customer_id = t.customer_id  
JOIN articles a ON t.article_id = a.article_id
WHERE c.customer_id = ?
ORDER BY t.t_dat DESC;
```

#### Product Co-Purchase Analysis
```sql
SELECT a1.prod_name, a2.prod_name, COUNT(*) as co_purchases
FROM transactions_train t1
JOIN transactions_train t2 ON t1.customer_id = t2.customer_id 
    AND t1.t_dat = t2.t_dat 
    AND t1.article_id < t2.article_id
JOIN articles a1 ON t1.article_id = a1.article_id
JOIN articles a2 ON t2.article_id = a2.article_id  
GROUP BY a1.article_id, a2.article_id
ORDER BY co_purchases DESC;
```

#### Customer Segmentation
```sql
SELECT c.age, c.club_member_status, 
       COUNT(DISTINCT t.article_id) as unique_items,
       COUNT(*) as total_purchases,
       AVG(t.price) as avg_price
FROM customers c
LEFT JOIN transactions_train t ON c.customer_id = t.customer_id
GROUP BY c.customer_id, c.age, c.club_member_status;
```

### Referential Integrity

#### Foreign Key Constraints (Production Implementation)
```sql
-- Transactions to Customers
ALTER TABLE transactions_train 
ADD CONSTRAINT fk_transactions_customer 
FOREIGN KEY (customer_id) REFERENCES customers(customer_id);

-- Transactions to Articles  
ALTER TABLE transactions_train
ADD CONSTRAINT fk_transactions_article
FOREIGN KEY (article_id) REFERENCES articles(article_id);
```

#### Cascade Behaviors
- **DELETE**: RESTRICT (prevent deletion of referenced customers/articles)
- **UPDATE**: CASCADE (update dependent transaction records)

## Database Summary

### Storage Requirements (Production)
- **articles**: ~105K rows × ~500 bytes = ~50MB
- **customers**: ~137K rows × ~200 bytes = ~25MB  
- **transactions_train**: ~31M rows × ~100 bytes = ~3GB
- **Total**: ~3.1GB base data + indexes (~1GB) = **~4GB**

### Performance Optimization
- **Partitioning**: Partition transactions_train by date (monthly/yearly)
- **Compression**: Enable row compression for historical transaction data
- **Caching**: Cache frequent article lookups and customer profiles
- **Indexes**: Implement all recommended indexes for sub-second query performance

### Recommendation System Integration
This schema directly supports:
- **Collaborative Filtering**: User-item interaction matrix from transactions
- **Content-Based Filtering**: Rich product attributes in articles table  
- **Hybrid Models**: Combined user profiles and item features
- **Temporal Patterns**: Time-series analysis via transaction dates
- **Customer Segmentation**: Demographic and behavioral clustering
```

- [ ] **Step 2: Add final summary and commit**

```bash
git add docs/project-info/schema-info.md
git commit -m "docs: complete H&M schema documentation

- Entity relationship mapping with star schema diagram
- Common join patterns for recommendation queries  
- Referential integrity constraints
- Performance optimization recommendations
- Integration guidance for ML systems"
```

- [ ] **Step 3: Final validation**

```bash
# Verify document completeness
wc -w docs/project-info/schema-info.md
grep -c "##\|###" docs/project-info/schema-info.md
```

Expected: Complete document with proper section structure

---

## Self-Review

**Spec coverage check:**
- ✅ Database overview with technical summary and scale metrics  
- ✅ Individual table schemas with dual data types (basic + detailed)
- ✅ Primary key specifications and index recommendations
- ✅ Column descriptions with concise business context
- ✅ Relationship indicators linking tables
- ✅ Entity relationship mapping and join patterns
- ✅ Performance optimization guidance

**Placeholder scan:** No TBD, TODO, or incomplete sections - all code blocks contain actual SQL and markdown content.

**Type consistency:** All data types, column names, and table references are consistent throughout the document.