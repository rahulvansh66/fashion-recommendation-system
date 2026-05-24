# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fashion recommendation system project built around the H&M dataset.

## Data Architecture

### Dataset Structure
The project uses the H&M dataset with three core tables:
- **articles.csv** (105K records): Product catalog with hierarchical classification
- **customers.csv** (1.37M records): Customer demographics and preferences  
- **transactions_train.csv** (31.8M records): Purchase transaction history

### Schema Design
- **Pattern**: Star schema with transactions as fact table
- **Relationships**: Many-to-many through transactions (customers ↔ articles)
- **Scale**: ~33M total records, ~3GB data volume
- **Recommended DB**: PostgreSQL with pgvector extension for embeddings

Detailed schema documentation is available in [docs/project-info/schema-info.md](docs/project-info/schema-info.md).

## Data Commands

### Dataset Exploration
```bash
# Check dataset sizes and structure
wc -l dataset/full/*.csv

# View file headers and sample data
head -2 dataset/full/articles.csv
head -2 dataset/full/customers.csv  
head -2 dataset/full/transactions_train.csv

# Get column structure for analysis
head -1 dataset/full/articles.csv | tr ',' '\n' | nl
```

### Data Validation
```bash
# Count unique values in key columns
cut -d',' -f1 dataset/full/articles.csv | tail -n +2 | sort | uniq | wc -l
cut -d',' -f1 dataset/full/customers.csv | tail -n +2 | sort | uniq | wc -l

# Check for data quality issues
grep -c "^," dataset/full/*.csv  # Empty first fields
grep -c ",,$" dataset/full/*.csv  # Empty last fields
```

## Documentation Structure

### Core Documentation Areas
- `docs/project-info/`: Schema specifications and technical documentation
- `docs/superpowers/`: Implementation plans and design specifications
- `docs/code-explanation-info/`: Code documentation (when code is added)
- `docs/implementation-info/`: Implementation details and decisions
- `docs/outcomes-info/`: Results and analysis outcomes

### Key Documentation Files
- `docs/project-info/schema-info.md`: Complete H&M dataset schema documentation with table structures, relationships, and SQL patterns

## Development Workflow

### Documentation Standards
- Follow the existing doc structure under `docs/`
- Include both technical specifications and business context
- Use dual data types (basic + detailed) for schema documentation
- Provide SQL examples for common query patterns

### Data Privacy
The dataset contains hashed customer identifiers and postal codes for privacy protection. Never attempt to reverse hash values or expose customer data.

### File Organization
- Raw datasets live in `dataset/full/` (ignored by version control)
- Documentation follows structured categories under `docs/`
- System design artifacts go in `system-design/`

## Recommendation System Context

This project supports multiple recommendation approaches:
- **Collaborative Filtering**: Using customer-item interaction matrix from transactions
- **Content-Based Filtering**: Leveraging rich product attributes from articles table
- **Hybrid Models**: Combining user profiles with item features
- **Temporal Analysis**: Time-series patterns from transaction dates
- **Customer Segmentation**: Demographics and behavioral clustering

When working on recommendation algorithms, reference the schema documentation for optimal query patterns and indexing strategies.