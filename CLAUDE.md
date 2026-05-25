# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fashion recommendation system project built around the H&M dataset.

### Learning Objectives
**Primary Goal:** Learn to build production-grade, scalable ML systems using modern cloud architecture patterns.

**Key Focus Areas:**
- **System Architecture:** Serverless, event-driven, microservices patterns
- **ML Engineering:** Two-tower models, vector databases, feature pipelines
- **AWS Ecosystem:** SageMaker, Lambda, OpenSearch, DynamoDB integration
- **Production Patterns:** CI/CD, monitoring, caching strategies, security

### Cost Optimization Strategy
**Budget Constraint:** Minimize costs while maintaining production-ready architecture patterns.

**Cost Control Approaches:**
- **Small Dataset First:** Start with subset of H&M data (10K users vs 1.37M, 5K items vs 105K)
- **Simplified Architecture:** Skip DynamoDB, use S3 + FAISS + SageMaker-centric ML serving
- **Local-to-AWS Migration:** Develop locally ($0 cost), migrate to AWS for final deployment
- **Development-Only Usage:** Turn off services between development sessions
- **Free Tier Maximization:** Leverage AWS free tier limits extensively
- **Right-Sizing:** Use minimal instance sizes that demonstrate patterns
- **Cleanup Strategy:** Delete resources after learning objectives achieved

**Architecture Decision:** SageMaker-centric ML inference for full managed ML capabilities (A/B testing, canary deployment, model monitoring).

**Target:** $25-40 total learning cost over 2-3 months with local development approach.

## Data Architecture

### Dataset Structure
The project uses the H&M dataset with three core tables:
- **articles.csv** (105K records): Product catalog with hierarchical classification
- **customers.csv** (1.37M records): Customer demographics and preferences  
- **transactions_train.csv** (31.8M records): Purchase transaction history

### Development Dataset Strategy
**Full Dataset** (Production-scale learning):
- **Scale**: ~33M total records, ~3GB data volume
- **Cost**: ~$98/month operational costs

**Small Dataset** (Cost-optimized learning - RECOMMENDED):
- **Users**: 10K subset (0.7% sample with diverse demographics)
- **Articles**: 5K items (top sellers + category diversity)
- **Transactions**: 100K interactions (recent, representative patterns)
- **Scale**: ~50MB total, same architecture complexity
- **Cost**: ~$50/month operational, ~$60-80 total learning cost

### Schema Design
- **Pattern**: Star schema with transactions as fact table
- **Relationships**: Many-to-many through transactions (customers ↔ articles)
- **Storage Strategy**: S3-based data lake with Redis caching (DynamoDB eliminated for cost optimization)
- **Vector Search**: FAISS for similarity search (portable local-to-AWS)
- **ML Serving**: SageMaker endpoints for Two-Tower and CatBoost inference (full managed ML capabilities)
- **Scaling**: Architecture identical for small/full dataset (configuration change only)

Detailed schema documentation is available in [system-design/schema-info.md](system-design/schema-info.md).

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
- `docs/ref-project-info/`: **REFERENCE PROJECT ONLY** - Contains old code and documentation from previous implementation. Only consult when explicitly asked to reference legacy code.
- `system-design/`: **CURRENT IMPLEMENTATION TARGET** - All current project info, plans, designs, and documentation for what we're building now.
- `docs/superpowers/`: Implementation plans and design specifications
- `docs/code-explanation-info/`: Code documentation (when code is added)
- `docs/implementation-info/`: Implementation details and decisions
- `docs/outcomes-info/`: Results and analysis outcomes

### Key Documentation Files
- `system-design/schema-info.md`: Complete H&M dataset schema documentation with table structures, relationships, and SQL patterns

### Important: Reference vs Current Project
- **Reference Project** (`docs/ref-project-info/`): Archived implementation - DO NOT use unless explicitly asked to reference old code
- **Current Project** (`system-design/`): Active development target - use for all current work, planning, and project information
- When asked to "refer to reference project", only then consult `docs/ref-project-info/` as legacy reference

## Development Workflow

### Cost-Conscious Development
**Local Development First:**
- Complete development locally with $0 AWS costs
- Use LocalStack + Docker for AWS service simulation
- Migrate to AWS only for final testing and portfolio demonstration

**Session-Based AWS Usage:**
- Turn off expensive services between development sessions
- Use Terraform for quick infrastructure spin-up/tear-down
- Leverage spot instances and auto-scaling for cost optimization

**Smart Resource Usage:**
- Start with free tier services where possible
- SageMaker endpoints only during active testing
- Document cost-optimization decisions for future reference

### Documentation Standards
- Follow the existing doc structure under `docs/`
- Document cost implications of architectural decisions
- Track actual vs estimated costs for learning

### File Organization
- Raw datasets live in `dataset/full/` (ignored by version control)
- Small dataset samples in `dataset/sample/` for cost-optimized development


### Data Privacy
The dataset contains hashed customer identifiers and postal codes for privacy protection. Never attempt to reverse hash values or expose customer data.

## Recommendation System Context

This project supports multiple recommendation approaches:
- **Collaborative Filtering**: Using customer-item interaction matrix from transactions
- **Content-Based Filtering**: Leveraging rich product attributes from articles table
- **Hybrid Models**: Combining user profiles with item features
- **Temporal Analysis**: Time-series patterns from transaction dates
- **Customer Segmentation**: Demographics and behavioral clustering

When working on recommendation algorithms, reference the schema documentation for optimal query patterns and indexing strategies.

## System Architecture

### Production ML Pipeline Architecture
**Pipeline Components:**

| Pipeline Step | Local Development | AWS Production | What It Does |
|---------------|-------------------|----------------|--------------|
| **Two-Tower Model Training** | Docker + PyTorch | **SageMaker** | Train user/item embedding model |
| **Two-Tower Inference** | Local PyTorch model | **SageMaker Endpoint** | Generate embeddings from user/item features |
| **FAISS Index Building** | Local FAISS index | **Lambda** | Create searchable vector index |
| **FAISS Similarity Search** | Local FAISS search | **Lambda + FAISS** | Find similar items using embeddings |
| **CatBoost Training** | Local CatBoost | **SageMaker** | Train ranking/scoring model |
| **CatBoost Inference** | Local serving | **SageMaker Endpoint** | Score and rank candidates |
| **API Orchestration** | FastAPI | **Lambda + Mangum** | Coordinate pipeline and business logic |

### SageMaker-Centric ML Capabilities
**Managed ML Features Available:**

| Capability | SageMaker Benefit |
|------------|-------------------|
| **A/B testing** | Built-in using production variants |
| **Canary deployment** | Supported through deployment/update patterns |
| **Shadow testing** | Supported for testing new model variants |
| **Drift monitoring** | SageMaker Model Monitor |
| **Data quality monitoring** | SageMaker Model Monitor |
| **Model quality monitoring** | SageMaker Model Monitor |
| **Model registry integration** | Native SageMaker Model Registry |
| **Endpoint metrics** | Native CloudWatch metrics |
| **Approval workflow** | Native model package approval states |
| **Lineage** | SageMaker jobs, registry, pipelines integration |

## Migration-Friendly Development Patterns

### Local-to-AWS Migration Strategy
**Zero-Migration-Effort Patterns:**

1. **PySpark for Data Processing** ✅
```python
# Same PySpark code works everywhere
from pyspark.sql import SparkSession

# Local development (single machine)
spark = SparkSession.builder.master("local[*]").getOrCreate()

# AWS Glue (managed cluster) - NO CODE CHANGES
spark = SparkSession.builder.getOrCreate()
```

2. **FAISS Instead of OpenSearch** ✅
```python
# Local FAISS (same code everywhere)
import faiss
index = faiss.IndexFlatIP(256)
scores, indices = index.search(query_embedding, k=100)

# AWS deployment: Lambda + FAISS (selected)
# Same FAISS API, index loaded from S3 into Lambda memory
# - <1ms search latency (warm Lambda)
# - Auto-scales without configuration
# - Pay-per-request (~$0.20 per 1M requests)
```

3. **Terraform for Infrastructure** ✅
```bash
# One command: spin up entire AWS stack
terraform apply

# One command: tear down everything  
terraform destroy
```

4. **S3-Only Data Storage** ✅
```python
# Local development
df = pd.read_parquet("./data/users.parquet")

# AWS (same code, different path)  
df = pd.read_parquet("s3://bucket/users.parquet")
```

5. **AWS SDK from Day 1 + LocalStack** ✅
```python
import boto3

# Local development with LocalStack
s3 = boto3.client('s3', endpoint_url='http://localhost:4566')

# AWS production (same code, no endpoint)
s3 = boto3.client('s3')
```

6. **Docker Containers for Everything** ✅
```dockerfile
# Same container locally and on AWS ECS/Lambda
FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
CMD ["python", "app.py"]
```

7. **FastAPI → Lambda with Mangum** ✅
```python
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    return {"recommendations": []}

# Local development
if __name__ == "__main__":
    uvicorn.run(app)

# AWS Lambda (zero code changes)
lambda_handler = Mangum(app)
```

8. **Environment-Driven Configuration** ✅
```python
# config.py - Same code, different behavior
import os

DATA_PATH = os.getenv('DATA_PATH', './data/')
FAISS_INDEX_PATH = os.getenv('FAISS_PATH', './faiss_index')
MODEL_ENDPOINT = os.getenv('SAGEMAKER_ENDPOINT', 'http://localhost:8080')

# Local: DATA_PATH=./data/
# AWS: DATA_PATH=s3://bucket/data/
```

9. **SageMaker Python SDK for Local Development** ✅
```python
from sagemaker.pytorch import PyTorch

# Works locally for testing job definitions
estimator = PyTorch(
    entry_point='train.py',
    role='arn:aws:iam::...',
    instance_type='local' if local_dev else 'ml.m5.large'
)
```

10. **Redis Protocol Compatibility** ✅
```python
import redis

# Local Redis
r = redis.Redis(host='localhost', port=6379)

# AWS ElastiCache (same Redis commands!)
r = redis.Redis(host='cache-cluster.amazonaws.com', port=6379)

# Identical operations
r.set('key', 'value')
r.get('key')
```

### FAISS Deployment Decision

**Selected: Lambda + FAISS** ✅

Given the SageMaker-centric ML architecture and learning objectives, Lambda + FAISS is the optimal choice:

**Why Lambda + FAISS:**
- **Index size**: ~5MB (small dataset) to ~300MB (full dataset) — fits Lambda 10GB memory limit
- **Cost**: Pay-per-request (~$0.20 per 1M requests), no idle costs
- **Latency**: <1ms search (warm Lambda), ~500ms cold start
- **Scalability**: Auto-scales to handle traffic bursts without configuration
- **Learning objective**: Industry-standard serverless pattern for this scale
- **Integration**: Seamless with SageMaker endpoints and API Gateway
- **Ops overhead**: Zero — fully managed, no infrastructure to maintain

**When to consider alternatives:**
- **ECS containers**: Only if index grows >10GB or sustained traffic >10K req/s
- **SageMaker custom container**: Only for unified ML platform preference (adds ~$50+/month cost)