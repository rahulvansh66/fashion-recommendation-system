# Fashion Recommendation System — Architecture Overview

## Project Goal

Build a production-grade, scalable ML recommendation system on the H&M dataset. The primary learning objective is to understand modern cloud architecture patterns: serverless, event-driven, and microservices — while keeping costs minimal during development.

---

## Architecture Philosophy

| Principle | Decision |
|-----------|----------|
| **Cost-first** | Small dataset, local dev first, AWS only for final testing |
| **Migration-friendly** | Same code runs locally and on AWS — environment-driven config |
| **SageMaker-centric ML** | All ML inference goes through SageMaker for managed capabilities |
| **FAISS over OpenSearch** | Portable vector search; no managed service cost |
| **S3 as data lake** | No DynamoDB — eliminates per-read/write costs |
| **Serverless API** | FastAPI + AWS Lambda Web Adapter on Lambda — pay-per-request |

**Target total cost:** $25–40 over 2–3 months with local-first development.

---

## Dataset

### Source Tables (H&M Dataset)

| Table | Full Scale | Dev Sample | Contents |
|-------|-----------|------------|----------|
| `articles.csv` | 105K records | Derived from sampled users' transactions | Product catalog, hierarchical classification |
| `customers.csv` | 1.37M records | ~1K users (stratified sample) | Demographics, preferences |
| `transactions_train.csv` | 31.8M records | All interactions for sampled users | Purchase history |

### Schema Pattern

- **Star schema** — transactions as fact table, articles and customers as dimensions
- **Relationships** — many-to-many via transactions (customers ↔ articles)
- **Full dataset volume** — ~3GB, ~$98/month to operate
- **Dev dataset volume** — ~50MB, ~$50/month, same architectural complexity

Full schema details: [`system-design/schema-info.md`](schema-info.md)

---

## System Layers

### 1. Data Layer

```
Raw CSV (S3)
    ↓
PySpark Feature Pipeline (local / AWS Glue)
    ↓
Processed Parquet (S3)
    ↓
Redis Cache (hot user/item features)
```

- **Storage**: S3-only data lake (parquet format for all processed data)
- **Caching**: Redis for hot-path feature lookups (ElastiCache in AWS, local Redis in dev)
- **Processing**: PySpark — identical code runs locally (`local[*]`) and on AWS Glue (no changes)
- **Privacy**: All customer IDs and postal codes are pre-hashed in the source dataset

### 2. ML Layer

```
Feature Pipeline (PySpark)
        ↓
 ┌──────────────────────────────────┐
 │       Two-Tower Model Training    │
 │  (Docker + PyTorch → SageMaker)   │
 └──────────────────────────────────┘
        ↓
 ┌──────────────────────────────────┐
 │   CatBoost Ranking Model Training │
 │  (Local CatBoost → SageMaker)     │
 └──────────────────────────────────┘
        ↓
 ┌──────────────────────────────────┐
 │   FAISS Index Build               │
 │  (item embeddings → .index file)  │
 └──────────────────────────────────┘
```

#### Two-Tower Model

- **Purpose**: Learns separate embeddings for users and items; retrieval stage
- **Framework**: PyTorch
- **Local**: Docker container
- **AWS**: SageMaker Training Job → SageMaker Endpoint
- **Output**: 256-dimensional user/item embedding vectors

#### CatBoost Ranking Model

- **Purpose**: Re-ranks top-K candidates retrieved by FAISS using rich feature set
- **Framework**: CatBoost
- **Local**: Local training script
- **AWS**: SageMaker Training Job → SageMaker Endpoint
- **Input**: Candidate items + user context features
- **Output**: Ranked list of items with scores

#### FAISS Vector Index

- **Purpose**: Approximate nearest-neighbor search over item embedding space
- **Deployment**: Lambda + FAISS (industry-standard pattern for this scale)
- **Index size**: ~5MB (small dataset) to ~300MB (full dataset) — fits Lambda 10GB memory limit
- **Performance**: <1ms search latency (warm Lambda), <50ms total (including cold start)
- **Cost**: ~$0.20 per 1M requests (pay-per-use, no idle costs)
- **Rationale**: Serverless learning objective, zero infrastructure overhead, scales automatically

### 3. Inference Pipeline (Request Path)

The system has two independent request paths. They share the same Lambda + API Gateway infrastructure but have no shared code or data flow beyond that.

#### Path A — Recommendation Request (`GET /recommendations/{user_id}`)

```
Client Request  →  GET /recommendations/{user_id}
      ↓
API Gateway
      ↓
Lambda (FastAPI + AWS Lambda Web Adapter)
      ↓
 ┌─────────────────────────────────────────────┐
 │ 1. Fetch user features (Redis / S3)          │
 │ 2. SageMaker Endpoint → user embedding       │
 │    (Two-Tower user tower)                    │
 │ 3. Lambda FAISS → top-100 candidates         │
 │    (faiss_items.index — item embeddings)     │
 │ 4. SageMaker Endpoint → CatBoost re-ranking  │
 │ 5. Return top-K ranked recommendations       │
 └─────────────────────────────────────────────┘
      ↓
JSON Response
```

#### Path B — Chatbot Request (`POST /chat`)  *(future)*

```
Client Request  →  POST /chat  {"message": "something casual for summer"}
      ↓
API Gateway
      ↓
Lambda (FastAPI + AWS Lambda Web Adapter)
      ↓
 ┌─────────────────────────────────────────────┐
 │ 1. Embed user query (text encoder)           │
 │ 2. FAISS search over faiss_rag.index         │
 │    → top-K relevant product description      │
 │      chunks                                  │
 │ 3. LLM call: chunks + query → answer         │
 │ 4. Return natural language response          │
 └─────────────────────────────────────────────┘
      ↓
JSON Response  {"answer": "...", "products": [...]}
```

These two paths use **separate FAISS indices** built from different data and serving different purposes. `faiss_items.index` contains Two-Tower item embeddings; `faiss_rag.index` contains article description text chunk embeddings.

#### FAISS Lambda Optimization (Path A)

- **Cold start**: ~500ms (index loaded from S3 into Lambda memory at initialization)
- **Warm execution**: <1ms FAISS search (index cached in Lambda memory across invocations)
- **Memory allocation**: 2GB Lambda (comfortable for full dataset index + overhead)
- **Concurrent executions**: Auto-scales to handle traffic bursts without configuration

### 4. Application Layer

- **Framework**: FastAPI
- **Local**: `uvicorn` dev server in Docker container
- **AWS**: Lambda + AWS Lambda Web Adapter (zero code changes — same container runs everywhere)
- **Migration**: True zero-code-change deployment — LWA runs as Lambda extension, forwards events as HTTP to FastAPI
- **Endpoints**:
  - `GET /recommendations/{user_id}` — Two-Tower + CatBoost recommendation pipeline
  - `POST /chat` — RAG chatbot *(future)*
  - `GET /health` — health check

### 5. Infrastructure Layer

- **IaC**: Terraform (single `apply` / `destroy` cycle)
- **Local simulation**: LocalStack + Docker Compose
- **AWS services used**: S3, Lambda, API Gateway, SageMaker, ElastiCache (Redis), CloudWatch

---

## ML Pipeline — Component Table

### Core Recommendation Pipeline

| Pipeline Step | Local Dev | AWS Production | Purpose |
|---------------|-----------|----------------|---------|
| Data ingestion + preprocessing | PySpark `local[*]` | AWS Glue | Raw CSV → clean parquet |
| Feature engineering | PySpark `local[*]` | AWS Glue | Clean data → model-ready features |
| Two-Tower training | Docker + PyTorch | SageMaker Training Job | Learn user/item embeddings |
| Two-Tower inference | Local PyTorch server | SageMaker Endpoint | Generate user embedding at request time |
| Recommendation FAISS index build | Local script | Lambda | Build ANN index from item embeddings |
| Recommendation FAISS search | Local FAISS | Lambda + FAISS | Retrieve top-100 candidates |
| CatBoost training | Local CatBoost | SageMaker Training Job | Train ranking model |
| CatBoost inference | Local server | SageMaker Endpoint | Re-rank candidates |
| API orchestration | FastAPI + uvicorn | Lambda + AWS Lambda Web Adapter | Route requests, coordinate pipeline |
| Feature caching | Local Redis | ElastiCache | Hot-path user/item feature lookups |
| Storage | Local filesystem | S3 | Data lake, model artifacts, FAISS indices |

### Optional Enrichment (content_features/)

| Pipeline Step | Local Dev | AWS Production | Purpose |
|---------------|-----------|----------------|---------|
| LLM fine-tuning | Local HuggingFace + PEFT | SageMaker Training Job | Fine-tune tag extraction model |
| Tag batch inference | Local script | SageMaker Batch Transform | Article descriptions → style tags |
| User tag aggregation | Local PySpark | AWS Glue | Per-user tag features from purchase history |

### Standalone Chatbot — RAG (future)

| Pipeline Step | Local Dev | AWS Production | Purpose |
|---------------|-----------|----------------|---------|
| RAG index build | Local script | Lambda | Chunk + embed article descriptions → faiss_rag.index |
| RAG retrieval | Local FAISS | Lambda + FAISS | User query → relevant product chunks |
| LLM generation | Local LLM server | SageMaker Endpoint | Chunks + query → natural language answer |

---

## SageMaker Managed ML Capabilities

By routing all ML inference through SageMaker, the following production capabilities are available out of the box:

| Capability | How |
|------------|-----|
| A/B testing | Production variants on a single endpoint |
| Canary deployment | Gradual traffic shifting between model versions |
| Shadow testing | Run new model in shadow alongside production |
| Drift detection | SageMaker Model Monitor — data & concept drift |
| Data quality monitoring | Model Monitor data quality baselines |
| Model quality monitoring | Model Monitor model quality baselines |
| Model registry | Native SageMaker Model Registry with approval workflow |
| Endpoint metrics | Native CloudWatch integration |
| Lineage tracking | SageMaker Pipelines + Registry integration |

---

## Recommendation Approaches Supported

| Approach | Data Source | Notes |
|----------|-------------|-------|
| Collaborative filtering | Transaction history | Customer-item interaction matrix |
| Content-based filtering | Articles table | Rich product attribute embeddings |
| Hybrid (two-tower) | Transactions + articles + customers | Primary production approach |
| Temporal analysis | Transaction dates | Seasonal patterns, recency weighting |
| Customer segmentation | Customers table | Demographics + behavioral clustering |

---

## Local-to-AWS Migration Patterns

**Core principle:** Write all local code as if it is already running on AWS. The only differences between local and production are configuration values — never code logic. This makes AWS migration a config change, not a rewrite.

### Quick Reference

| Pattern | Local | AWS |
|---------|-------|-----|
| PySpark | `master("local[*]")` | AWS Glue (drop `master()`) |
| S3 data paths | `./data/file.parquet` | `s3://bucket/file.parquet` |
| Redis | `localhost:6379` | ElastiCache endpoint |
| boto3 | `endpoint_url='http://localhost:4566'` (LocalStack) | No `endpoint_url` |
| FastAPI | `uvicorn` in Docker | AWS Lambda Web Adapter (same container) |
| SageMaker SDK | `instance_type='local'` | `instance_type='ml.m5.large'` |
| FAISS index | Local `.index` file | S3-backed, loaded into Lambda memory |

---

### Pattern 1 — AWS SDK from Day 1 + LocalStack

Use `boto3` everywhere from the start. LocalStack intercepts calls locally; AWS intercepts them in production. The only change is removing `endpoint_url`.

```python
import boto3
import os

# Local development — LocalStack intercepts all calls
s3 = boto3.client('s3', endpoint_url='http://localhost:4566')

# AWS production — remove endpoint_url, everything else identical
s3 = boto3.client('s3')

# Identical operations in both environments:
s3.upload_file('local_file.csv', 'my-bucket', 'data/file.csv')
s3.download_file('my-bucket', 'data/file.csv', '/tmp/file.csv')
```

Wrap this in a factory so the switch is automatic:

```python
# aws_client.py
import boto3
import os

def get_s3_client():
    if os.getenv('ENV') == 'local':
        return boto3.client('s3', endpoint_url='http://localhost:4566')
    return boto3.client('s3')  # AWS picks up IAM role automatically
```

---

### Pattern 2 — Environment-Driven Configuration

All environment-specific values come from environment variables with local defaults. No hardcoded paths or hostnames anywhere in the codebase.

```python
# config.py — same file, different behavior per environment
import os

DATA_PATH          = os.getenv('DATA_PATH',          './data/')
FAISS_INDEX_PATH   = os.getenv('FAISS_PATH',         './faiss_index')
MODEL_ENDPOINT     = os.getenv('SAGEMAKER_ENDPOINT', 'http://localhost:8080')
REDIS_HOST         = os.getenv('REDIS_HOST',         'localhost')
REDIS_PORT         = int(os.getenv('REDIS_PORT',     '6379'))
S3_BUCKET          = os.getenv('S3_BUCKET',          'local-dev-bucket')
AWS_ENV            = os.getenv('ENV',                'local')  # 'local' | 'aws'
```

Local `.env`:
```
ENV=local
DATA_PATH=./data/
REDIS_HOST=localhost
```

AWS Lambda environment variables:
```
ENV=aws
DATA_PATH=s3://my-fashion-bucket/data/
REDIS_HOST=my-cluster.cache.amazonaws.com
SAGEMAKER_ENDPOINT=https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/two-tower/invocations
```

---

### Pattern 3 — SageMaker Python SDK for Local Testing

Define SageMaker Training Jobs and Endpoints locally using `instance_type='local'`. The same job definition runs on real AWS instances with a single config change.

```python
from sagemaker.pytorch import PyTorch
import os

local_dev = os.getenv('ENV') == 'local'

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src/models/two_tower/',
    role='arn:aws:iam::123456789:role/SageMakerRole',
    framework_version='2.0',
    py_version='py310',
    instance_type='local' if local_dev else 'ml.m5.large',
    instance_count=1,
    hyperparameters={
        'epochs': 10,
        'embedding_dim': 256,
        'batch_size': 1024,
    }
)

estimator.fit({'train': 's3://my-bucket/data/train/'})
```

The exact same script runs locally in a Docker container and on AWS SageMaker — no code changes needed.

---

### Pattern 4 — Redis Protocol Compatibility

Redis commands are identical whether connecting to a local Redis container or AWS ElastiCache. Only the hostname changes.

```python
import redis
import os

r = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', '6379')),
    decode_responses=True
)

# Identical operations — local and AWS:
r.set('user:1234:embedding', embedding_bytes, ex=3600)  # TTL 1 hour
r.get('user:1234:embedding')
r.hset('item:5678:features', mapping={'price': '29.99', 'category': 'tops'})
```

Local Docker Compose:
```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

AWS: ElastiCache Redis cluster — same port, same commands.

---

### Pattern 5 — PySpark for Data Processing

PySpark runs in local mode during development and on AWS Glue in production. No code changes — only the `SparkSession` initialization differs, and even that can be unified with a factory.

```python
from pyspark.sql import SparkSession
import os

def get_spark():
    builder = SparkSession.builder.appName('FashionFeaturePipeline')
    if os.getenv('ENV') == 'local':
        builder = builder.master('local[*]')
    # On AWS Glue, master() is not set — Glue manages the cluster
    return builder.getOrCreate()

spark = get_spark()

# Identical DataFrame operations everywhere:
transactions = spark.read.parquet(os.getenv('DATA_PATH') + 'transactions/')
user_features = transactions.groupBy('customer_id').agg(...)
user_features.write.parquet(os.getenv('DATA_PATH') + 'user_features/')
```

---

### Pattern 6 — FastAPI → Lambda with AWS Lambda Web Adapter

Write the API as a standard FastAPI app. The same application code and Docker container runs locally and on Lambda with **absolutely zero code modifications**. AWS Lambda Web Adapter (a Lambda extension) runs as a sidecar process that intercepts Lambda events and forwards them as standard HTTP requests to your FastAPI server.

```python
# api/main.py
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title='Fashion Recommendation API')

@app.get('/recommendations/{user_id}')
async def get_recommendations(user_id: str, k: int = 10):
    # Same business logic regardless of environment
    candidates = faiss_search(user_id, top_k=100)
    ranked     = catboost_rank(user_id, candidates, top_k=k)
    return {'user_id': user_id, 'recommendations': ranked}

# Local development
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

**Dockerfile (same for local and AWS):**

```dockerfile
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ .

# Install AWS Lambda Web Adapter as a Lambda Extension
# Only active when running on Lambda; no effect locally
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Environment variables for LWA (Lambda ignores PORT, uses adapter)
ENV PORT=8000

# Same command for local and Lambda - no conditional logic needed
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key advantages:**
- **Zero code changes**: Application never imports or references any Lambda-specific code
- **Language agnostic**: Works with any HTTP server (Node.js, Go, Rust, Java, etc.)
- **True portability**: Same container image runs locally, on ECS, on Lambda — anywhere
- **No lock-in**: Application has zero AWS dependencies; pure FastAPI code

---

### Pattern 7 — Docker Containers for Everything

All services run in Docker locally. The same images deploy to AWS ECS or Lambda container images — no environment-specific Dockerfiles.

```dockerfile
# Dockerfile — same for local and AWS
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

# Local: override CMD via docker-compose
# AWS Lambda: set CMD to handler in function config
CMD ["python", "api/main.py"]
```

Local Docker Compose:
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      ENV: local
      DATA_PATH: ./data/
    volumes:
      - ./data:/app/data
  redis:
    image: redis:7-alpine
  localstack:
    image: localstack/localstack
    ports: ["4566:4566"]
    environment:
      SERVICES: s3,lambda,apigateway
```

---

## Development Workflow

### Phase 1 — Local Development ($0 AWS cost)
1. Set up Docker Compose: LocalStack + Redis + local Spark
2. Build and validate feature pipeline on small dataset
3. Train Two-Tower and CatBoost models locally
4. Build FAISS index and validate retrieval quality
5. Run full inference pipeline end-to-end locally

### Phase 2 — AWS Deployment (cost incurred only during active sessions)
1. `terraform apply` — spin up all AWS resources
2. Upload processed data and model artifacts to S3
3. Register models in SageMaker Model Registry
4. Deploy SageMaker endpoints (Two-Tower + CatBoost)
5. Deploy Lambda (FAISS search + FastAPI API)
6. Validate end-to-end on AWS
7. `terraform destroy` — tear down everything when done

### Cost Controls
- SageMaker endpoints only active during active testing sessions
- Terraform destroy between development sessions
- Spot instances for training jobs
- Free tier maximization for S3, Lambda, CloudWatch

---

## Directory Structure

Full directory structure with rationale is documented in [`system-design/project-structure.md`](project-structure.md).

---

## Cost Summary

| Phase | Estimated Cost |
|-------|---------------|
| Local development | $0 |
| AWS testing sessions (total) | $25–40 |
| Full dataset operation | ~$98/month |
| Small dataset operation | ~$50/month |
| **Target total (2–3 months)** | **$25–40** |
