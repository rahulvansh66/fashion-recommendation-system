# Project Structure

## Organization Principles

| Principle | Decision |
|-----------|----------|
| **Pipeline-stage layout** | Top-level `src/` directories map to stages in the recommendation funnel |
| **Core pipeline** | `data_pipeline/` → `feature_pipeline/` → `retrieval/` → `ranking/`. These four are always required. |
| **Optional enrichment** | `content_features/` sits between `data_pipeline/` and `feature_pipeline/` but is not mandatory. Core pipeline works without it. |
| **Standalone features** | `generation/` is a separate chatbot feature. It is NOT a funnel stage — it runs independently of the recommendation pipeline. |
| **Entity-based within steps** | Inside each step, files are organized by table/model (e.g., `articles.py`, `customers.py`) |
| **Train vs Inference split** | Every model has separate `train.py` (SageMaker Training Job) and `inference.py` (SageMaker Endpoint handler) |
| **Single config entrypoint** | All `os.getenv()` calls live only in `src/config.py` — nowhere else |
| **AWS client factory** | All `boto3.client()` creation goes through `src/shared/aws_clients.py` — LocalStack-aware |
| **Notebooks are isolated** | Notebooks never import from `src/` and never become production code |
| **Split requirements** | Training, content enrichment, and serving deps are separate requirements files |

---

## Directory Structure

```
fashion-recommendation-system/
│
├── src/                                        # All production application code
│   ├── config.py                               # Single env-driven config — all os.getenv() lives here
│   │
│   ├── shared/                                 # Cross-cutting infrastructure concerns
│   │   ├── __init__.py
│   │   ├── aws_clients.py                      # boto3 factory (LocalStack-aware)
│   │   ├── s3_utils.py                         # Upload/download helpers, presigned URLs
│   │   ├── redis_client.py                     # Redis connection singleton
│   │   └── logging.py                          # Structured logging setup
│   │
│   ├── data_pipeline/                          # CORE: Raw → Clean  (runs once per data refresh)
│   │   ├── __init__.py
│   │   ├── ingestion/                          # Step 1: Load raw CSVs into S3 parquet
│   │   │   ├── __init__.py
│   │   │   └── h_and_m.py                     # Read raw CSVs, write parquet with no transforms
│   │   ├── preprocessing/                      # Step 2: Clean + join → validated parquet
│   │   │   ├── __init__.py
│   │   │   ├── articles.py                     # Null fills, type casting, category normalization
│   │   │   ├── customers.py                    # Age bucketing, missing demographic handling
│   │   │   ├── transactions.py                 # Date parsing, dedup, price/date filtering
│   │   │   └── joiner.py                       # Joins all three tables into unified fact table
│   │   └── validation/                         # Schema contracts after each step
│   │       ├── __init__.py
│   │       └── schemas.py
│   │
│   ├── content_features/                       # OPTIONAL: Text → Structured features  (offline, periodic)
│   │   ├── __init__.py                         # Not required for core pipeline; enriches feature_pipeline/ when enabled
│   │   ├── finetuning/                         # Train the tag extraction model (runs rarely — manual trigger)
│   │   │   ├── __init__.py
│   │   │   ├── dataset_prep.py                 # Build fine-tuning dataset from purchase history + article text
│   │   │   ├── train.py                        # SageMaker Training Job entry point (HF + PEFT/LoRA)
│   │   │   ├── evaluate.py                     # Tag quality evaluation (precision, coverage)
│   │   │   └── Dockerfile                      # SageMaker custom container (transformers + peft)
│   │   └── batch_inference/                    # Run fine-tuned model on all articles (runs periodically)
│   │       ├── __init__.py
│   │       ├── tag_extractor.py                # Batch inference: article description → tags
│   │       │                                   #   input:  s3://bucket/processed/articles_clean.parquet
│   │       │                                   #   output: s3://bucket/enriched/article_tags.parquet
│   │       └── user_tag_aggregator.py          # Aggregate per-user tags from purchase history
│   │                                           #   input:  enriched/article_tags.parquet
│   │                                           #           processed/transactions_clean.parquet
│   │                                           #   output: s3://bucket/enriched/user_tag_features.parquet
│   │
│   ├── feature_pipeline/                       # CORE: Clean → Model-ready signals  (runs per training cycle)
│   │   ├── __init__.py                         # Consumes s3://bucket/processed/ (always) + s3://bucket/enriched/ (if content_features enabled)
│   │   ├── spark_session.py                    # SparkSession factory (local[*] vs Glue)
│   │   ├── user_features.py                    # Purchase aggregates; optionally includes tag features if enriched/ exists
│   │   ├── item_features.py                    # Popularity scores, category stats; optionally includes article tags
│   │   ├── interaction_features.py             # User-item interaction matrix
│   │   └── ranking_features.py                 # CatBoost candidate feature assembly (user × item)
│   │
│   ├── retrieval/                              # CORE: Stage 1 — Candidate retrieval
│   │   ├── __init__.py
│   │   ├── two_tower/                          # Two-Tower embedding model (PyTorch)
│   │   │   ├── __init__.py
│   │   │   ├── model.py                        # nn.Module definition (imported by train + inference)
│   │   │   ├── dataset.py                      # PyTorch Dataset class
│   │   │   ├── train.py                        # SageMaker Training Job entry point
│   │   │   ├── inference.py                    # SageMaker inference handler (model_fn, predict_fn)
│   │   │   └── Dockerfile                      # SageMaker custom container
│   │   └── faiss/
│   │       ├── __init__.py
│   │       ├── builder.py                      # Build recommendation .index file from item embeddings
│   │       └── searcher.py                     # ANN search — shared by local runner + Lambda
│   │
│   ├── ranking/                                # CORE: Stage 2 — Re-ranking
│   │   ├── __init__.py
│   │   └── catboost/
│   │       ├── __init__.py
│   │       ├── model.py                        # CatBoostClassifier wrapper
│   │       ├── train.py                        # SageMaker Training Job entry point
│   │       ├── inference.py                    # SageMaker inference handler
│   │       └── feature_builder.py              # Assemble candidate feature matrix at inference time
│   │
│   ├── generation/                             # STANDALONE CHATBOT — NOT part of the recommendation funnel  (future)
│   │   ├── __init__.py
│   │   └── rag/                                # RAG: answers user natural language queries about products
│   │       ├── __init__.py                     # Triggered by POST /chat; shares nothing with recommendation pipeline
│   │       ├── ingestion.py                    # Chunk + embed all article descriptions → separate FAISS index
│   │       ├── retriever.py                    # Semantic search: user query → relevant product description chunks
│   │       └── generator.py                    # LLM call: retrieved chunks + query → natural language answer
│   │
│   └── serving/                                # API + Lambda handlers
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py                         # FastAPI app definition (zero Lambda-specific code; LWA handles Lambda integration)
│       │   ├── routers/
│       │   │   ├── recommendations.py          # GET /recommendations/{user_id}  — Two-Tower + CatBoost pipeline
│       │   │   ├── chat.py                     # POST /chat  — RAG chatbot (future)
│       │   │   └── health.py                   # GET /health
│       │   └── services/
│       │       ├── recommendation_service.py   # Orchestrates full retrieval → ranking pipeline
│       │       ├── feature_service.py          # Fetches user/item features from Redis / S3
│       │       └── cache_service.py            # Redis read/write abstraction
│       └── lambdas/
│           ├── api_handler.py                  # Lambda entry point — imports serving/api/main.py
│           ├── faiss_handler.py                # Lambda entry point — recommendation FAISS search only
│           └── chat_handler.py                 # Lambda entry point — RAG chatbot (future)
│
├── pipelines/                                  # Orchestration scripts
│   │                                           # Core steps (always run in this order):
│   ├── run_data_pipeline.py                    # Step 1: Ingestion + preprocessing end-to-end
│   ├── run_feature_pipeline.py                 # Step 2: Feature engineering (depends on step 1)
│   ├── run_training_pipeline.py                # Step 3: Train two-tower + CatBoost (depends on step 2)
│   ├── run_index_pipeline.py                   # Step 4: Build recommendation FAISS index + upload to S3
│   │                                           # Optional (run before step 2 to enrich features):
│   ├── run_content_features.py                 # Optional: LLM batch inference → enriched/ in S3
│   │                                           # Standalone (independent of recommendation pipeline):
│   └── run_rag_pipeline.py                     # Standalone (FUTURE): Build RAG product knowledge index
│
├── notebooks/                                  # Exploration and experiments only
│   ├── 01_eda_articles.ipynb                   # Never import from src/ — read-only consumers
│   ├── 02_eda_transactions.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_two_tower_experiments.ipynb
│   └── 05_catboost_experiments.ipynb
│
├── terraform/                                  # Infrastructure as Code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── modules/
│   │   ├── s3/
│   │   ├── lambda/
│   │   ├── api_gateway/
│   │   ├── sagemaker/
│   │   └── elasticache/
│   └── environments/
│       ├── local/                              # LocalStack tfvars
│       └── aws/                               # AWS production tfvars
│
├── docker/
│   ├── docker-compose.yml                      # Full local dev stack (API + Redis + LocalStack)
│   ├── docker-compose.test.yml                 # Integration test stack
│   └── localstack/
│       └── init-scripts/                       # Auto-create S3 buckets at LocalStack startup
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_content_features.py            # Tag extractor + user tag aggregator
│   │   ├── test_feature_pipeline.py
│   │   ├── test_two_tower.py
│   │   ├── test_catboost.py
│   │   └── test_api.py
│   ├── integration/
│   │   ├── test_faiss_search.py
│   │   └── test_recommendation_pipeline.py     # End-to-end with LocalStack
│   └── conftest.py                             # Shared fixtures (LocalStack, mock Redis)
│
├── scripts/
│   ├── create_sample_dataset.py                # Generate dev subset from full CSVs
│   ├── seed_localstack.py                      # Bootstrap LocalStack with test data
│   ├── upload_artifacts_to_s3.py               # Push models + FAISS index to S3
│   └── deploy_sagemaker_endpoint.py            # Register + deploy SageMaker endpoint
│
├── dataset/
│   ├── full/                                   # Raw H&M CSVs (gitignored)
│   └── sample/                                 # 10K users, 5K items, 100K transactions
│
├── system-design/                              # Architecture documentation
│   ├── infrastructure-layer.md
│   ├── schema-info.md
│   └── project-structure.md                    # This file
│
├── docs/
│   ├── ref-project-info/                       # Legacy reference (archive only)
│   ├── implementation-info/
│   └── outcomes-info/
│
├── .env.local                                  # Local dev env vars (gitignored)
├── .env.example                                # Template — safe to commit
├── pyproject.toml                              # Packaging, tool config (black, ruff, mypy, pytest)
├── requirements-training.txt                  # torch, catboost, pyspark, sagemaker-sdk (~2GB)
├── requirements-content.txt                   # transformers, peft, datasets, accelerate (~5GB)
├── requirements-serving.txt                   # fastapi, uvicorn, redis, boto3, faiss-cpu (~50MB; no mangum needed)
├── requirements-dev.txt                       # pytest, black, ruff, mypy, localstack
├── Makefile                                   # One-line dev commands
└── CLAUDE.md
```

---

## S3 Data Lineage

Each pipeline stage owns a distinct S3 prefix. Downstream stages consume from the previous stage's output — they never skip a level or read from an earlier stage's prefix directly.

```
s3://bucket/
├── raw/                        ← data_pipeline/ingestion/ writes here
│   ├── articles.parquet
│   ├── customers.parquet
│   └── transactions.parquet
│
├── processed/                  ← data_pipeline/preprocessing/ writes here
│   ├── articles_clean.parquet
│   ├── customers_clean.parquet
│   ├── transactions_clean.parquet
│   └── unified_fact.parquet
│
├── enriched/                   ← content_features/batch_inference/ writes here  (OPTIONAL)
│   ├── article_tags.parquet         # article_id → [tag1, tag2, ...]
│   └── user_tag_features.parquet    # customer_id → aggregated tag feature vector
│
├── features/                   ← feature_pipeline/ writes here
│   ├── user_features.parquet        # always present; includes tag features only if enriched/ exists
│   ├── item_features.parquet
│   ├── interaction_features.parquet
│   └── ranking_features.parquet
│
├── models/                     ← SageMaker Training Jobs write here
│   ├── two_tower/
│   │   └── model.tar.gz
│   ├── catboost/
│   │   └── model.tar.gz
│   └── content_features/            # fine-tuned tag extraction model  (optional)
│       └── model.tar.gz
│
└── indices/                    ← pipeline scripts write here
    ├── faiss_items.index            # recommendation ANN index (run_index_pipeline.py)
    └── faiss_rag.index              # RAG product knowledge index (run_rag_pipeline.py — future)
```

---

## Who Consumes What

| Consumer | Reads From |
|----------|-----------|
| `content_features/batch_inference/tag_extractor.py` | `s3://bucket/processed/articles_clean.parquet` |
| `content_features/batch_inference/user_tag_aggregator.py` | `enriched/article_tags.parquet` + `processed/transactions_clean.parquet` |
| `feature_pipeline/` | `s3://bucket/processed/` (always) + `s3://bucket/enriched/` (when content_features enabled) |
| `retrieval/two_tower/train.py` | `s3://bucket/features/` |
| `ranking/catboost/train.py` | `s3://bucket/features/ranking_features.parquet` |
| `retrieval/faiss/builder.py` | Item embeddings from two-tower inference |
| `generation/rag/ingestion.py` | `s3://bucket/processed/articles_clean.parquet` (direct — bypasses feature_pipeline) |
| `serving/api/routers/recommendations.py` | Redis (hot features) + `indices/faiss_items.index` + SageMaker endpoints |
| `serving/api/routers/chat.py` | `indices/faiss_rag.index` + LLM endpoint |

---

## `requirements` Split Rationale

Lambda cold start time and package size are directly linked. PyTorch alone is ~700MB — it cannot be in a Lambda deployment package.

| File | Contents | Used By |
|------|----------|---------|
| `requirements-training.txt` | torch, catboost, pyspark, sagemaker-sdk | `retrieval/two_tower/`, `ranking/catboost/`, `feature_pipeline/` |
| `requirements-content.txt` | transformers, peft, datasets, accelerate | `content_features/` only — ~5GB, must stay isolated |
| `requirements-serving.txt` | fastapi, uvicorn, redis, boto3, faiss-cpu, httpx | Lambda (API + FAISS), local API server |
| `requirements-dev.txt` | pytest, black, ruff, mypy, moto, localstack | Local development and CI only |

`content_features/` gets its own requirements file because HuggingFace + PEFT must not pollute the PySpark training environment or the Lambda serving package.

---

## `train.py` vs `inference.py` Split

Every model directory contains both. They are deployed to completely different environments.

| File | Deployed To | Dependencies | Entry Point |
|------|-------------|-------------|-------------|
| `train.py` | SageMaker Training Job | `requirements-training.txt` | `estimator.fit()` |
| `inference.py` | SageMaker Endpoint | lightweight subset | `model_fn`, `predict_fn` |
| `model.py` | Both (imported) | shared architecture only | — |

---

## Preprocessing Organization: Step-based vs Entity-based

**Rule: step-based at the pipeline directory level, entity-based within each step.**

`data_pipeline/preprocessing/` is organized by **entity** (not by operation type) because each table's cleaning logic is entirely independent — article category normalization, customer demographic handling, and transaction temporal filtering share no code. A `cleaning.py` file containing all three would be a grab-bag.

```python
# preprocessing/transactions.py — ONLY cleaning, no model-specific logic
def clean_transactions(df):
    return (
        df
        .filter(col("t_dat") <= current_date())
        .dropDuplicates(["customer_id", "article_id", "t_dat"])
        .withColumn("price", col("price").cast("float"))
        .filter(col("price") > 0)
    )

# feature_pipeline/user_features.py — model signals, trusts input is already clean
def build_user_features(clean_transactions_df):
    return (
        clean_transactions_df
        .groupBy("customer_id")
        .agg(
            count("article_id").alias("total_purchases"),
            countDistinct("article_id").alias("unique_items"),
            max("t_dat").alias("last_purchase_date"),
            avg("price").alias("avg_spend"),
        )
    )
```

`preprocessing/transactions.py` knows nothing about models. `feature_pipeline/user_features.py` knows nothing about nulls or type casting.

---

## `content_features/` vs `generation/rag/`: Two Different LLM Uses

Both involve LLMs but are entirely different in purpose, timing, and placement in the system.

| | `content_features/` | `generation/rag/` |
|--|--------------------|--------------------|
| **Purpose** | Extract structured style tags from article text → features for user profile | Answer user natural language queries about products via a chatbot panel |
| **Relation to recommendations** | Optional enrichment — feeds Two-Tower + CatBoost as richer input features | Completely independent — shares no code or data flow with recommendation pipeline |
| **When it runs** | Offline batch, periodically (e.g. weekly cron) | Online, per chat message |
| **Trigger** | `pipelines/run_content_features.py` | `POST /chat` API call |
| **Output** | `s3://bucket/enriched/user_tag_features.parquet` → consumed by `feature_pipeline/` | Natural language answer or product list returned to chatbot UI |
| **On the recommendation request path?** | No | No |
| **Mandatory for core pipeline?** | No — core pipeline works without it | No — fully standalone |
| **Dependencies** | transformers, peft, accelerate | LLM inference SDK + vector search |

---

## RAG Chatbot: What It Is and How It Fits

**User-facing behaviour:** A chatbot panel alongside the product feed. The user can ask natural language questions — "show me something casual for summer" or "what's good for a wedding?". The chatbot searches a vector index of all product descriptions, retrieves the most relevant chunks, and passes them to an LLM to generate a grounded answer or product list.

**This is separate from recommendations.** The Two-Tower + CatBoost system recommends items based on purchase history (collaborative signal). The RAG chatbot answers queries based on product content (semantic signal). They serve different user intents and run on different request paths.

**Two FAISS indices — never conflated:**

| Index | Built By | Contains | Used By |
|-------|----------|----------|---------|
| `s3://bucket/indices/faiss_items.index` | `retrieval/faiss/builder.py` | Item embedding vectors (256-dim, Two-Tower output) | `GET /recommendations` only |
| `s3://bucket/indices/faiss_rag.index` | `generation/rag/ingestion.py` | Article description text chunks (text encoder output) | `POST /chat` only |

**What gets added when implementing RAG (no existing directories change):**

| Addition | Location |
|----------|----------|
| Product knowledge index build | `src/generation/rag/ingestion.py` |
| Semantic product search | `src/generation/rag/retriever.py` |
| LLM answer generation | `src/generation/rag/generator.py` |
| Chat API route | `src/serving/api/routers/chat.py` |
| Chatbot Lambda handler | `src/serving/lambdas/chat_handler.py` |
| RAG index pipeline script | `pipelines/run_rag_pipeline.py` |

---

## Makefile Reference

```makefile
# Core recommendation pipeline (run in order)
data-pipeline:        # Step 1: raw → clean
feature-pipeline:     # Step 2: clean → model-ready features
train:                # Step 3: train two-tower + CatBoost
build-index:          # Step 4: build recommendation FAISS index and upload to S3

# Optional enrichment (run before feature-pipeline to enable tag features)
content-features:     # LLM batch inference → article_tags + user_tag_features
finetune-tags:        # One-off: fine-tune the tag extraction model (manual trigger)

# Standalone chatbot (independent of recommendation pipeline)
rag-pipeline:         # Build RAG product knowledge index (future)

# Development
dev:                  # Start full local stack (API + Redis + LocalStack)
test-unit:            # Run unit tests only
test-integration:     # Spin up LocalStack, run integration tests, tear down

# Infrastructure
infra-up:             # terraform apply (AWS)
infra-down:           # terraform destroy (AWS)
```
