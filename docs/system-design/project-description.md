# Fashion Recommendation System — Project Description

## What This Project Is

A production-grade fashion recommendation system built on the H&M dataset. The primary goal is learning to build scalable ML systems using modern cloud architecture patterns — serverless, event-driven, microservices — while keeping total costs under $40 over the development period.

Every architectural decision mirrors how production recommendation systems are built at scale. The only difference from production is dataset size: 10K users, 5K items, 100K transactions instead of the full H&M data (1.37M users, 105K items, 31.8M transactions). The architecture handles both identically — switching is a configuration change, not a rewrite.

---

## Dataset

**Source:** H&M Personalization Challenge dataset

| Table | Full Scale | Dev Sample | Contents |
|-------|-----------|------------|----------|
| `articles.csv` | 105K records | 5K items | Product catalog, hierarchical classification, text descriptions |
| `customers.csv` | 1.37M records | 10K users | Demographics, age, club membership |
| `transactions_train.csv` | 31.8M records | 100K interactions | Purchase history with dates and prices |

**Schema pattern:** Star schema — transactions as fact table, articles and customers as dimensions.
Full schema: [`system-design/schema-info.md`](schema-info.md)

---

## Core System: Two-Stage Recommendation Pipeline

The system recommends fashion articles to users based on their purchase history. It uses a two-stage funnel — the industry-standard pattern used at Spotify, Pinterest, and Netflix.

### Stage 1 — Retrieval (Two-Tower + FAISS)

A Two-Tower neural network (PyTorch) learns separate 256-dimensional embedding spaces for users and items. At training time, user-item pairs from transactions are pushed together in embedding space; random negatives are pushed apart (contrastive loss).

At inference time, FAISS (Approximate Nearest Neighbor search) finds the top-100 most similar item embeddings for the requesting user. This narrows 5K–105K items down to 100 candidates in under 1ms.

### Stage 2 — Ranking (CatBoost)

The 100 FAISS candidates are re-ranked using a CatBoost gradient boosting model with a richer feature set: user × item cross features, recency, price affinity, category overlap, and optionally LLM-derived style tags. CatBoost scores each candidate and returns the top-K items.

### Recommendation Request Path

```
GET /recommendations/{user_id}
      ↓
API Gateway → Lambda (FastAPI + AWS Lambda Web Adapter)
      ↓
 1. Fetch user features          (Redis hot-path / S3 fallback)
 2. Two-Tower SageMaker Endpoint → user embedding
 3. FAISS Lambda                 → top-100 candidates
 4. CatBoost SageMaker Endpoint  → ranked scores
 5. Return top-K items
```

---

## Optional Enrichment (Future Plan - skip as of now): LLM Tag Extraction (`content_features/`)

A fine-tuned LLM (HuggingFace + LoRA/PEFT) runs as an **offline batch process** to extract structured style tags from article text descriptions — e.g., "casual", "formal", "streetwear", "minimalist".

These tags are aggregated per user from their purchase history and stored as features in S3. The Two-Tower and CatBoost models consume them through the standard feature pipeline — they never call the LLM at inference time.

**This is not required for the core pipeline.** The recommendation system works without it. When enabled, it runs periodically (e.g. weekly) and enriches the user profile with content-based signals that improve cold-start performance and recommendation diversity.

**Flow:**
```
articles_clean.parquet
    → Fine-tuned LLM (batch inference)
    → s3://bucket/enriched/article_tags.parquet
    → s3://bucket/enriched/user_tag_features.parquet
    → feature_pipeline/user_features.py (as one feature among many)
    → Two-Tower + CatBoost training
```

---

## Standalone Feature (Future Plan - skip as of now): RAG Chatbot (`generation/rag/`) 

A chatbot panel alongside the product feed. Users ask natural language questions — "show me something casual for summer" or "what's good for a wedding?" — and the chatbot responds with a grounded answer or product list.

**This is completely separate from the recommendation pipeline.** The Two-Tower + CatBoost system recommends items based on purchase history (collaborative signal). The RAG chatbot answers queries based on product content (semantic signal). They serve different user intents and run on different request paths.

### Chatbot Request Path

```
POST /chat  {"message": "something casual for summer"}
      ↓
API Gateway → Lambda (FastAPI + AWS Lambda Web Adapter)
      ↓
 1. Embed user query (text encoder)
 2. FAISS search over faiss_rag.index → relevant product description chunks
 3. LLM call: chunks + query → natural language answer
 4. Return {"answer": "...", "products": [...]}
```

### Two FAISS Indices — Never Conflated

| Index | Built From | Used By |
|-------|-----------|---------|
| `faiss_items.index` | Two-Tower item embeddings (256-dim) | `GET /recommendations` only |
| `faiss_rag.index` | Article description text chunks (RAG encoder) | `POST /chat` only |

---

## Full System Picture

```
                     OFFLINE PIPELINES
─────────────────────────────────────────────────────────────
 data_pipeline/       Raw CSV → Clean parquet (S3)

 content_features/    Article text → Style tags → User tag features
 [OPTIONAL]           (LLM fine-tuning + batch inference)

 feature_pipeline/    Clean data + optional tags → Model features (S3)

 retrieval/           Two-Tower training → Item embeddings → faiss_items.index
 ranking/             CatBoost training

 generation/rag/      Article descriptions → faiss_rag.index
 [STANDALONE]         (independent of recommendation pipeline)

─────────────────────────────────────────────────────────────
                     ONLINE SERVING
─────────────────────────────────────────────────────────────
 GET /recommendations/{user_id}
   → Two-Tower SageMaker → FAISS items → CatBoost SageMaker → top-K items

 POST /chat
   → FAISS RAG → LLM SageMaker → natural language answer
─────────────────────────────────────────────────────────────
```

---

## Infrastructure

| Concern | Local Development | AWS Production |
|---------|-------------------|----------------|
| Data processing | PySpark `local[*]` | AWS Glue |
| ML training | Docker + SageMaker SDK (`instance_type='local'`) | SageMaker Training Jobs |
| ML inference | Local model servers | SageMaker Endpoints |
| Vector search | Local FAISS `.index` files | Lambda + FAISS (index loaded from S3) |
| API serving | FastAPI + uvicorn | Lambda + AWS Lambda Web Adapter + API Gateway |
| Feature cache | Local Redis | ElastiCache |
| Storage | Local filesystem | S3 (data lake + model artifacts + indices) |
| AWS simulation | LocalStack + Docker Compose | — |
| IaC | — | Terraform (`apply` / `destroy`) |

**Core principle:** The only differences between local and AWS are configuration values — never code logic. All `os.getenv()` calls live in a single `src/config.py`. All `boto3.client()` calls go through `src/shared/aws_clients.py` which handles LocalStack vs real AWS transparently.

---

## Development Strategy

**Local-first, AWS only for final validation.** All development happens at $0 cost using LocalStack + Docker Compose. AWS is used only for final testing and portfolio demonstration, then torn down with `terraform destroy`.

### Pipeline Execution Order

```
Step 1  run_data_pipeline.py         raw → clean
Step 2  run_feature_pipeline.py      clean → model features
Step 3  run_training_pipeline.py     train Two-Tower + CatBoost
Step 4  run_index_pipeline.py        build faiss_items.index

Optional (before Step 2):
        run_content_features.py      article text → enriched tag features

Standalone:
        run_rag_pipeline.py          build faiss_rag.index  (future)
```

### Cost Summary

| Phase | Estimated Cost |
|-------|---------------|
| Local development | $0 |
| AWS testing sessions (total) | $25–40 |
| Small dataset operation | ~$50/month |
| Full dataset operation | ~$98/month |
| **Target total (2–3 months)** | **$25–40** |

---

## What You Learn From This Project

| Area | Specifics |
|------|-----------|
| **ML Engineering** | Two-tower contrastive learning, FAISS ANN search, CatBoost ranking, LLM fine-tuning with LoRA |
| **Feature Engineering** | PySpark feature pipelines, user/item aggregates, LLM-derived features, Redis feature serving |
| **System Design** | Two-stage retrieval-ranking funnel, offline training vs online serving separation, RAG architecture |
| **AWS** | SageMaker Training Jobs + Endpoints, Lambda, API Gateway, S3 data lake, ElastiCache, Glue |
| **MLOps** | SageMaker model registry, A/B testing, canary deployment, drift monitoring, Terraform IaC |
| **Software Engineering** | FastAPI, Docker, LocalStack, environment-driven config, production Python project structure |

---

## Related Documentation

| Document | Contents |
|----------|----------|
| [`system-design/infrastructure-layer.md`](infrastructure-layer.md) | Full architecture overview, all migration patterns, local-to-AWS guide |
| [`system-design/schema-info.md`](schema-info.md) | H&M dataset schema, table relationships, SQL patterns |
| [`system-design/project-structure.md`](project-structure.md) | Directory layout, S3 lineage, design rationale |
