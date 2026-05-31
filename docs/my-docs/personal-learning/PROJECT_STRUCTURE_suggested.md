# Fashion Recommendation Platform - Project Structure Guide

## 1. Purpose

This document explains the recommended production-grade project structure for a scalable AWS-based fashion recommendation platform.

The platform contains four major systems:

1. **Two-stage recommendation pipeline**
   - Stage 1: Two-Tower retrieval model with FAISS candidate generation
   - Stage 2: CatBoost ranking model

2. **Offline feature engineering pipeline**
   - Cleans raw data
   - Handles missing values, outliers, duplicates, and schema issues
   - Builds reusable user, item, and user-item cross features

3. **Optional LLM content enrichment pipeline**
   - Extracts style tags from product descriptions using a fine-tuned LLM
   - Runs offline only
   - Enriches recommendation features

4. **Separate RAG chatbot**
   - Handles natural-language product discovery
   - Uses semantic search over product descriptions
   - Does not share the same request path as the recommendation engine

The goal of this structure is to keep the project modular, maintainable, testable, and production-ready.

---

## 2. High-Level Architecture

```text
GET /recommendations/{user_id}
        |
        v
API Gateway -> Lambda / FastAPI
        |
        v
Recommendation Orchestrator
        |
        |-- Fetch user features from Redis / S3 fallback
        |-- Call Two-Tower SageMaker Endpoint
        |-- Call FAISS candidate retrieval
        |-- Call CatBoost SageMaker Endpoint
        |-- Apply business rules and post-processing
        v
Top-K recommended articles
```

```text
POST /chat
        |
        v
API Gateway -> Lambda / FastAPI
        |
        v
RAG Chat Orchestrator
        |
        |-- Embed query
        |-- Search FAISS RAG index
        |-- Build grounded prompt
        |-- Call LLM
        |-- Return answer and products
        v
Grounded chatbot response
```

The recommendation pipeline and chatbot are intentionally separated because they solve different problems:

| System | Main Signal | User Intent | Runtime Path |
|---|---|---|---|
| Recommendation engine | Purchase behavior | Personalized feed | `/recommendations/{user_id}` |
| RAG chatbot | Product text semantics | User asks a question | `/chat` |

---

## 3. Recommended Repository Structure

```text
fashion-recsys-platform/
|
├── README.md
├── pyproject.toml
├── Makefile
├── docker-compose.yml
├── .env.example
├── .gitignore
|
├── configs/
│   ├── base.yaml
│   ├── dev.yaml
│   ├── staging.yaml
│   ├── prod.yaml
│   │
│   ├── data/
│   │   ├── preprocessing.yaml
│   │   ├── raw_sources.yaml
│   │   ├── s3_paths.yaml
│   │   └── schemas.yaml
│   │
│   ├── features/
│   │   ├── feature_sets.yaml
│   │   ├── user_features.yaml
│   │   ├── item_features.yaml
│   │   └── cross_features.yaml
│   │
│   ├── models/
│   │   ├── two_tower.yaml
│   │   ├── catboost_ranker.yaml
│   │   ├── llm_tag_extractor.yaml
│   │   └── rag.yaml
│   │
│   └── serving/
│       ├── recommendation_api.yaml
│       ├── faiss.yaml
│       ├── redis.yaml
│       └── rag_api.yaml
|
├── src/
│   └── fashion_recsys/
│       |
│       ├── common/
│       ├── contracts/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── serving/
│       ├── generation/
│       ├── pipelines/
│       └── evaluation/
|
├── infra/
├── deployment/
├── tests/
├── notebooks/
├── data_contracts/
├── model_artifacts/
├── monitoring/
├── ci/
└── docs/
```

---

## 4. Source Code Structure

The main application code lives under:

```text
src/fashion_recsys/
```

This keeps the Python package clean and makes it easy to install, test, and deploy.

---

## 5. `common/` - Shared Utilities

```text
src/fashion_recsys/common/
├── logging.py
├── metrics.py
├── exceptions.py
├── constants.py
├── serialization.py
├── s3.py
├── redis.py
├── config.py
└── tracing.py
```

### Purpose

The `common/` module contains shared utilities used across training, serving, feature pipelines, and batch jobs.

### What belongs here

- Logging setup
- CloudWatch metric helpers
- S3 read/write utilities
- Redis client helpers
- Configuration loader
- Custom exceptions
- Serialization helpers
- Request tracing helpers

### What should not belong here

Do not put model logic, feature logic, or business rules here. This folder should only contain generic reusable utilities.

---

## 6. `contracts/` - Shared Interfaces and Schemas

```text
src/fashion_recsys/contracts/
├── recommendation.py
├── ranking.py
├── features.py
├── rag.py
└── events.py
```

### Purpose

The `contracts/` module defines stable interfaces between components.

For example:

- API request schemas
- API response schemas
- SageMaker model input/output schemas
- Feature vector schemas
- Event logging schemas

### Example contracts

```text
RecommendationRequest
RecommendationResponse
UserFeatureVector
ItemFeatureVector
CandidateItem
CatBoostPredictionRequest
TwoTowerEmbeddingResponse
RagChatRequest
RagChatResponse
```

### Why this matters

Production ML systems often fail because training, serving, and feature pipelines silently drift from each other. Contracts reduce this risk by making interfaces explicit and testable.

---

## 7. `data/` - Raw Data Ingestion, Cleaning, Validation, and Datasets

```text
src/fashion_recsys/data/
├── ingestion/
│   ├── transactions_ingest.py
│   ├── articles_ingest.py
│   └── users_ingest.py
│
├── transforms/
│   ├── clean_transactions.py
│   ├── clean_articles.py
│   ├── clean_users.py
│   ├── handle_nulls.py
│   ├── handle_outliers.py
│   ├── handle_duplicates.py
│   ├── normalize_types.py
│   └── standardize_categories.py
│
├── validation/
│   ├── schemas.py
│   ├── validators.py
│   ├── data_quality_checks.py
│   └── leakage_checks.py
│
└── datasets/
    ├── two_tower_dataset.py
    ├── catboost_dataset.py
    └── rag_dataset.py
```

---

### 7.1 `data/ingestion/`

This folder is responsible for loading source data.

Examples:

- Load transaction data
- Load article metadata
- Load customer data
- Read from S3, database exports, or batch files

This layer should not perform heavy feature engineering. It should only read and standardize source inputs.

---

### 7.2 `data/transforms/`

This is where raw data cleaning belongs.

Use this folder for:

| Task | File |
|---|---|
| Fill missing values | `handle_nulls.py` |
| Remove duplicate records | `handle_duplicates.py` |
| Cap or remove outliers | `handle_outliers.py` |
| Normalize date and ID types | `normalize_types.py` |
| Clean transaction data | `clean_transactions.py` |
| Clean article data | `clean_articles.py` |
| Clean user data | `clean_users.py` |
| Normalize category names | `standardize_categories.py` |

### Important rule

Cleaning should answer this question:

```text
Is the raw data valid, consistent, and safe to use?
```

Cleaning should not answer this question:

```text
Which predictive signals should the model use?
```

That belongs in `features/`.

---

### 7.3 `data/validation/`

This folder validates data quality before data is used for features or training.

Checks should include:

- Required columns exist
- Primary IDs are not null
- Prices are non-negative
- Dates are parseable
- Article IDs in transactions exist in articles table
- User IDs in transactions exist in users table
- Duplicate rows are within expected limits
- Feature values are within expected ranges
- No future data leakage exists in training sets

Validation should run after ingestion and after important transformation stages.

---

### 7.4 `data/datasets/`

This folder converts cleaned and feature-enriched data into model-specific dataset objects.

Examples:

- Two-Tower PyTorch dataset
- CatBoost ranking dataset
- RAG indexing dataset

This folder prepares data for model consumption, but it should not contain general cleaning logic.

---

## 8. `features/` - Feature Engineering, Selection, and Feature Store Logic

```text
src/fashion_recsys/features/
├── feature_registry.py
│
├── user_features/
│   ├── purchase_history.py
│   ├── recency.py
│   ├── price_affinity.py
│   ├── category_affinity.py
│   ├── user_style_tags.py
│   └── build_user_features.py
│
├── item_features/
│   ├── article_metadata.py
│   ├── item_price.py
│   ├── item_category.py
│   ├── item_style_tags.py
│   └── build_item_features.py
│
├── cross_features/
│   ├── user_item_category_overlap.py
│   ├── price_match.py
│   ├── style_tag_overlap.py
│   └── build_cross_features.py
│
├── selection/
│   ├── catboost_feature_selection.py
│   ├── two_tower_feature_selection.py
│   ├── feature_importance.py
│   ├── shap_analysis.py
│   └── leakage_checks.py
│
├── pipelines/
│   ├── build_clean_dataset.py
│   ├── build_user_features.py
│   ├── build_item_features.py
│   ├── build_cross_features.py
│   ├── build_training_features.py
│   └── build_serving_features.py
│
└── store/
    ├── offline_writer.py
    ├── online_writer.py
    ├── redis_hot_path.py
    └── s3_feature_store.py
```

---

### 8.1 `features/user_features/`

User-level features are built here.

Examples:

- Number of purchases
- Average purchase price
- Recent purchase count
- Favorite product category
- Category distribution
- Style tag preferences
- Days since last purchase

These features can be consumed by both the Two-Tower model and CatBoost ranker.

---

### 8.2 `features/item_features/`

Item-level features are built here.

Examples:

- Product category
- Department
- Color
- Price bucket
- Product age
- Style tags
- Text-derived metadata

These features are used during ranking, retrieval training, RAG indexing, and content enrichment.

---

### 8.3 `features/cross_features/`

Cross features describe the relationship between a user and an item.

Examples:

- User preferred category equals item category
- Price difference between user average purchase price and item price
- Style tag overlap between user profile and item tags
- Color affinity match
- Department affinity match

These are especially important for CatBoost ranking.

---

### 8.4 `features/selection/`

Feature selection logic belongs here.

Examples:

- CatBoost feature importance
- SHAP analysis
- Permutation importance
- Correlation checks
- Leakage checks
- Feature ablation experiments

Feature selection should not be buried inside `train.py`. Keeping it separate makes experiments reproducible and easier to review.

---

### 8.5 `features/pipelines/`

This folder composes feature generation jobs.

Example flow:

```text
raw data
  -> data/transforms/
  -> data/validation/
  -> features/user_features/
  -> features/item_features/
  -> features/cross_features/
  -> features/store/
```

Use this folder for scripts that create complete feature tables for training or serving.

---

### 8.6 `features/store/`

This folder writes and reads features from storage.

Recommended storage pattern:

```text
Offline features -> S3 or SageMaker Feature Store offline store
Online features  -> Redis or SageMaker Feature Store online store
```

The offline store is used for historical data, model training, batch inference, and analysis. Online storage is used when serving needs low-latency access to latest features.

---

## 9. `models/` - Retrieval, Ranking, Content Features, and Model Registry

```text
src/fashion_recsys/models/
├── retrieval/
│   ├── two_tower/
│   │   ├── model.py
│   │   ├── loss.py
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export.py
│   │   └── inference.py
│   │
│   └── faiss_index/
│       ├── build_index.py
│       ├── validate_index.py
│       ├── search.py
│       ├── index_metadata.py
│       └── publish_index.py
│
├── ranking/
│   └── catboost/
│       ├── train.py
│       ├── evaluate.py
│       ├── features.py
│       ├── preprocess.py
│       ├── feature_config.py
│       ├── export.py
│       └── inference.py
│
├── content_features/
│   └── llm_tag_extractor/
│       ├── prompts/
│       ├── fine_tune_lora.py
│       ├── batch_inference.py
│       ├── postprocess_tags.py
│       ├── aggregate_user_tags.py
│       └── evaluate_tags.py
│
└── registry/
    ├── model_card.py
    ├── register_model.py
    ├── promote_model.py
    └── versioning.py
```

---

### 9.1 `models/retrieval/two_tower/`

This folder contains the Two-Tower model.

Responsibilities:

- Build user tower
- Build item tower
- Train contrastive objective
- Generate user embeddings
- Generate item embeddings
- Export model for SageMaker inference

Model-specific preprocessing belongs here.

Examples:

- Encode user IDs
- Encode item IDs
- Create positive user-item pairs
- Perform negative sampling
- Build time-based train/validation split
- Prepare PyTorch tensors

Important rule:

```text
General data cleaning does not belong here.
Two-Tower-specific formatting does belong here.
```

---

### 9.2 `models/retrieval/faiss_index/`

This folder owns FAISS index lifecycle.

Responsibilities:

- Build ANN index from item embeddings
- Validate recall and index health
- Store item ID mapping
- Publish versioned index artifacts
- Provide search logic for serving

A FAISS index should always be versioned with:

```text
faiss.index
item_id_mapping.parquet
index_metadata.json
validation_report.json
```

Never deploy `faiss.index` without its item mapping.

---

### 9.3 `models/ranking/catboost/`

This folder contains the CatBoost ranker.

Responsibilities:

- Build ranking training dataset
- Merge user, item, and cross features
- Define categorical and numerical feature lists
- Train ranker or classifier
- Evaluate ranking quality
- Export model for SageMaker inference

CatBoost-specific preprocessing belongs here.

Examples:

- Build candidate training frame
- Add labels
- Add group/query IDs
- Define categorical columns
- Define numerical columns
- Prepare CatBoost Pool object

Important rule:

```text
CatBoost feature list should live in config or feature_config.py, not inside train.py.
```

---

### 9.4 `models/content_features/llm_tag_extractor/`

This folder contains the optional offline LLM tag extraction pipeline.

Responsibilities:

- Fine-tune LLM using LoRA/PEFT if needed
- Run batch inference over article descriptions
- Extract structured style tags
- Validate and postprocess tags
- Aggregate article tags into user preference tags

This system should not run in the online recommendation path.

Correct flow:

```text
articles_clean.parquet
  -> LLM batch inference
  -> article_tags.parquet
  -> user_tag_features.parquet
  -> normal feature pipeline
  -> Two-Tower and CatBoost training
```

---

### 9.5 `models/registry/`

This folder handles model versioning and promotion.

Responsibilities:

- Register trained model artifacts
- Store metadata
- Promote model from staging to production
- Generate model cards
- Track training data snapshots

Each model version should store:

```text
model name
model version
git SHA
training data snapshot
feature pipeline version
metrics
approval status
owner
created timestamp
```

---

## 10. `serving/` - Online Recommendation Serving

```text
src/fashion_recsys/serving/
├── recommendation_api/
│   ├── app.py
│   ├── routes.py
│   ├── dependencies.py
│   ├── request_models.py
│   ├── response_models.py
│   ├── health.py
│   └── Dockerfile
│
├── recommendation_service/
│   ├── orchestrator.py
│   ├── feature_fetcher.py
│   ├── retrieval_client.py
│   ├── faiss_client.py
│   ├── ranking_client.py
│   ├── business_rules.py
│   ├── fallback.py
│   └── postprocessing.py
│
├── faiss_lambda/
│   ├── handler.py
│   ├── index_loader.py
│   ├── search_service.py
│   ├── warmup.py
│   └── Dockerfile
│
├── sagemaker_inference/
│   ├── two_tower/
│   │   ├── inference.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── catboost/
│       ├── inference.py
│       ├── requirements.txt
│       └── Dockerfile
│
└── middleware/
    ├── auth.py
    ├── rate_limit.py
    ├── request_id.py
    └── error_handler.py
```

---

### 10.1 `serving/recommendation_api/`

This contains the FastAPI application used by API Gateway and Lambda Web Adapter.

Responsibilities:

- Define HTTP routes
- Validate API inputs
- Format API responses
- Expose health checks
- Call the recommendation service orchestrator

This folder should not contain ML training logic.

---

### 10.2 `serving/recommendation_service/`

This is the online recommendation orchestration layer.

Responsibilities:

- Fetch user features
- Call Two-Tower endpoint
- Call FAISS retrieval service
- Call CatBoost ranking endpoint
- Apply business rules
- Apply fallback logic
- Return final top-K products

The orchestrator should coordinate services. It should not contain model internals.

---

### 10.3 `serving/faiss_lambda/`

This folder contains the FAISS online search service.

Responsibilities:

- Load FAISS index
- Load item ID mapping
- Keep index warm
- Perform vector search
- Return top-N article IDs

If the index becomes too large for Lambda memory or cold-start constraints, migrate this service to ECS Fargate, EKS, or SageMaker endpoint.

---

### 10.4 `serving/sagemaker_inference/`

This folder contains SageMaker inference container code.

Separate inference containers are recommended for:

- Two-Tower user embedding endpoint
- CatBoost ranking endpoint

Each should have its own `inference.py`, dependencies, and Dockerfile.

---

## 11. `generation/rag/` - Separate RAG Chatbot

```text
src/fashion_recsys/generation/rag/
├── api/
│   ├── app.py
│   ├── routes.py
│   ├── request_models.py
│   ├── response_models.py
│   └── Dockerfile
│
├── indexing/
│   ├── chunk_articles.py
│   ├── embed_chunks.py
│   ├── build_faiss_rag_index.py
│   ├── validate_rag_index.py
│   └── publish_rag_index.py
│
├── retrieval/
│   ├── query_encoder.py
│   ├── faiss_search.py
│   └── rerank_chunks.py
│
├── generation/
│   ├── prompt_builder.py
│   ├── llm_client.py
│   ├── guardrails.py
│   └── response_parser.py
│
└── service/
    ├── chat_orchestrator.py
    ├── product_grounding.py
    └── fallback.py
```

### Purpose

This module owns the chatbot experience.

It is separate from recommendation serving because it uses different signals, different models, and different latency/cost tradeoffs.

### Main flow

```text
user query
  -> query embedding
  -> FAISS RAG search
  -> retrieve relevant product chunks
  -> build grounded prompt
  -> call LLM
  -> parse response
  -> return answer and products
```

### Best practice

Do not call the recommendation Two-Tower or CatBoost stack from the chatbot by default. The chatbot should be content-grounded and query-driven.

---

## 12. `pipelines/` - Batch and ML Workflow Orchestration

```text
src/fashion_recsys/pipelines/
├── sagemaker/
│   ├── two_tower_pipeline.py
│   ├── catboost_pipeline.py
│   ├── llm_tag_pipeline.py
│   ├── rag_index_pipeline.py
│   └── shared_steps.py
│
├── airflow/
│   ├── dags/
│   │   ├── daily_feature_refresh.py
│   │   ├── weekly_llm_tag_extraction.py
│   │   ├── two_tower_retrain.py
│   │   ├── catboost_retrain.py
│   │   └── rag_index_refresh.py
│   └── plugins/
│
└── step_functions/
    ├── recommendation_training.asl.json
    ├── feature_refresh.asl.json
    └── rag_refresh.asl.json
```

### Purpose

This folder orchestrates repeatable workflows.

Examples:

- Daily feature refresh
- Weekly LLM tag extraction
- Two-Tower retraining
- CatBoost retraining
- FAISS index rebuild
- RAG index rebuild
- Model registration and promotion

### Recommendation

Use one primary orchestrator in production. Good options are:

- SageMaker Pipelines for ML workflows
- Step Functions for AWS service orchestration
- MWAA / Airflow for complex scheduled data workflows

Avoid maintaining multiple orchestration systems unless there is a strong reason.

---

## 13. `evaluation/` - Offline and Online Evaluation

```text
src/fashion_recsys/evaluation/
├── offline/
│   ├── retrieval_metrics.py
│   ├── ranking_metrics.py
│   ├── diversity_metrics.py
│   └── calibration.py
│
├── online/
│   ├── ab_testing.py
│   ├── exposure_logging.py
│   ├── clickstream_metrics.py
│   └── attribution.py
│
└── rag/
    ├── faithfulness.py
    ├── retrieval_recall.py
    ├── answer_quality.py
    └── safety_eval.py
```

### Offline recommendation metrics

For retrieval:

- Recall@K
- Precision@K
- Coverage
- Embedding similarity distribution
- Candidate diversity

For ranking:

- NDCG@K
- MAP@K
- MRR
- Hit rate
- Calibration
- Diversity

### Online metrics

- CTR
- Add-to-cart rate
- Purchase conversion
- Revenue per session
- Latency
- Error rate
- Cold-start performance
- Repeat recommendation rate

### RAG metrics

- Retrieval recall
- Groundedness
- Faithfulness
- Hallucination rate
- Product citation correctness
- Safety and policy compliance

---

## 14. `configs/` - Environment and Model Configuration

```text
configs/
├── base.yaml
├── dev.yaml
├── staging.yaml
├── prod.yaml
├── data/
├── features/
├── models/
└── serving/
```

### Purpose

Configuration should be externalized and environment-specific.

Do not hardcode:

- S3 bucket names
- Redis endpoints
- SageMaker endpoint names
- Model versions
- Index versions
- Top-K values
- Timeout values
- Batch sizes
- Feature lists

### Example preprocessing config

```yaml
transactions:
  required_columns:
    - customer_id
    - article_id
    - price
    - t_dat

  null_handling:
    price: median
    customer_id: drop_row
    article_id: drop_row

  outlier_handling:
    price:
      method: winsorize
      lower_quantile: 0.01
      upper_quantile: 0.99

articles:
  null_handling:
    detail_desc: empty_string
    product_type_name: unknown
    colour_group_name: unknown
```

### Example CatBoost config

```yaml
features:
  categorical:
    - product_type_name
    - department_name
    - colour_group_name
    - preferred_category

  numerical:
    - user_avg_price
    - item_price
    - price_diff
    - recency_score
    - category_overlap_score
    - style_tag_overlap

target:
  name: label
  type: binary

ranking:
  group_id: customer_id
```

---

## 15. `infra/` - Infrastructure as Code

```text
infra/
├── cdk/
│   ├── app.py
│   ├── requirements.txt
│   ├── cdk.json
│   ├── stacks/
│   └── constructs/
│
├── terraform/
│   ├── modules/
│   └── envs/
│       ├── dev/
│       ├── staging/
│       └── prod/
│
└── permissions/
    ├── recommendation_api_policy.json
    ├── sagemaker_execution_role.json
    ├── batch_jobs_policy.json
    └── least_privilege_notes.md
```

### Purpose

All AWS infrastructure should be created through code.

This includes:

- VPC and networking
- S3 buckets
- IAM roles
- ECR repositories
- Lambda functions
- API Gateway routes
- Redis cluster
- SageMaker endpoints
- SageMaker Feature Store, if used
- CloudWatch dashboards and alarms
- CI/CD resources

### Recommendation

Choose either CDK or Terraform as the primary IaC tool. Do not maintain both long-term unless your organization requires it.

---

## 16. `deployment/` - Build and Release Assets

```text
deployment/
├── docker/
│   ├── recommendation_api.Dockerfile
│   ├── faiss_lambda.Dockerfile
│   ├── rag_api.Dockerfile
│   ├── two_tower_inference.Dockerfile
│   └── catboost_inference.Dockerfile
│
├── scripts/
│   ├── build_images.sh
│   ├── push_images.sh
│   ├── deploy_dev.sh
│   ├── deploy_staging.sh
│   ├── deploy_prod.sh
│   └── rollback.sh
│
└── manifests/
    ├── lambda_env.json
    ├── sagemaker_endpoints.json
    └── api_gateway_routes.json
```

### Purpose

This folder contains deployment-specific files and scripts.

The application code lives in `src/`. Deployment logic lives here.

---

## 17. `tests/` - Unit, Integration, Load, and Contract Tests

```text
tests/
├── unit/
├── integration/
├── load/
└── contract/
```

### Unit tests

Test small functions in isolation:

- Feature builders
- Data validators
- Preprocessing functions
- Business rules
- Response parsers

### Integration tests

Test real component flows:

- API to recommendation orchestrator
- Redis fallback behavior
- SageMaker endpoint client
- FAISS search service
- RAG chat flow

### Contract tests

Contract tests are critical for this system.

They should verify:

- Recommendation API schema
- Two-Tower endpoint input/output
- CatBoost endpoint input/output
- Feature schema compatibility
- RAG request/response schema

### Load tests

Load tests should validate:

- P95 and P99 latency
- Lambda cold-start impact
- FAISS query latency
- SageMaker endpoint throughput
- Redis hot-path performance

---

## 18. `data_contracts/` - Explicit Data Schemas

```text
data_contracts/
├── transactions.schema.json
├── users.schema.json
├── articles.schema.json
├── user_features.schema.json
├── item_features.schema.json
├── catboost_training.schema.json
└── rag_chunks.schema.json
```

### Purpose

Data contracts define expected columns, types, nullability, and constraints.

They protect the system from upstream data changes.

Example checks:

- `customer_id` must be string and non-null
- `article_id` must be string and non-null
- `price` must be numeric and non-negative
- `t_dat` must be a valid date
- `detail_desc` may be nullable before cleaning but not after cleaning

---

## 19. `model_artifacts/` - Local Placeholder Only

```text
model_artifacts/
├── README.md
├── two_tower/
├── catboost/
├── faiss/
└── rag/
```

### Important rule

Do not commit large model files, FAISS indexes, or embeddings to Git.

This folder should only contain placeholders and documentation.

Actual artifacts should live in S3.

Recommended S3 layout:

```text
s3://fashion-recsys-prod-artifacts/
├── models/
│   ├── two_tower/version=2026-05-26-001/
│   └── catboost/version=2026-05-26-001/
│
├── indexes/
│   ├── faiss_retrieval/version=2026-05-26-001/
│   └── faiss_rag/version=2026-05-26-001/
│
└── reports/
    ├── evaluation/
    └── validation/
```

---

## 20. `monitoring/` - Dashboards, Alarms, and Logging Docs

```text
monitoring/
├── dashboards/
├── alarms/
└── logs/
```

### What to monitor

Recommendation API:

- Request count
- Error rate
- P50/P95/P99 latency
- Timeout rate
- Empty recommendation rate
- Fallback rate

Two-Tower endpoint:

- Invocation count
- Latency
- Error rate
- CPU/GPU utilization
- Model version

FAISS service:

- Search latency
- Index version
- Index load time
- Candidate count distribution

CatBoost endpoint:

- Ranking latency
- Error rate
- Score distribution drift

Data and features:

- Missing feature rate
- Null percentage
- Feature freshness
- Schema drift
- Training-serving skew

RAG chatbot:

- Retrieval latency
- LLM latency
- Groundedness failures
- Empty answer rate
- Unsafe response rate

---

## 21. `ci/` - CI/CD Quality Gates

```text
ci/
├── github-actions/
│   ├── lint.yml
│   ├── test.yml
│   ├── build-images.yml
│   ├── deploy-dev.yml
│   ├── deploy-staging.yml
│   └── deploy-prod.yml
│
└── quality/
    ├── ruff.toml
    ├── mypy.ini
    ├── pytest.ini
    └── bandit.yaml
```

### CI checks to enforce

- Linting
- Type checking
- Unit tests
- Contract tests
- Security scanning
- Docker image build
- Infrastructure validation
- No large artifacts committed
- No secrets committed

---

## 22. `docs/` - Human Documentation and ADRs

```text
docs/
├── architecture.md
├── recommendation_flow.md
├── rag_flow.md
├── feature_definitions.md
├── model_training.md
├── deployment.md
├── runbooks.md
├── security.md
├── data_governance.md
└── adr/
    ├── 0001-use-two-stage-retrieval-ranking.md
    ├── 0002-use-faiss-for-ann.md
    ├── 0003-use-catboost-for-ranking.md
    ├── 0004-separate-rag-from-recommendations.md
    └── 0005-use-lambda-web-adapter.md
```

### Purpose

Use this folder for documentation that engineers, ML scientists, and operations teams need.

### ADRs

ADR means Architecture Decision Record.

Use ADRs to document important choices, such as:

- Why Two-Tower + CatBoost was selected
- Why FAISS was selected
- Why RAG is separate from recommendations
- Why Lambda Web Adapter is used
- Why Redis is used for hot-path features
- Why LLM tag extraction is offline only

---

## 23. Recommended S3 Data Lake Layout

```text
s3://fashion-recsys-prod-data/
├── raw/
│   ├── transactions/
│   ├── articles/
│   └── users/
│
├── clean/
│   ├── transactions_clean/
│   ├── articles_clean/
│   └── users_clean/
│
├── validated/
│   ├── transactions/
│   ├── articles/
│   └── users/
│
├── enriched/
│   ├── article_tags/
│   └── user_tag_features/
│
├── features/
│   ├── user_features/
│   ├── item_features/
│   └── cross_features/
│
├── training/
│   ├── two_tower/
│   └── catboost/
│
├── inference/
│   ├── user_features_latest/
│   ├── item_features_latest/
│   └── popular_items_latest/
│
└── rag/
    ├── chunks/
    ├── embeddings/
    └── index_inputs/
```

### Recommended data flow

```text
raw
  -> clean
  -> validated
  -> enriched, optional
  -> features
  -> training
  -> model artifacts
  -> serving
```

Do not overwrite historical training data. Use date partitions or versioned paths.

---

## 24. Where Preprocessing Should Live

| Work | Correct Location |
|---|---|
| Fill missing article description | `data/transforms/handle_nulls.py` |
| Fill missing category | `data/transforms/handle_nulls.py` |
| Remove duplicate transactions | `data/transforms/handle_duplicates.py` |
| Cap price outliers | `data/transforms/handle_outliers.py` |
| Normalize date columns | `data/transforms/normalize_types.py` |
| Validate required columns | `data/validation/schemas.py` |
| Validate no null user/item IDs | `data/validation/data_quality_checks.py` |
| Build user purchase history | `features/user_features/purchase_history.py` |
| Build price affinity | `features/user_features/price_affinity.py` |
| Build category overlap | `features/cross_features/user_item_category_overlap.py` |
| Select CatBoost features | `features/selection/catboost_feature_selection.py` |
| Select Two-Tower features | `features/selection/two_tower_feature_selection.py` |
| Prepare Two-Tower pairs | `models/retrieval/two_tower/preprocess.py` |
| Negative sampling | `models/retrieval/two_tower/dataset.py` |
| Prepare CatBoost ranking frame | `models/ranking/catboost/preprocess.py` |
| Define CatBoost feature list | `models/ranking/catboost/feature_config.py` |
| Write features to S3 | `features/store/offline_writer.py` |
| Write hot features to Redis | `features/store/online_writer.py` |
| Fetch features at inference | `serving/recommendation_service/feature_fetcher.py` |
```

---

## 25. Best-Practice Rules Followed by This Structure

### 25.1 Separate cleaning from feature engineering

Cleaning belongs in:

```text
data/transforms/
```

Feature engineering belongs in:

```text
features/
```

Model-specific formatting belongs in:

```text
models/<model_family>/<model_name>/preprocess.py
```

This prevents preprocessing logic from becoming scattered across notebooks and training scripts.

---

### 25.2 Keep training and serving consistent

Training and serving must use the same feature definitions.

Bad pattern:

```text
Training features are built one way.
Serving features are manually recreated another way.
```

Good pattern:

```text
Reusable feature builders live in features/.
Training and serving consume versioned feature outputs.
```

---

### 25.3 Use time-based splits for recommendation models

For recommendation systems, avoid random train-test splits.

Use time-based splits:

```text
Older interactions -> training
Newer interactions -> validation/test
```

This better simulates real production behavior.

---

### 25.4 Keep online path lightweight

The online recommendation API should not perform heavy feature engineering, LLM calls, or batch transformations.

At inference time, it should mostly:

- Fetch precomputed features
- Call model endpoints
- Retrieve candidates
- Rank candidates
- Apply business rules
- Return response

---

### 25.5 Keep LLM tag extraction offline

The LLM tag extractor should run offline because:

- It is slower
- It is more expensive
- It introduces more failure modes
- It is not required for every request

The online recommendation path should only consume already-computed style tag features.

---

### 25.6 Treat FAISS indexes as production artifacts

A FAISS index is not just a temporary file. It is a deployable artifact.

Each version should include:

```text
faiss.index
item_id_mapping.parquet
index_metadata.json
validation_report.json
```

---

### 25.7 Add fallback recommendations

Production recommendation APIs should degrade gracefully.

Recommended fallback order:

```text
Personalized recommendations
  -> User segment recommendations
  -> Category popular items
  -> Global trending items
```

Fallback logic belongs in:

```text
serving/recommendation_service/fallback.py
```

---

### 25.8 Keep business rules outside the model

Rules such as these should not be hidden inside model training:

- Remove already purchased items
- Remove out-of-stock items
- Apply diversity rules
- Filter blocked products
- Enforce category balance

They belong in:

```text
serving/recommendation_service/business_rules.py
serving/recommendation_service/postprocessing.py
```

---

### 25.9 Use explicit contracts

The system has many boundaries:

```text
API -> Orchestrator
Orchestrator -> Redis
Orchestrator -> SageMaker Two-Tower
Orchestrator -> FAISS
Orchestrator -> SageMaker CatBoost
Feature pipeline -> training
Training -> serving
RAG retrieval -> LLM generation
```

Each boundary should have a tested schema.

---

### 25.10 Version everything important

Version these assets:

- Raw data snapshot
- Clean data snapshot
- Feature tables
- Feature definitions
- Model artifacts
- FAISS indexes
- RAG indexes
- Docker images
- Infrastructure definitions
- Config files

A production model should always be traceable back to the exact data and code that created it.

---

## 26. AWS Service Mapping

| Component | Recommended AWS Service |
|---|---|
| Public API | API Gateway |
| FastAPI runtime | Lambda container + AWS Lambda Web Adapter |
| Online orchestration | Lambda |
| Hot user features | ElastiCache Redis |
| Offline features | S3 or SageMaker Feature Store offline store |
| Feature management | SageMaker Feature Store, optional |
| Two-Tower inference | SageMaker real-time endpoint |
| CatBoost inference | SageMaker real-time endpoint |
| FAISS search | Lambda, ECS Fargate, EKS, or SageMaker endpoint |
| Batch feature jobs | SageMaker Processing, Glue, or EMR Serverless |
| Training jobs | SageMaker Training Jobs |
| ML workflow orchestration | SageMaker Pipelines, Step Functions, or MWAA |
| Model registry | SageMaker Model Registry |
| Container registry | ECR |
| Artifact store | S3 |
| Monitoring | CloudWatch, X-Ray, OpenTelemetry |
| CI/CD | GitHub Actions, CodePipeline, or CodeBuild |
```

---

## 27. Recommended Development Workflow

### Step 1: Ingest raw data

```text
data/ingestion/
```

Output:

```text
s3://bucket/raw/
```

### Step 2: Clean and validate data

```text
data/transforms/
data/validation/
```

Output:

```text
s3://bucket/clean/
s3://bucket/validated/
```

### Step 3: Build features

```text
features/user_features/
features/item_features/
features/cross_features/
features/pipelines/
```

Output:

```text
s3://bucket/features/
```

### Step 4: Train retrieval model

```text
models/retrieval/two_tower/
```

Output:

```text
s3://bucket/artifacts/models/two_tower/
```

### Step 5: Build FAISS retrieval index

```text
models/retrieval/faiss_index/
```

Output:

```text
s3://bucket/artifacts/indexes/faiss_retrieval/
```

### Step 6: Train ranking model

```text
models/ranking/catboost/
```

Output:

```text
s3://bucket/artifacts/models/catboost/
```

### Step 7: Register and promote models

```text
models/registry/
```

Output:

```text
SageMaker Model Registry
```

### Step 8: Deploy serving stack

```text
serving/
infra/
deployment/
```

Output:

```text
API Gateway
Lambda
SageMaker endpoints
Redis
CloudWatch dashboards
```

---

## 28. Anti-Patterns to Avoid

Avoid these patterns:

1. Putting all preprocessing inside `train.py`
2. Cleaning data inside notebooks only
3. Reimplementing feature logic separately for training and serving
4. Calling LLMs in the online recommendation path
5. Deploying FAISS index without item ID mapping
6. Using random train-test split for time-based recommendations
7. Hardcoding feature lists inside model training scripts
8. Mixing RAG chatbot code with recommendation orchestration
9. Manually creating production AWS resources from the console
10. Committing model artifacts or large indexes into Git
11. Skipping contract tests between services
12. Not tracking which data snapshot trained each model
13. Not having fallback recommendations
14. Burying business rules inside ML model logic

---

## 29. Practical Naming Conventions

### AWS resources

```text
fashion-recsys-dev-recommendation-api
fashion-recsys-staging-recommendation-api
fashion-recsys-prod-recommendation-api

fashion-recsys-prod-two-tower-endpoint
fashion-recsys-prod-catboost-endpoint

fashion-recsys-prod-faiss-retrieval-v2026-05-26
fashion-recsys-prod-faiss-rag-v2026-05-26
```

### S3 artifacts

```text
s3://fashion-recsys-prod-artifacts/models/two_tower/version=2026-05-26-001/model.tar.gz
s3://fashion-recsys-prod-artifacts/models/catboost/version=2026-05-26-001/model.tar.gz
s3://fashion-recsys-prod-artifacts/indexes/faiss_retrieval/version=2026-05-26-001/faiss.index
s3://fashion-recsys-prod-artifacts/indexes/faiss_rag/version=2026-05-26-001/faiss_rag.index
```

### Docker images

```text
fashion-recsys/recommendation-api:git-sha
fashion-recsys/faiss-lambda:git-sha
fashion-recsys/two-tower-inference:git-sha
fashion-recsys/catboost-inference:git-sha
fashion-recsys/rag-api:git-sha
```

---

## 30. Minimum Production Readiness Checklist

Before production launch, verify:

- [ ] Data cleaning is separate from feature engineering
- [ ] Feature definitions are versioned
- [ ] Training and serving use the same feature contracts
- [ ] Required columns are validated
- [ ] Null handling is config-driven
- [ ] Outlier handling is documented
- [ ] Time-based train/test split is used
- [ ] Two-Tower model is evaluated with Recall@K
- [ ] CatBoost model is evaluated with NDCG@K or ranking metrics
- [ ] FAISS index includes item ID mapping
- [ ] FAISS index is versioned
- [ ] Model artifacts are stored in S3
- [ ] Models are registered before promotion
- [ ] API has fallback logic
- [ ] API has timeout handling
- [ ] API has rate limiting
- [ ] CloudWatch dashboards exist
- [ ] Latency and error alarms exist
- [ ] Contract tests exist
- [ ] Load tests exist
- [ ] Secrets are not committed
- [ ] IAM permissions follow least privilege
- [ ] Infrastructure is deployed through IaC
- [ ] RAG chatbot is separate from recommendation serving
- [ ] LLM tag extraction is offline only

---

## 31. Summary

The recommended structure follows this mental model:

```text
Raw data problem?
  -> data/transforms/

Data quality problem?
  -> data/validation/

Feature creation?
  -> features/

Feature selection?
  -> features/selection/

Model-specific formatting?
  -> models/<model>/preprocess.py

Online API orchestration?
  -> serving/recommendation_service/

Chatbot logic?
  -> generation/rag/

Infrastructure?
  -> infra/

Deployment assets?
  -> deployment/

Monitoring and alarms?
  -> monitoring/

Architecture explanation?
  -> docs/
```

This structure keeps the system modular without over-splitting it into too many repositories too early.

Start with this modular monorepo. Later, if the engineering team or traffic grows significantly, split it into service-specific repositories such as:

```text
recommendation-serving
recommendation-training
feature-platform
rag-chatbot
infra-platform
```

Until then, the modular monorepo is the best balance of clarity, speed, and production discipline.

---

## 32. References

- AWS Well-Architected Framework: https://docs.aws.amazon.com/wellarchitected/latest/operational-excellence-pillar/welcome.html
- Amazon SageMaker Feature Store: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
- SageMaker Feature Store Offline Store: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-storage-configurations-offline-store.html
- Amazon SageMaker Model Registry: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html
- AWS Lambda Web Adapter: https://aws.github.io/aws-lambda-web-adapter/
