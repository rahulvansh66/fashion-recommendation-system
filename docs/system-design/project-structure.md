# Fashion Recommendation System — Final Project Structure

## 1. Purpose

This document defines the authoritative repository structure for the fashion recommendation system. It supersedes both `project-structure.md` (previous structure) and `PROJECT_STRUCTURE_suggested.md` (senior's suggestion), merging the best of both with explicit decisions recorded for each divergence.

### Key Decisions Summary

| Topic | Decision |
|---|---|
| Python package | `src/fashion_recommendation_system/` — pip-installable, matches repo name |
| Configuration | Hybrid: `configs/` YAML for model/feature params + `config.py` for infra env vars |
| Infrastructure as Code | `infra/` folder, Terraform only |
| Deployment assets | Centralized `deployment/` (all Dockerfiles, scripts, manifests) |
| CI/CD | `ci/` folder with `github-actions/` and `quality/` subdirectories |
| Pipeline scripts | Root `pipelines/` for runner scripts + `src/.../pipelines/` for orchestration logic |
| Documentation | Existing nested `docs/` retained + `docs/adr/` added for Architecture Decision Records |
| New modules | Both `contracts/` and `evaluation/` included in the package |

---

## 2. High-Level Architecture

```text
GET /recommendations/{user_id}
        |
        v
API Gateway -> Lambda (FastAPI + Lambda Web Adapter)
        |
        v
serving/api — Recommendation Orchestrator
        |
        |-- Fetch user features  (Redis hot path / S3 fallback)
        |-- Call Two-Tower SageMaker Endpoint  (user embedding)
        |-- Call FAISS Lambda  (candidate retrieval)
        |-- Call XGBoost SageMaker Endpoint  (ranking)
        |-- Apply business rules + postprocessing
        v
Top-K recommended articles

POST /chat
        |
        v
API Gateway -> Lambda (FastAPI + Lambda Web Adapter)
        |
        v
generation/rag — Chat Orchestrator
        |
        |-- Embed query
        |-- Search FAISS RAG index
        |-- Build grounded prompt
        |-- Call LLM
        v
Grounded chatbot response
```

The recommendation pipeline and RAG chatbot are intentionally separated: different signals, different latency/cost profiles, different request paths.

---

## 3. Full Repository Structure

```text
fashion-recommendation-system/
│
├── README.md
├── pyproject.toml                          # Packaging, tool config (ruff, mypy, pytest, black)
├── Makefile                                # One-line dev commands
├── .env.example                            # Template — safe to commit
├── .env.local                              # Local dev env vars (gitignored)
├── .gitignore
│
├── src/
│   └── fashion_recommendation_system/      # Main Python package (pip install -e .)
│       ├── __init__.py
│       ├── config.py                       # Single env-driven config — all os.getenv() lives here
│       │                                   # Handles: S3 bucket names, Redis endpoint, SageMaker
│       │                                   # endpoint names, Redis host, feature flags
│       │
│       ├── common/                         # Shared utilities (no model/feature/business logic)
│       │   ├── __init__.py
│       │   ├── logging.py                  # Structured logging setup
│       │   ├── metrics.py                  # CloudWatch metric helpers
│       │   ├── exceptions.py               # Custom exception classes
│       │   ├── constants.py                # Project-wide constants
│       │   ├── serialization.py            # Serialization helpers
│       │   ├── s3.py                       # S3 read/write utilities, presigned URLs
│       │   ├── redis.py                    # Redis connection singleton
│       │   ├── aws_clients.py              # boto3 factory — LocalStack-aware
│       │   └── tracing.py                  # Request tracing / X-Ray helpers
│       │
│       ├── contracts/                      # Stable interfaces between all components
│       │   ├── __init__.py                 # Prevents training-serving skew via explicit schemas
│       │   ├── recommendation.py           # RecommendationRequest, RecommendationResponse
│       │   ├── ranking.py                  # XGBoostPredictionRequest, RankingResponse
│       │   ├── features.py                 # UserFeatureVector, ItemFeatureVector, CandidateItem
│       │   ├── rag.py                      # RagChatRequest, RagChatResponse
│       │   └── events.py                   # Event logging schemas (exposure, click, purchase)
│       │
│       ├── data/                           # Raw data ingestion, cleaning, validation, datasets
│       │   ├── __init__.py
│       │   ├── ingestion/                  # Step 1: Load raw source data — no transforms
│       │   │   ├── __init__.py
│       │   │   └── h_and_m.py              # Read raw CSVs, write parquet to S3 raw/
│       │   │
│       │   ├── transforms/                 # Step 2: Raw data cleaning — not feature engineering
│       │   │   ├── __init__.py             # Rule: clean_x.py asks "Is data valid and safe to use?"
│       │   │   ├── clean_transactions.py   # Date parsing, dedup, price/date filtering
│       │   │   ├── clean_articles.py       # Category normalization, text cleanup
│       │   │   ├── clean_users.py          # Age bucketing, missing demographic handling
│       │   │   ├── handle_nulls.py         # Config-driven null filling strategies
│       │   │   ├── handle_outliers.py      # Winsorize/cap strategies
│       │   │   ├── handle_duplicates.py    # Dedup logic
│       │   │   ├── normalize_types.py      # Date/ID type casting
│       │   │   └── standardize_categories.py
│       │   │
│       │   ├── validation/                 # Schema contracts — runs after ingestion and transforms
│       │   │   ├── __init__.py
│       │   │   ├── schemas.py              # Pandera / Great Expectations schema definitions
│       │   │   ├── validators.py           # Validation runner helpers
│       │   │   ├── data_quality_checks.py  # Null %, duplicate %, value range checks
│       │   │   └── leakage_checks.py       # Future data leakage detection in training sets
│       │   │
│       │   └── datasets/                   # Model-specific dataset objects
│       │       ├── __init__.py
│       │       ├── two_tower_dataset.py    # PyTorch Dataset — positive pairs + negative sampling
│       │       ├── xgboost_dataset.py     # XGBoost Pool builder — user × item candidate frame
│       │       └── rag_dataset.py          # RAG indexing dataset — article chunks
│       │
│       ├── features/                       # Feature engineering, selection, and feature store
│       │   ├── __init__.py
│       │   ├── feature_registry.py         # Central registry of all feature definitions
│       │   ├── spark_session.py            # SparkSession factory (local[*] vs Glue — no code change)
│       │   │
│       │   ├── user_features/              # User-level signals from purchase history
│       │   │   ├── __init__.py
│       │   │   ├── purchase_history.py     # Total purchases, unique items, purchase frequency
│       │   │   ├── recency.py              # Days since last purchase, recency score
│       │   │   ├── price_affinity.py       # Average spend, price bucket preference
│       │   │   ├── category_affinity.py    # Favorite categories, category distribution
│       │   │   ├── user_style_tags.py      # Aggregated style tags (from content_features if enabled)
│       │   │   └── build_user_features.py  # Composes all user feature builders
│       │   │
│       │   ├── item_features/              # Item-level signals from article metadata
│       │   │   ├── __init__.py
│       │   │   ├── article_metadata.py     # Category, department, colour, product age
│       │   │   ├── item_price.py           # Price bucket, price tier
│       │   │   ├── item_category.py        # Category encoding, hierarchy features
│       │   │   ├── item_style_tags.py      # Style tags from content_features (if enabled)
│       │   │   └── build_item_features.py  # Composes all item feature builders
│       │   │
│       │   ├── cross_features/             # User × item interaction signals — critical for XGBoost
│       │   │   ├── __init__.py
│       │   │   ├── user_item_category_overlap.py
│       │   │   ├── price_match.py          # Diff between user avg price and item price
│       │   │   ├── style_tag_overlap.py    # Tag overlap score between user profile and item
│       │   │   └── build_cross_features.py
│       │   │
│       │   ├── interaction_features.py     # User-item interaction matrix features
│       │   │
│       │   ├── selection/                  # Feature selection — not buried inside train.py
│       │   │   ├── __init__.py
│       │   │   ├── xgboost_feature_selection.py
│       │   │   ├── two_tower_feature_selection.py
│       │   │   ├── feature_importance.py
│       │   │   ├── shap_analysis.py
│       │   │   └── leakage_checks.py
│       │   │
│       │   ├── pipelines/                  # Feature pipeline composition scripts
│       │   │   ├── __init__.py
│       │   │   ├── build_clean_dataset.py
│       │   │   ├── build_user_features.py
│       │   │   ├── build_item_features.py
│       │   │   ├── build_cross_features.py
│       │   │   ├── build_training_features.py  # Assembles complete training feature tables
│       │   │   └── build_serving_features.py   # Assembles serving feature snapshot for Redis/S3
│       │   │
│       │   └── store/                      # Feature read/write to offline and online stores
│       │       ├── __init__.py
│       │       ├── offline_writer.py       # Write features to S3 (parquet, partitioned)
│       │       ├── online_writer.py        # Write hot features to Redis
│       │       ├── redis_hot_path.py       # Redis read patterns for serving
│       │       └── s3_feature_store.py     # S3 read patterns for batch/training
│       │
│       ├── models/                         # All model code: retrieval, ranking, content, registry
│       │   ├── __init__.py
│       │   │
│       │   ├── retrieval/                  # Stage 1 — Candidate generation
│       │   │   ├── __init__.py
│       │   │   ├── two_tower/              # Two-Tower embedding model (PyTorch)
│       │   │   │   ├── __init__.py
│       │   │   │   ├── model.py            # nn.Module — imported by train.py and inference.py
│       │   │   │   ├── loss.py             # Contrastive / in-batch negative loss
│       │   │   │   ├── dataset.py          # PyTorch Dataset — positive pairs + negative sampling
│       │   │   │   ├── preprocess.py       # Two-Tower-specific formatting (not general cleaning)
│       │   │   │   ├── train.py            # SageMaker Training Job entry point
│       │   │   │   ├── evaluate.py         # Recall@K, embedding similarity distribution
│       │   │   │   ├── export.py           # Export model.tar.gz for SageMaker
│       │   │   │   └── inference.py        # SageMaker inference handler (model_fn, predict_fn)
│       │   │   │
│       │   │   └── faiss_index/            # FAISS index lifecycle — treat as production artifact
│       │   │       ├── __init__.py
│       │   │       ├── build_index.py      # Build ANN index from item embeddings
│       │   │       ├── validate_index.py   # Recall validation and index health checks
│       │   │       ├── search.py           # ANN search — shared by local runner + Lambda
│       │   │       ├── index_metadata.py   # Version metadata for deployed index
│       │   │       └── publish_index.py    # Upload versioned index bundle to S3
│       │   │                               # Each version: faiss.index + item_id_mapping.parquet
│       │   │                               #               + index_metadata.json + validation_report.json
│       │   │
│       │   ├── ranking/                    # Stage 2 — Re-ranking
│       │   │   ├── __init__.py
│       │   │   └── xgboost/
│       │   │       ├── __init__.py
│       │   │       ├── model.py            # XGBoostClassifier/Ranker wrapper
│       │   │       ├── train.py            # SageMaker Training Job entry point
│       │   │       ├── evaluate.py         # NDCG@K, MAP@K, MRR, hit rate
│       │   │       ├── features.py         # Assemble candidate feature matrix at inference time
│       │   │       ├── preprocess.py       # XGBoost-specific formatting (build Pool object)
│       │   │       ├── feature_config.py   # Feature lists live here — not inside train.py
│       │   │       ├── export.py           # Export model.tar.gz for SageMaker
│       │   │       └── inference.py        # SageMaker inference handler
│       │   │
│       │   ├── content_features/           # OPTIONAL: Offline LLM tag extraction pipeline
│       │   │   ├── __init__.py             # Never runs on the online recommendation path
│       │   │   └── llm_tag_extractor/
│       │   │       ├── __init__.py
│       │   │       ├── prompts/            # Prompt templates for tag extraction
│       │   │       ├── fine_tune_lora.py   # LoRA/PEFT fine-tuning (manual trigger, runs rarely)
│       │   │       ├── batch_inference.py  # Batch: article description → style tags
│       │   │       ├── postprocess_tags.py # Validate, normalise, deduplicate extracted tags
│       │   │       ├── aggregate_user_tags.py  # article_tags + transactions → user_tag_features
│       │   │       └── evaluate_tags.py    # Tag precision, coverage, quality metrics
│       │   │
│       │   └── registry/                   # Model versioning and promotion
│       │       ├── __init__.py
│       │       ├── model_card.py           # Generate model card (metrics, data snapshot, owner)
│       │       ├── register_model.py       # Register artifact in SageMaker Model Registry
│       │       ├── promote_model.py        # Promote staging → production with approval gate
│       │       └── versioning.py           # Version naming, git SHA tracking, lineage
│       │
│       ├── serving/                        # Online serving — API, Lambda handlers, FAISS service
│       │   ├── __init__.py
│       │   │
│       │   ├── api/                        # FastAPI application (zero Lambda-specific code)
│       │   │   ├── __init__.py             # Lambda integration handled by Lambda Web Adapter
│       │   │   ├── main.py                 # FastAPI app definition
│       │   │   ├── routers/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── recommendations.py  # GET /recommendations/{user_id}
│       │   │   │   ├── chat.py             # POST /chat (RAG chatbot — future)
│       │   │   │   └── health.py           # GET /health
│       │   │   ├── services/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── recommendation_service.py   # Orchestrates retrieval → ranking pipeline
│       │   │   │   ├── feature_service.py          # Fetches user/item features (Redis / S3)
│       │   │   │   ├── cache_service.py            # Redis read/write abstraction
│       │   │   │   ├── business_rules.py           # Remove purchased items, out-of-stock, diversity rules
│       │   │   │   ├── postprocessing.py           # Re-order, truncate, format final top-K
│       │   │   │   └── fallback.py                 # Graceful degradation: segment → category → trending
│       │   │   └── middleware/
│       │   │       ├── __init__.py
│       │   │       ├── auth.py
│       │   │       ├── rate_limit.py
│       │   │       ├── request_id.py       # Attach request ID for distributed tracing
│       │   │       └── error_handler.py    # Consistent error response formatting
│       │   │
│       │   ├── lambdas/                    # Lambda entry points — thin wrappers only
│       │   │   ├── __init__.py
│       │   │   ├── api_handler.py          # Lambda entry point — imports serving/api/main.py
│       │   │   ├── faiss_handler.py        # Lambda entry point — FAISS search only
│       │   │   └── chat_handler.py         # Lambda entry point — RAG chatbot (future)
│       │   │
│       │   └── faiss_lambda/               # FAISS online search service
│       │       ├── __init__.py
│       │       ├── handler.py              # Lambda handler for vector search requests
│       │       ├── index_loader.py         # Load versioned FAISS index from S3 on cold start
│       │       ├── search_service.py       # ANN search logic — returns top-N article IDs
│       │       └── warmup.py               # Lambda warmup to reduce cold start impact
│       │
│       ├── generation/                     # STANDALONE CHATBOT — not part of recommendation funnel
│       │   ├── __init__.py
│       │   └── rag/                        # RAG: answers natural language product queries
│       │       ├── __init__.py             # Triggered by POST /chat; shares nothing with recsys pipeline
│       │       ├── api/
│       │       │   ├── __init__.py
│       │       │   ├── app.py
│       │       │   ├── routes.py
│       │       │   ├── request_models.py
│       │       │   └── response_models.py
│       │       ├── indexing/               # Build the RAG FAISS index from article text
│       │       │   ├── __init__.py
│       │       │   ├── chunk_articles.py
│       │       │   ├── embed_chunks.py
│       │       │   ├── build_faiss_rag_index.py
│       │       │   ├── validate_rag_index.py
│       │       │   └── publish_rag_index.py
│       │       ├── retrieval/
│       │       │   ├── __init__.py
│       │       │   ├── query_encoder.py
│       │       │   ├── faiss_search.py
│       │       │   └── rerank_chunks.py
│       │       ├── generation/
│       │       │   ├── __init__.py
│       │       │   ├── prompt_builder.py
│       │       │   ├── llm_client.py
│       │       │   ├── guardrails.py
│       │       │   └── response_parser.py
│       │       └── service/
│       │           ├── __init__.py
│       │           ├── chat_orchestrator.py
│       │           ├── product_grounding.py
│       │           └── fallback.py
│       │
│       ├── pipelines/                      # SageMaker / Airflow / Step Functions orchestration logic
│       │   ├── __init__.py
│       │   ├── sagemaker/                  # SageMaker Pipeline definitions
│       │   │   ├── __init__.py
│       │   │   ├── two_tower_pipeline.py
│       │   │   ├── xgboost_pipeline.py
│       │   │   ├── llm_tag_pipeline.py
│       │   │   ├── rag_index_pipeline.py
│       │   │   └── shared_steps.py         # Reusable pipeline steps (data validation, model reg)
│       │   ├── airflow/                    # MWAA DAGs for scheduled workflows
│       │   │   ├── dags/
│       │   │   │   ├── daily_feature_refresh.py
│       │   │   │   ├── weekly_llm_tag_extraction.py
│       │   │   │   ├── two_tower_retrain.py
│       │   │   │   ├── xgboost_retrain.py
│       │   │   │   └── rag_index_refresh.py
│       │   │   └── plugins/
│       │   └── step_functions/             # Step Functions ASL definitions
│       │       ├── recommendation_training.asl.json
│       │       ├── feature_refresh.asl.json
│       │       └── rag_refresh.asl.json
│       │
│       └── evaluation/                     # Offline and online evaluation — not buried in train.py
│           ├── __init__.py
│           ├── offline/
│           │   ├── __init__.py
│           │   ├── retrieval_metrics.py    # Recall@K, Precision@K, coverage, diversity
│           │   ├── ranking_metrics.py      # NDCG@K, MAP@K, MRR, hit rate, calibration
│           │   ├── diversity_metrics.py    # Intra-list diversity, serendipity
│           │   └── calibration.py         # Popularity calibration checks
│           ├── online/
│           │   ├── __init__.py
│           │   ├── ab_testing.py           # A/B test assignment and analysis helpers
│           │   ├── exposure_logging.py     # Log what was shown to whom
│           │   ├── clickstream_metrics.py  # CTR, add-to-cart, conversion tracking
│           │   └── attribution.py          # Revenue attribution per recommendation
│           └── rag/
│               ├── __init__.py
│               ├── faithfulness.py         # Groundedness checks
│               ├── retrieval_recall.py     # Chunk retrieval quality
│               ├── answer_quality.py       # Answer relevance scoring
│               └── safety_eval.py          # Hallucination and safety checks
│
├── configs/                                # YAML configs for model/feature hyperparams
│   │                                       # Does NOT contain infrastructure env vars (those in config.py)
│   ├── data/
│   │   ├── preprocessing.yaml              # Null handling, outlier strategies per column
│   │   ├── raw_sources.yaml                # Source file paths and formats
│   │   ├── s3_paths.yaml                   # S3 prefix layout
│   │   └── schemas.yaml                    # Expected column types and constraints
│   ├── features/
│   │   ├── feature_sets.yaml               # Which feature groups to enable
│   │   ├── user_features.yaml              # Recency window, aggregation configs
│   │   ├── item_features.yaml              # Popularity lookback, bucketing configs
│   │   └── cross_features.yaml             # Cross feature computation configs
│   ├── models/
│   │   ├── two_tower.yaml                  # Embedding dim, layers, learning rate, batch size
│   │   ├── xgboost_ranker.yaml            # Feature lists, tree params, eval metric
│   │   ├── llm_tag_extractor.yaml          # LoRA rank, base model, batch config
│   │   └── rag.yaml                        # Chunk size, overlap, top-k retrieval
│   └── serving/
│       ├── recommendation_api.yaml         # Top-K, timeout, fallback config
│       ├── faiss.yaml                      # Index type, nprobe, candidate count
│       ├── redis.yaml                      # TTL, key prefixes, max memory policy
│       └── rag_api.yaml                    # LLM temperature, max tokens, guardrail config
│
├── pipelines/                              # Root-level runner scripts — simple entry points
│   │                                       # These call into src/.../pipelines/ for orchestration logic
│   ├── run_data_pipeline.py                # Step 1: ingestion + transforms + validation
│   ├── run_feature_pipeline.py             # Step 2: build all feature tables
│   ├── run_training_pipeline.py            # Step 3: train Two-Tower + XGBoost
│   ├── run_index_pipeline.py               # Step 4: build + publish FAISS retrieval index
│   ├── run_content_features.py             # Optional: LLM batch inference → enriched/
│   └── run_rag_pipeline.py                 # Standalone: build RAG index (future)
│
├── infra/                                  # Infrastructure as Code — Terraform only
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── modules/
│   │   ├── s3/
│   │   ├── lambda/
│   │   ├── api_gateway/
│   │   ├── sagemaker/
│   │   └── elasticache/
│   ├── environments/
│   │   ├── local/                          # LocalStack tfvars
│   │   ├── dev/
│   │   ├── staging/
│   │   └── aws/                            # Production tfvars
│   └── permissions/
│       ├── recommendation_api_policy.json
│       ├── sagemaker_execution_role.json
│       ├── batch_jobs_policy.json
│       └── least_privilege_notes.md
│
├── deployment/                             # All build and release assets — centralized
│   ├── docker/
│   │   ├── recommendation_api.Dockerfile   # FastAPI + Lambda Web Adapter
│   │   ├── faiss_lambda.Dockerfile         # FAISS search Lambda
│   │   ├── rag_api.Dockerfile              # RAG chatbot API
│   │   ├── two_tower_inference.Dockerfile  # SageMaker Two-Tower endpoint
│   │   └── xgboost_inference.Dockerfile   # SageMaker XGBoost endpoint
│   ├── scripts/
│   │   ├── build_images.sh
│   │   ├── push_images.sh
│   │   ├── deploy_dev.sh
│   │   ├── deploy_staging.sh
│   │   ├── deploy_prod.sh
│   │   └── rollback.sh
│   └── manifests/
│       ├── lambda_env.json                 # Lambda function environment variables
│       ├── sagemaker_endpoints.json        # Endpoint config (instance type, variant weights)
│       └── api_gateway_routes.json         # Route definitions
│
├── docker/                                 # Local dev and test compose files
│   ├── docker-compose.yml                  # Full local stack: API + Redis + LocalStack
│   ├── docker-compose.test.yml             # Integration test stack
│   └── localstack/
│       └── init-scripts/                   # Auto-create S3 buckets at LocalStack startup
│
├── tests/
│   ├── unit/
│   │   ├── test_data.py                    # data/transforms/, data/validation/
│   │   ├── test_features.py                # features/user_features/, item_features/, cross_features/
│   │   ├── test_models.py                  # model wrappers, feature_config, preprocess
│   │   ├── test_serving.py                 # business_rules, postprocessing, fallback
│   │   └── test_contracts.py               # Pydantic schema validation
│   ├── integration/
│   │   ├── test_faiss_search.py            # FAISS index build + search round-trip
│   │   └── test_recommendation_pipeline.py # End-to-end with LocalStack
│   ├── load/
│   │   └── test_api_load.py                # P95/P99 latency, cold-start impact
│   ├── contract/
│   │   ├── test_recommendation_api_contract.py
│   │   └── test_sagemaker_contracts.py     # Two-Tower and XGBoost endpoint schemas
│   └── conftest.py                         # Shared fixtures (LocalStack, mock Redis)
│
├── notebooks/                              # Exploration and experiments — never import from src/
│   ├── 01_eda_articles.ipynb
│   ├── 02_eda_transactions.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_two_tower_experiments.ipynb
│   └── 05_xgboost_experiments.ipynb
│
├── data_contracts/                         # Explicit data schemas — protect against upstream drift
│   ├── transactions.schema.json
│   ├── users.schema.json
│   ├── articles.schema.json
│   ├── user_features.schema.json
│   ├── item_features.schema.json
│   ├── xgboost_training.schema.json
│   └── rag_chunks.schema.json
│
├── model_artifacts/                        # Local placeholders ONLY — real artifacts live in S3
│   ├── README.md                           # Do not commit model files or FAISS indexes to git
│   ├── two_tower/
│   ├── xgboost/
│   ├── faiss/
│   └── rag/
│
├── monitoring/
│   ├── dashboards/                         # CloudWatch dashboard definitions (JSON)
│   ├── alarms/                             # CloudWatch alarm definitions
│   └── logs/                               # Log format specs, query templates
│
├── ci/
│   ├── github-actions/
│   │   ├── lint.yml
│   │   ├── test.yml
│   │   ├── build-images.yml
│   │   ├── deploy-dev.yml
│   │   ├── deploy-staging.yml
│   │   └── deploy-prod.yml
│   └── quality/
│       ├── ruff.toml
│       ├── mypy.ini
│       ├── pytest.ini
│       └── bandit.yaml
│
├── scripts/                                # Utility and ops scripts
│   ├── create_sample_dataset.py            # Generate dev subset from full CSVs
│   ├── seed_localstack.py                  # Bootstrap LocalStack with test data
│   ├── upload_artifacts_to_s3.py           # Push models + FAISS index to S3
│   └── deploy_sagemaker_endpoint.py        # Register + deploy SageMaker endpoint
│
├── dataset/
│   ├── full/                               # Raw H&M CSVs (gitignored)
│   └── sample/                             # ~1K stratified users (see notebooks/stratified_user_sampling.ipynb)
│
├── docs/
│   ├── system-design/                      # Current architecture documentation
│   │   ├── project-structure.md      # This file
│   │   ├── infrastructure-layer.md
│   │   ├── schema-info.md
│   │   ├── project-description.md
│   │   └── feature-engineering/
│   ├── adr/                                # Architecture Decision Records
│   │   ├── 0001-use-two-stage-retrieval-ranking.md
│   │   ├── 0002-use-faiss-for-ann.md
│   │   ├── 0003-use-xgboost-for-ranking.md
│   │   ├── 0004-separate-rag-from-recommendations.md
│   │   ├── 0005-use-lambda-web-adapter.md
│   │   ├── 0006-use-terraform-only-for-iac.md
│   │   └── 0007-hybrid-config-yaml-plus-env.md
│   ├── superpowers/                        # Implementation plans and specs
│   │   ├── plans/
│   │   └── specs/
│   ├── ref-project-info/                   # Legacy reference — archive only, do not modify
│   ├── implementation-info/
│   └── outcomes-info/
│
├── requirements-training.txt               # torch, xgboost, pyspark, sagemaker-sdk (~2GB)
├── requirements-content.txt               # transformers, peft, datasets, accelerate (~5GB)
├── requirements-serving.txt               # fastapi, uvicorn, redis, boto3, faiss-cpu (~50MB)
└── requirements-dev.txt                   # pytest, black, ruff, mypy, moto, localstack
```

---

## 4. Configuration Design (Hybrid Approach)

Two distinct configuration concerns are kept separate:

### `configs/` — Model and Feature Configuration (YAML)

Used for: hyperparameters, feature lists, preprocessing strategies, Top-K values, batch sizes. These are not secrets and change across experiments, not environments.

```yaml
# configs/models/xgboost_ranker.yaml
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

### `config.py` — Infrastructure Configuration (env vars)

Used for: S3 bucket names, Redis endpoints, SageMaker endpoint names, feature flags. These change between local/dev/prod environments and are injected via environment variables.

```python
# src/fashion_recommendation_system/config.py
import os

S3_BUCKET = os.getenv("S3_BUCKET", "fashion-recsys-dev")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
TWO_TOWER_ENDPOINT = os.getenv("TWO_TOWER_ENDPOINT", "http://localhost:8080")
XGBOOST_ENDPOINT = os.getenv("XGBOOST_ENDPOINT", "http://localhost:8081")
FAISS_LAMBDA_ARN = os.getenv("FAISS_LAMBDA_ARN", "")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "")  # Empty = real AWS
```

**Rule:** `os.getenv()` calls live only in `config.py`. All other modules import from `config`.

---

## 5. S3 Data Lineage

Each pipeline stage owns a distinct S3 prefix. Downstream stages consume only from the previous stage's output — never skip a level.

```text
s3://bucket/
├── raw/                        ← data/ingestion/ writes here
│   ├── articles.parquet
│   ├── customers.parquet
│   └── transactions.parquet
│
├── processed/                  ← data/transforms/ + data/validation/ writes here
│   ├── articles_clean.parquet
│   ├── customers_clean.parquet
│   ├── transactions_clean.parquet
│   └── unified_fact.parquet
│
├── enriched/                   ← models/content_features/ writes here (OPTIONAL)
│   ├── article_tags.parquet
│   └── user_tag_features.parquet
│
├── features/                   ← features/ writes here
│   ├── user_features.parquet
│   ├── item_features.parquet
│   ├── interaction_features.parquet
│   └── ranking_features.parquet
│
├── models/                     ← SageMaker Training Jobs write here
│   ├── two_tower/model.tar.gz
│   ├── xgboost/model.tar.gz
│   └── content_features/model.tar.gz   (optional)
│
├── indexes/                    ← pipeline scripts write here
│   ├── faiss_retrieval/version=YYYY-MM-DD-NNN/
│   │   ├── faiss.index
│   │   ├── item_id_mapping.parquet
│   │   ├── index_metadata.json
│   │   └── validation_report.json
│   └── faiss_rag/version=YYYY-MM-DD-NNN/
│
└── rag/
    ├── chunks/
    ├── embeddings/
    └── index_inputs/
```

---

## 6. Two FAISS Indexes — Never Conflated

| Index | Built By | Contains | Used By |
|---|---|---|---|
| `indexes/faiss_retrieval/` | `models/retrieval/faiss_index/` | Item embedding vectors (Two-Tower output) | `GET /recommendations` only |
| `indexes/faiss_rag/` | `generation/rag/indexing/` | Article description text chunks (text encoder output) | `POST /chat` only |

---

## 7. `requirements` Split Rationale

Lambda cold start and package size are directly linked. PyTorch (~700MB) must never enter a Lambda deployment package.

| File | Contents | Used By |
|---|---|---|
| `requirements-training.txt` | torch, xgboost, pyspark, sagemaker-sdk | `models/retrieval/`, `models/ranking/`, `features/` |
| `requirements-content.txt` | transformers, peft, datasets, accelerate | `models/content_features/` only — must stay isolated |
| `requirements-serving.txt` | fastapi, uvicorn, redis, boto3, faiss-cpu | Lambda (API + FAISS), local API server |
| `requirements-dev.txt` | pytest, black, ruff, mypy, moto, localstack | Local development and CI only |

---

## 8. `train.py` vs `inference.py` Split

Every model directory contains both files. They deploy to completely different environments.

| File | Deployed To | Dependencies | Entry Point |
|---|---|---|---|
| `train.py` | SageMaker Training Job | `requirements-training.txt` | `estimator.fit()` |
| `inference.py` | SageMaker Endpoint | lightweight subset | `model_fn`, `predict_fn` |
| `model.py` | Both (imported) | shared architecture only | — |

---

## 9. Key Separation Rules

### Cleaning vs Feature Engineering

```text
Is the raw data valid, consistent, and safe to use?
  → data/transforms/

Which predictive signals should the model use?
  → features/

Two-Tower-specific tensor formatting?
  → models/retrieval/two_tower/preprocess.py

XGBoost-specific Pool construction?
  → models/ranking/xgboost/preprocess.py
```

### Online Path Must Stay Lightweight

At inference time, the recommendation API must only:
- Fetch precomputed features (Redis / S3)
- Call SageMaker endpoints
- Call FAISS Lambda
- Apply business rules
- Return response

It must NOT perform heavy feature engineering, LLM calls, or batch transformations.

### LLM Tag Extraction is Offline Only

`models/content_features/` runs as a periodic batch job. Its output feeds the normal feature pipeline. It never runs on the recommendation request path.

---

## 10. Preprocessing Location Reference

| Work | Correct Location |
|---|---|
| Fill missing article description | `data/transforms/handle_nulls.py` |
| Remove duplicate transactions | `data/transforms/handle_duplicates.py` |
| Cap price outliers | `data/transforms/handle_outliers.py` |
| Normalize date columns | `data/transforms/normalize_types.py` |
| Validate required columns | `data/validation/schemas.py` |
| Validate no null user/item IDs | `data/validation/data_quality_checks.py` |
| Build user purchase history | `features/user_features/purchase_history.py` |
| Build price affinity | `features/user_features/price_affinity.py` |
| Build category overlap | `features/cross_features/user_item_category_overlap.py` |
| Select XGBoost features | `features/selection/xgboost_feature_selection.py` |
| Prepare Two-Tower pairs + negative sampling | `models/retrieval/two_tower/dataset.py` |
| Define XGBoost feature list | `models/ranking/xgboost/feature_config.py` |
| Write features to S3 | `features/store/offline_writer.py` |
| Write hot features to Redis | `features/store/online_writer.py` |
| Fetch features at inference | `serving/api/services/feature_service.py` |

---

## 11. Makefile Reference

```makefile
# Core recommendation pipeline (run in order)
data-pipeline:        # Step 1: raw → clean + validated
feature-pipeline:     # Step 2: clean → model-ready features
train:                # Step 3: train Two-Tower + XGBoost
build-index:          # Step 4: build + publish recommendation FAISS index

# Optional enrichment (run before feature-pipeline to enable tag features)
content-features:     # LLM batch inference → article_tags + user_tag_features
finetune-tags:        # One-off: fine-tune the tag extraction model (manual trigger)

# Standalone chatbot (independent of recommendation pipeline)
rag-pipeline:         # Build RAG product knowledge index (future)

# Development
dev:                  # Start full local stack (API + Redis + LocalStack)
test-unit:            # Run unit tests
test-integration:     # Spin up LocalStack, run integration tests, tear down

# Infrastructure
infra-up:             # terraform apply (AWS)
infra-down:           # terraform destroy (AWS)
```

---

## 12. AWS Service Mapping

| Component | AWS Service |
|---|---|
| Public API | API Gateway |
| FastAPI runtime | Lambda container + Lambda Web Adapter |
| FAISS search | Lambda + FAISS (index in Lambda memory, loaded from S3) |
| Hot user features | ElastiCache Redis |
| Offline features | S3 (parquet) |
| Two-Tower inference | SageMaker real-time endpoint |
| XGBoost inference | SageMaker real-time endpoint |
| Batch feature jobs | SageMaker Processing or Glue |
| Training jobs | SageMaker Training Jobs |
| ML workflow orchestration | SageMaker Pipelines + Step Functions |
| Model registry | SageMaker Model Registry |
| Container registry | ECR |
| Artifact store | S3 |
| Monitoring | CloudWatch + X-Ray |
| CI/CD | GitHub Actions |
| IaC | Terraform |

---

## 13. Anti-Patterns to Avoid

1. Putting preprocessing logic inside `train.py`
2. Rebuilding feature logic separately for training and serving
3. Calling LLMs on the online recommendation path
4. Deploying a FAISS index without its `item_id_mapping.parquet`
5. Using random train-test split for time-series recommendation data
6. Hardcoding feature lists inside training scripts (use `feature_config.py`)
7. Mixing RAG chatbot code with recommendation orchestration
8. Creating AWS resources manually from the console
9. Committing model artifacts, FAISS indexes, or embeddings to git
10. Skipping contract tests between services
11. Not tracking which data snapshot trained each model version
12. Not having fallback recommendations for when personalization fails
13. Calling `os.getenv()` outside of `config.py`
14. Importing from `src/` inside notebooks
