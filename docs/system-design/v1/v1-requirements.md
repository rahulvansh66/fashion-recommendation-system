# Fashion Recommendation System — V1 Requirements

| Field         | Value                                                                                                     |
|---------------|-----------------------------------------------------------------------------------------------------------|
| **Status**    | Approved                                                                                                  |
| **Version**   | v1.0                                                                                                      |
| **Last Updated** | 2026-05-29                                                                                             |
| **Author**    | rahul.vansh                                                                                               |
| **Related Docs** | [`v1-hld.md`](./v1-hld.md) · [`v1-deliverable.md`](../v1-deliverable.md) · [`schema-info.md`](../schema-info.md) |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Stakeholders & Actors](#2-stakeholders--actors)
3. [Scope](#3-scope)
4. [Functional Requirements](#4-functional-requirements)
   - 4.1 [User Interface & Authentication](#41-user-interface--authentication)
   - 4.2 [Recommendation Pipeline](#42-recommendation-pipeline)
   - 4.3 [Caching & Pre-Warming](#43-caching--pre-warming)
   - 4.4 [Backend API](#44-backend-api)
   - 4.5 [ML Inference](#45-ml-inference)
   - 4.6 [Offline Batch Pipelines](#46-offline-batch-pipelines)
   - 4.7 [Observability & Alerting](#47-observability--alerting)
5. [Non-Functional Requirements](#5-non-functional-requirements)
   - 5.1 [Performance](#51-performance)
   - 5.2 [Availability & Reliability](#52-availability--reliability)
   - 5.3 [Scalability](#53-scalability)
   - 5.4 [Security](#54-security)
   - 5.5 [Observability](#55-observability)
   - 5.6 [Cost](#56-cost)
   - 5.7 [Maintainability & Operability](#57-maintainability--operability)
   - 5.8 [Portability & Deployability](#58-portability--deployability)
6. [Constraints](#6-constraints)
7. [Assumptions](#7-assumptions)
8. [Out of Scope (V1)](#8-out-of-scope-v1)
9. [Requirements Traceability Matrix](#9-requirements-traceability-matrix)

---

## 1. Introduction

### 1.1 Purpose

This document defines the complete functional and non-functional requirements for the Fashion Recommendation System, version 1. It is the reference contract between design, implementation, and acceptance testing. All implementation decisions in [`v1-hld.md`](./v1-hld.md) must be traceable to at least one requirement listed here.

### 1.2 Background

The system serves personalized top-10 fashion article recommendations for customers of the H&M dataset. V1 is built as a learning-grade, production-pattern system: it demonstrates real-world ML engineering patterns (two-stage retrieval + ranking, result caching, circuit breakers, canary deployment) while remaining deployable on a $30–40 total budget.

### 1.3 Requirement Priority Convention

This document uses **MoSCoW** prioritisation:

| Label     | Meaning                                                                 |
|-----------|-------------------------------------------------------------------------|
| **MUST**  | Non-negotiable. V1 cannot ship without this.                            |
| **SHOULD** | Strongly desired. Omission requires explicit sign-off.                 |
| **COULD** | Nice to have. Included only if effort is low.                           |
| **WONT**  | Explicitly deferred to v1.1 or later. Documented to prevent scope creep. |

---

## 2. Stakeholders & Actors

| Actor                         | Role                                                                                                        | Primary Interaction                                         |
|-------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| **End user** (v1: portfolio reviewer / developer) | Views recommendations through the web UI. Not an authenticated production user in v1. | Browses the user-picker page; clicks a customer card to view top-10 recommendations. |
| **ML engineer**               | Trains models, monitors drift, approves model promotions.                                                   | SageMaker Model Registry approval workflow; CloudWatch dashboards. |
| **API consumer** (programmatic) | Consumes the `GET /recommendations/{user_id}` JSON endpoint directly.                                   | API Gateway HTTP API → Fargate application.                                        |
| **CI/CD system** (GitHub Actions) | Builds, tests, and deploys the application on every push.                                             | GitHub Actions pipeline; Terraform; ECR; SageMaker Pipelines. |

---

## 3. Scope

### 3.1 In Scope

- End-to-end online serving pipeline: Cache → Retrieve → Filter → Rank → Order (5 stages)
- Server-rendered web application with unified frontend and backend (FastAPI monolith on ECS Fargate)
- Visible latency demonstration (pre-warmed vs. cold recommendations)
- Backend API with rate limiting, circuit breakers, and graceful fallbacks
- Offline batch pipelines for data preparation, feature engineering, ML training, and index building
- SQS-based cache pre-warming work queue
- CI/CD pipeline with canary model deployment and auto-rollback
- Full observability stack (CloudWatch metrics, X-Ray tracing, SNS alarms)
- Infrastructure as Code (Terraform) for one-command deploy and destroy

### 3.2 Explicitly Out of Scope (V1)

See [Section 8](#8-out-of-scope-v1) for the full deferred list with rationale.

---

## 4. Functional Requirements

> Each requirement carries an ID in the form `FR-{area}-{number}`. IDs are stable and used in the traceability matrix.

---

### 4.1 User Interface & Authentication

#### FR-UI-01 — Login Page
**Priority:** MUST

The system must display a login page as the application entry point. In v1, the only accepted credential pair is `rr / rr`. On successful login, the user is redirected to the user-picker page.

**Acceptance criteria:**
- `GET /` renders a login form.
- `POST /login` with `username=rr&password=rr` sets a signed session cookie and returns HTTP 302 to the user-picker page.
- Any other credential pair returns HTTP 401.

---

#### FR-UI-02 — User-Picker Page
**Priority:** MUST

After login, the system must display exactly six customer cards representing the six most-active customers in the dataset. Each card must show: `customer_id`, `age`, and `current_date` (today's UTC date, computed at page-load time).

**Acceptance criteria:**
- Page renders six cards populated from the `active:users:top6` Redis key.
- `current_date` reflects `datetime.utcnow().date().isoformat()` at render time — it changes naturally across calendar days without a deployment.
- Each card is individually clickable.

---

#### FR-UI-03 — Pre-Warm Badge
**Priority:** MUST

The first three customer cards must display a visible "pre-warmed" badge indicating that their recommendations are already cached. The last three cards must have no such badge.

**Acceptance criteria:**
- Cards at positions 1–3 in the `active:users:top6` list render with a pre-warm indicator.
- Cards at positions 4–6 render without the indicator.
- The indicator is purely informational; it does not gate any functionality.

---

#### FR-UI-04 — Recommendations Fragment (HTMX Partial Update)
**Priority:** MUST

Clicking a customer card must trigger an HTMX request to the backend. The recommendations for that customer must appear in-page without a full page reload.

**Acceptance criteria:**
- Clicking card N issues `GET /recommendations/{customer_id}?age={age}&date={today}&k=10` via HTMX.
- The recommendations fragment (top-10 article list with article metadata) is injected into the page via HTMX `hx-swap`.
- No full page reload occurs.

---

### 4.2 Recommendation Pipeline

#### FR-PIPE-01 — Five-Stage Online Pipeline
**Priority:** MUST

The recommendation serving path must implement exactly five ordered stages: Stage 0 (Cache Check), Stage 1 (Retrieve), Stage 2 (Filter), Stage 3 (Rank), Stage 4 (Order). Stages 1–4 run only on a cache miss.

**Acceptance criteria:**
- A request for a pre-warmed user returns the cached result after Stage 0 with no invocations of downstream ML services.
- A request for a non-cached user executes all five stages in order.

---

#### FR-PIPE-02 — Stage 0: Result Cache Check
**Priority:** MUST

The pipeline must first check the Redis key `reco:{customer_id}`. If the key exists and its embedded `created_at` is within 12 hours, the cached JSON list of 10 articles must be returned immediately.

**Acceptance criteria:**
- Cache hit: response contains the cached top-10 list; no SageMaker invocations occur.
- Cache miss or stale entry (> 12 h): pipeline proceeds to Stage 1.
- Cache TTL is set to 43,200 seconds (12 hours).

---

#### FR-PIPE-03 — Stage 1: Retrieval (Feature Fetch + User Embedding + ANN Search)
**Priority:** MUST

Stage 1 must perform three sequential sub-steps:
1. Fetch user features from Redis (`HGETALL user:{customer_id}:features`); fall back to S3 on a cache miss and populate Redis with a 1-hour TTL.
2. Invoke the SageMaker `two-tower-user-tower` endpoint with the user feature vector; receive a 256-dimensional embedding.
3. Invoke the FAISS Lambda with the user embedding; receive the top-100 candidate article IDs with similarity scores.

**Acceptance criteria:**
- User features are resolved from Redis on warm requests; from S3 on the first request for that user.
- The SageMaker endpoint receives a correctly shaped feature vector and returns a 256-dim embedding.
- The FAISS Lambda returns exactly 100 candidate article IDs (or fewer if the index contains fewer items).

---

#### FR-PIPE-04 — Stage 2: Filter (Seen Items)
**Priority:** MUST

Stage 2 must remove from the candidate set any article IDs present in the Redis `seen:{customer_id}` set (items the user has already purchased).

**Acceptance criteria:**
- Any article ID present in `seen:{customer_id}` is absent from the Stage 3 input.
- If `seen:{customer_id}` is empty and no user features exist (cold-start user), the pipeline short-circuits to return `popular:items:top100` from Redis.

---

#### FR-PIPE-05 — Stage 3: Ranking (CatBoost)
**Priority:** MUST

Stage 3 must invoke the SageMaker `catboost-ranker` endpoint with a batch of feature vectors (one per candidate article). Each vector must include: user features, item features, and cross features (preferred category vs. item category, price affinity vs. item price, days since last purchase in category). The endpoint must return candidates sorted by predicted purchase probability.

**Acceptance criteria:**
- Item features are fetched from Redis via a single `HMGET` call covering all remaining candidates.
- Cross features are computed in the orchestrator Lambda before the CatBoost invocation.
- CatBoost returns a score per candidate, and candidates are sorted descending by score.

---

#### FR-PIPE-06 — Stage 4: Diversity-Aware Reorder
**Priority:** MUST

Stage 4 must apply the following reorder rule to the CatBoost-ranked candidates before writing to cache:

```
positions 1–4   → top 4 by raw CatBoost score
positions 5–6   → top 2 by diversity_score relative to positions 1–4
positions 7–10  → next 4 by raw CatBoost score (excluding those in positions 5–6)
```

The diversity score formula is:

```
diversity_score(c, S) = w1 * categorical_diff(c.product_type_no, S)
                      + w2 * categorical_diff(c.colour_group_code, S)
                      + w3 * bucket_diff(c.price_bucket, S)
```

Where `categorical_diff` returns 1 if the value differs from all items in set S, and `bucket_diff` normalises the minimum price-bucket distance. Default weights `w1 = w2 = w3 = 1.0` must be configurable via Lambda environment variables without redeployment.

**Acceptance criteria:**
- Positions 1–4 always reflect the top-4 CatBoost-ranked items.
- Positions 5–6 introduce at least one item from a different `product_type_no` or `colour_group_code` than positions 1–4 (when such diversity exists in the candidate pool).
- Weights are read from environment variables at cold-start time; changing them requires only an environment variable update.

---

#### FR-PIPE-07 — Rate Limiting
**Priority:** MUST

The pipeline must enforce two independent rate-limit layers:
1. **Stage-level (API Gateway):** 60 requests per second, burst 100.
2. **Application-level:** 30 requests per minute per `customer_id` using a token-bucket implemented as FastAPI middleware with a Redis counter.

**Acceptance criteria:**
- Requests exceeding the API Gateway limit receive HTTP 429 before reaching the Fargate application.
- Requests exceeding the per-customer application limit receive HTTP 429 from the application.

---

#### FR-PIPE-08 — Circuit Breakers & Fallbacks
**Priority:** MUST

Every downstream dependency call (Redis, SageMaker user-tower, FAISS Lambda, SageMaker CatBoost) must be wrapped in a circuit breaker. The breaker trips after 5 failures within 30 seconds and remains open for 30 seconds before allowing a single probe request.

Fallback behaviour per failing dependency:

| Failing Dependency          | Fallback Behaviour                                                             |
|-----------------------------|--------------------------------------------------------------------------------|
| Redis (cache or feature)    | Skip cache; continue pipeline; emit CloudWatch alarm.                          |
| SageMaker user-tower        | Use `embedding:user:{customer_id}` from Redis (24-hour TTL). If absent, serve popular items. |
| FAISS Lambda                | Return `popular:items:by_category:{cat}` from Redis.                           |
| SageMaker CatBoost          | Return FAISS top-K ordered by raw similarity score; still apply diversity reorder. |
| All ML dependencies open    | Return `popular:items:top100` from Redis with `degraded: true` in the response JSON. |

**Acceptance criteria:**
- Each fallback path is exercisable by injecting a simulated downstream failure in integration tests.
- Every fallback emits a `recommendation.fallback.{component}` CloudWatch metric.

---

### 4.3 Caching & Pre-Warming

#### FR-CACHE-01 — Recommendation Result Cache
**Priority:** MUST

Recommendation results for a customer must be written to Redis key `reco:{customer_id}` as a JSON string immediately after Stage 4, with a TTL of 43,200 seconds (12 hours).

**Acceptance criteria:**
- After a cache-miss pipeline run, the result is readable from `reco:{customer_id}`.
- A subsequent request within the TTL returns the cached result without running the pipeline.

---

#### FR-CACHE-02 — Feature Caches
**Priority:** MUST

User features must be cached at `user:{customer_id}:features` with a 1-hour TTL. Item features must be cached at `item:{article_id}:features` with a 24-hour TTL. Both are populated on S3 cache miss and refreshed nightly by the Glue batch pipeline.

**Acceptance criteria:**
- A second request for the same user within 1 hour does not trigger an S3 read for user features.
- Item features survive for 24 hours without re-reading S3.

---

#### FR-CACHE-03 — Cache Pre-Warming (SQS Work Queue)
**Priority:** MUST

A nightly EventBridge cron (05:00 UTC) must trigger a Lambda producer that reads the top-3 entries from `active:users:top6` in Redis and enqueues one SQS message per customer. A consumer Lambda (reserved concurrency 5) must process each message by running the full five-stage pipeline and writing the result to `reco:{customer_id}` before the first user request of the day.

**Acceptance criteria:**
- By 06:00 UTC daily, `reco:{customer_id}` exists for all three pre-warm customers.
- The consumer is idempotent: processing the same message twice (at-least-once delivery) does not double-write or corrupt the cache. Idempotency is enforced via `SETNX prewarm:done:{customer_id}:{date}`.
- Messages that fail three delivery attempts are routed to the DLQ.

---

#### FR-CACHE-04 — Cold-Start Fallback Keys
**Priority:** MUST

The Redis keys `popular:items:top100` and `popular:items:by_category:{category}` must be populated nightly by the Glue cache warm-up job with a 24-hour TTL. They serve as the fallback recommendation set for new users and for FAISS Lambda failures, respectively.

**Acceptance criteria:**
- Both key families are present in Redis by 04:00 UTC every day.
- A cold-start user (no features, empty seen-set) receives the top-100 popular items.

---

### 4.4 Application Endpoints (Unified FastAPI Monolith)

#### FR-API-01 — Health Endpoint
**Priority:** MUST

The application must expose `GET /health` returning HTTP 200 with a JSON body indicating service status. This endpoint must not require authentication and is used for service monitoring.

**Acceptance criteria:**
- `GET /health` returns `{"status": "ok"}` within 200 ms under normal operating conditions.

---

#### FR-API-02 — Login Endpoint
**Priority:** MUST

`POST /login` must validate the `rr / rr` credential and set a signed, HTTP-only session cookie on success.

**Acceptance criteria:**
- Valid credentials → HTTP 302 redirect to user-picker page + `Set-Cookie` header.
- Invalid credentials → HTTP 401.

---

#### FR-API-03 — Active Users Endpoint
**Priority:** MUST

`GET /users/active` must return the top-6 customer records from `active:users:top6` Redis key as a JSON array of objects with fields `customer_id`, `age`, and `prewarmed`.

**Acceptance criteria:**
- Returns exactly 6 records when the Redis key is populated.
- Returns an appropriate error (HTTP 503) if the Redis key is absent.

---

#### FR-API-04 — Recommendations Endpoint
**Priority:** MUST

`GET /recommendations/{customer_id}?age={age}&date={YYYY-MM-DD}&k=10` must invoke the five-stage online pipeline and return a JSON array of top-K articles with scores.

**Acceptance criteria:**
- `age` and `date` are accepted as query parameters and passed as pipeline features.
- `k` defaults to 10; values 1–50 are accepted.
- Response JSON includes at minimum `article_id` and `score` per item.
- A `degraded: true` flag appears in the response when any fallback was exercised.

---

### 4.5 ML Inference

#### FR-ML-01 — Two-Tower User Embedding
**Priority:** MUST

The system must serve a user embedding model via a dedicated SageMaker Endpoint (`two-tower-user-tower`). The endpoint must accept a user feature vector and return a 256-dimensional float32 embedding.

**Acceptance criteria:**
- The endpoint returns a correctly shaped `[1, 256]` tensor for any valid user feature input.
- The endpoint is reachable from the orchestrator Lambda via the AWS SDK without additional network configuration.

---

#### FR-ML-02 — FAISS Approximate Nearest Neighbour Search
**Priority:** MUST

The system must serve a FAISS index as a Lambda function. The Lambda must load the `.index` file from S3 at cold start (via `/tmp`) and return the top-100 nearest item embeddings for a query embedding.

**Acceptance criteria:**
- Warm Lambda search latency is under 20 ms p95 for top-100 search.
- The Lambda reads the FAISS index version from the `FAISS_INDEX_VERSION` environment variable, enabling a zero-downtime index swap by updating only the environment variable and uploading a new index to S3.

---

#### FR-ML-03 — CatBoost Ranking
**Priority:** MUST

The system must serve a CatBoost ranking model via a dedicated SageMaker Endpoint (`catboost-ranker`). The endpoint must accept a batch of feature vectors (one per candidate article) and return a predicted purchase probability per candidate.

**Acceptance criteria:**
- Batch input (up to 100 candidates) returns the same number of scores.
- Score order is descending (highest probability first) as returned by the endpoint or sorted by the orchestrator.

---

#### FR-ML-04 — FAISS Zero-Downtime Index Swap
**Priority:** SHOULD

Updating the FAISS index must not require Lambda redeployment or incur downtime. Warm Lambda containers continue serving the previous index version; new containers load the new version.

**Acceptance criteria:**
- Uploading a new `.index` file to S3 and updating the `FAISS_INDEX_VERSION` env var causes new cold-starts to load the new index.
- No 5xx errors are emitted during the transition window.

---

### 4.6 Offline Batch Pipelines

#### FR-BATCH-01 — Data Preparation Pipeline
**Priority:** MUST

A Glue PySpark job must read raw H&M CSV files from the S3 `raw/` zone, validate schema, deduplicate records, and write partitioned Parquet to the `clean/` zone.

**Acceptance criteria:**
- Output Parquet is partitioned by month for `transactions`; unpartitioned for `articles` and `customers`.
- The job is idempotent: re-running it does not produce duplicate records.
- Schema validation failures abort the job and send an SNS alarm.

---

#### FR-BATCH-02 — Feature Engineering Pipeline
**Priority:** MUST

A Glue PySpark job must derive user and item features from the `clean/` zone and write them to the `features/` zone.

Required user features: purchase frequency, average price paid, top-N purchased categories, recency (days since last transaction).
Required item features: popularity score (transaction count), days since first sold.

**Acceptance criteria:**
- Feature output is written to `features/users/customer_id={cid}/` and `features/items/article_id={aid}/` as Parquet.
- All user features are computable from `customers.csv` and `transactions_train.csv` alone.

---

#### FR-BATCH-03 — Cache Warm-Up Pipeline
**Priority:** MUST

A Glue PySpark job (daily, 03:00 UTC) must write the following Redis keys from feature outputs:
- `popular:items:top100` (top-100 items by transaction count, 24 h TTL)
- `popular:items:by_category:{cat}` for every category (24 h TTL)
- `seen:{customer_id}` (set of purchased article IDs, no TTL; rebuilt nightly)
- `active:users:top6` (JSON list of the 6 most-active customers, 24 h TTL)

**Acceptance criteria:**
- All four key families are present and readable in Redis by 04:00 UTC.
- `seen:{customer_id}` entries reflect transactions up to the previous day.

---

#### FR-BATCH-04 — ML Training Pipeline (SageMaker Pipelines)
**Priority:** MUST

The ML pipeline must execute the following DAG:
1. SageMaker Processing job: build training tables from `features/`.
2. SageMaker Training job: train the Two-Tower model.
3. SageMaker Training job: train the CatBoost model (parallel with step 2).
4. SageMaker Processing job: evaluate both models on a holdout set.
5. Conditional gate: proceed only if `recall@100 > baseline` and `AUC > baseline`.
6. Register models in SageMaker Model Registry.
7. Manual approval gate.
8. SageMaker Batch Transform: compute item embeddings.
9. Lambda: build new FAISS index and upload to S3.
10. Update SageMaker endpoint production variants with 10% canary traffic.

**Acceptance criteria:**
- A failed evaluation gate terminates the pipeline and emits an SNS alarm.
- Model registration is blocked if the approval gate is not satisfied.
- The canary traffic split is 10% new / 90% current at deploy time.

---

#### FR-BATCH-05 — Drift Monitoring
**Priority:** SHOULD

SageMaker Model Monitor must run daily (04:00 UTC) to compare live inference data against the training baseline for both the user-tower and CatBoost endpoints.

**Acceptance criteria:**
- Model Monitor baseline is established on first model deployment.
- A drift score above threshold triggers an SNS alarm.

---

### 4.7 Observability & Alerting

#### FR-OBS-01 — CloudWatch Alarms
**Priority:** MUST

The following CloudWatch alarms must be configured to send notifications via SNS:

| Alarm                                         | Threshold                    |
|-----------------------------------------------|------------------------------|
| API p95 latency > 500 ms                      | 5-minute sustained           |
| Recommendation 5xx rate > 1%                  | 5-minute sustained           |
| Any fallback counter > 1% of requests         | 5-minute sustained           |
| Cache hit ratio < 30%                         | 30-minute sustained          |
| SageMaker endpoint 5xx > 0                    | Immediate                    |
| Step Functions execution failure              | Immediate                    |
| Drift score above baseline                    | Immediate                    |
| Pre-warm DLQ depth > 0                        | Immediate                    |

**Acceptance criteria:**
- Each alarm fires within the stated window when its threshold is breached in a test scenario.
- Alarm notifications are delivered to the configured SNS topic.

---

#### FR-OBS-02 — Custom Business Metrics
**Priority:** MUST

The orchestrator Lambda must emit the following custom CloudWatch metrics to the `Recommendation` namespace:

| Metric                                 | Purpose                                   |
|----------------------------------------|-------------------------------------------|
| `recommendation.cache.hit_ratio`       | Caching effectiveness                     |
| `recommendation.fallback.{component}`  | Per-component fallback activation count   |
| `recommendation.diversity.applied`     | Count of diversity reorder activations    |
| `recommendation.coldstart.user`        | Cold-start requests served by fallback    |
| `pipeline.stage.{stage}.latency_ms`    | Per-stage latency histogram               |

**Acceptance criteria:**
- All five metric families are visible in CloudWatch within 60 seconds of a relevant event.

---

#### FR-OBS-03 — Distributed Tracing
**Priority:** SHOULD

X-Ray tracing must be enabled end-to-end: API Gateway → Backend Lambda → SageMaker endpoints → FAISS Lambda. The X-Ray service map must render a coherent dependency graph of all components.

**Acceptance criteria:**
- A single recommendation request produces a connected X-Ray trace covering API Gateway, the orchestrator Lambda, and both SageMaker endpoints.

---

## 5. Non-Functional Requirements

> Each NFR carries an ID in the form `NFR-{area}-{number}`.

---

### 5.1 Performance

#### NFR-PERF-01 — Cache-Hit Latency
**Priority:** MUST

End-to-end recommendation latency for a cache-hit request must be below 15 ms at the 95th percentile, measured from API Gateway request receipt to response delivery.

---

#### NFR-PERF-02 — Cache-Miss Latency (Warm Pipeline)
**Priority:** MUST

End-to-end recommendation latency for a cache-miss request with all downstream components warm must be below 250 ms at the 95th percentile.

**Per-stage budget (p95):**

| Stage                              | Budget |
|------------------------------------|--------|
| Stage 0: cache check               | 12 ms  |
| Stage 1: feature fetch (Redis)     | 5 ms   |
| Stage 1: SageMaker user-tower      | 80 ms  |
| Stage 1: FAISS Lambda (warm)       | 15 ms  |
| Stage 2: filter (Redis SMEMBERS)   | 5 ms   |
| Stage 3: item features bulk read   | 5 ms   |
| Stage 3: SageMaker CatBoost        | 70 ms  |
| Stage 4: diversity reorder         | 2 ms   |
| Cache write + serialise response   | 6 ms   |
| **Total budget**                   | **190 ms** |

---

#### NFR-PERF-03 — FAISS Lambda Warm Latency
**Priority:** MUST

The FAISS Lambda must return top-100 results within 20 ms p95 when the Lambda execution environment is warm.

---

#### NFR-PERF-04 — Cold-Start Tolerance
**Priority:** SHOULD

Lambda cold starts on the FAISS Lambda (index load from S3 via `/tmp`) must complete within 600 ms. A cold-start recommendation response must stay below 750 ms p95.

---

#### NFR-PERF-05 — Diversity Reorder Overhead
**Priority:** MUST

Stage 4 diversity reorder must add no more than 5 ms to the end-to-end latency for a 100-candidate input set.

---

### 5.2 Availability & Reliability

#### NFR-REL-01 — Service Availability
**Priority:** MUST

The system must maintain 99.5% availability during active learning sessions (defined as any period when SageMaker endpoints are running).

---

#### NFR-REL-02 — Graceful Degradation
**Priority:** MUST

No single dependency failure must result in a complete service outage. The circuit breaker and fallback chain defined in FR-PIPE-08 must guarantee that a recommendation is returned (potentially popular-items fallback) for any user under any single-component failure scenario.

---

#### NFR-REL-03 — Pre-Warm Reliability
**Priority:** MUST

The cache pre-warming pipeline must successfully pre-warm all three target customers on ≥ 95% of calendar days. Failures must be surfaced via the DLQ alarm (FR-OBS-01) within 5 minutes.

---

#### NFR-REL-04 — At-Least-Once Delivery Safety
**Priority:** MUST

The SQS pre-warm consumer must be idempotent. Duplicate delivery of a pre-warm message must produce no observable side effect beyond the first processing (i.e., no double cache write, no extra SageMaker invocations).

---

### 5.3 Scalability

#### NFR-SCALE-01 — Horizontal Scaling
**Priority:** SHOULD

All stateless components (ECS Fargate application, FAISS Lambda) must scale horizontally without architectural changes or code modifications.

| Component                  | Min | Max  | Trigger               |
|----------------------------|-----|------|-----------------------|
| SageMaker user-tower       | 1   | 4    | 1,000 invocations/min |
| SageMaker CatBoost         | 1   | 4    | 1,000 invocations/min |
| FAISS Lambda               | 0   | 50   | Concurrency cap       |
| ECS Fargate (application)  | 1   | 4    | CPU 70% or memory 80% |
| Glue jobs                  | 2   | 10   | DPU auto-scaling      |

---

#### NFR-SCALE-02 — Architecture Parity (Dev vs. Full Dataset)
**Priority:** MUST

The architecture must be identical for the 10K-user development sample and the full 1.37M-user H&M dataset. Scaling must require only instance sizing and data path changes — no structural modifications.

---

### 5.4 Security

#### NFR-SEC-01 — Encryption at Rest
**Priority:** MUST

All persistent data stores must encrypt data at rest:
- S3: SSE-KMS
- ElastiCache: encryption at rest enabled
- ECR: image signing enabled

---

#### NFR-SEC-02 — Encryption in Transit
**Priority:** MUST

All inter-service communication must use TLS 1.2 or higher. This includes CloudFront, API Gateway, all internal AWS service calls, and communication between Lambda/Fargate and SageMaker/Redis.

---

#### NFR-SEC-03 — Least-Privilege IAM
**Priority:** MUST

Each Lambda function and ECS task must operate under a dedicated IAM role with only the permissions required for its function. No role may hold `*` resource actions.

---

#### NFR-SEC-04 — Secret Management
**Priority:** MUST

No secrets (API keys, credentials, tokens) may appear in environment variables, source code, or Git history. All secrets must be stored in SSM Parameter Store and read at runtime.

---

#### NFR-SEC-05 — VPC Isolation
**Priority:** MUST

ElastiCache Redis and SageMaker Endpoints must be deployed in VPC private subnets and must not be directly accessible from the public internet.

---

#### NFR-SEC-06 — Authentication (V1 Scope)
**Priority:** MUST

V1 uses `rr / rr` placeholder credentials. This is a documented production gap. The implementation must not expose any real customer data or allow unauthenticated API access to the recommendations endpoint from outside the frontend.

**Documented production path:** API Gateway JWT authorizer + Cognito User Pool (v1.1).

---

#### NFR-SEC-07 — Audit Logging
**Priority:** SHOULD

AWS CloudTrail must be enabled for the account region. Trail logs must be retained for a minimum of 30 days.

---

### 5.5 Observability

#### NFR-OBS-01 — Log Retention
**Priority:** MUST

All Lambda and ECS container logs written to CloudWatch Logs must be retained for a minimum of 14 days.

---

#### NFR-OBS-02 — Structured Logging
**Priority:** SHOULD

All application logs must be emitted as structured JSON (key-value pairs) to enable CloudWatch Insights queries without regex parsing.

**Required fields per log entry:** `timestamp`, `level`, `component`, `request_id`, `customer_id` (where applicable), `message`.

---

#### NFR-OBS-03 — Latency Visibility
**Priority:** MUST

Per-stage latency histograms (FR-OBS-02 `pipeline.stage.{stage}.latency_ms`) must provide sufficient granularity to diagnose which stage caused a p95 budget breach.

---

#### NFR-OBS-04 — Cache Hit Ratio Dashboard
**Priority:** SHOULD

A CloudWatch dashboard must display `recommendation.cache.hit_ratio` over time to confirm that the pre-warming strategy is effective and the cache is functioning as designed.

---

### 5.6 Cost

#### NFR-COST-01 — Active Session Monthly Cost
**Priority:** MUST

Total AWS spend during a calendar month with active learning sessions (SageMaker endpoints running approximately 6 hours per day on weekdays) must not exceed **$50**.

---

#### NFR-COST-02 — One-Command Infrastructure Control
**Priority:** MUST

A single `terraform destroy` command must tear down all billable resources for cost control between development sessions.

**Acceptance criteria:**
- `terraform destroy` removes all compute resources (Fargate, SageMaker endpoints, Lambda functions).
- S3 bucket contents and ECR images are retained for data persistence but incur minimal storage costs.

---

#### NFR-COST-03 — No Over-Provisioning
**Priority:** SHOULD

All components must use the smallest instance type that satisfies the performance NFRs. Upsizing must be documented with a justification.

---

### 5.7 Maintainability & Operability

#### NFR-MAINT-01 — Unified Codebase
**Priority:** MUST

The application must ship from a single Dockerfile and a single FastAPI codebase. Runtime behaviour differences must be controlled exclusively by environment variables, not conditional code paths.

---

#### NFR-MAINT-02 — Environment Parity
**Priority:** MUST

The application's business logic must be identical across local development (LocalStack + local Redis), `dev` (AWS dev prefix), and `prod` (AWS prod Terraform workspace). The only permitted differences are environment variable values.

---

#### NFR-MAINT-03 — Canary Rollback SLA
**Priority:** SHOULD

An auto-rollback triggered by a CloudWatch alarm on the canary variant must complete (revert traffic to 100% on the previous model version) within 5 minutes of the alarm firing.

---

#### NFR-MAINT-04 — Infrastructure as Code Coverage
**Priority:** MUST

100% of AWS resources must be managed by Terraform. No resources may be created through the AWS Console that are not reflected in the Terraform state. Manual console changes are a documented production gap.

---

#### NFR-MAINT-05 — Diversity Weights Without Redeployment
**Priority:** MUST

The diversity algorithm weights (`w1`, `w2`, `w3`) must be configurable via Lambda environment variables. Changing them must not require a code deployment.

---

### 5.8 Portability & Deployability

#### NFR-PORT-01 — Local Development with Zero AWS Cost
**Priority:** MUST

Developers must be able to run the complete system locally (LocalStack + local Redis + local SageMaker SDK) with zero AWS spend. No component may hard-code AWS endpoints or region-specific behaviour.

---

#### NFR-PORT-02 — Single-Command Deployment
**Priority:** MUST

A fully functional v1 environment must be deployable to AWS with a single `terraform apply` command after first-time credential setup. Teardown must be achievable with a single `terraform destroy`.

---

#### NFR-PORT-03 — CI/CD Automation
**Priority:** MUST

Every push to the `main` branch must trigger the full CI/CD pipeline (lint, unit tests, integration tests, Docker build, ECR push, Terraform plan). No deployment to AWS may occur without passing all pipeline stages.

---

## 6. Constraints

| ID       | Constraint                                                                                                 |
|----------|------------------------------------------------------------------------------------------------------------|
| CON-01   | The system must deploy entirely within a single AWS region (us-east-1) in v1.                              |
| CON-02   | The development dataset is limited to 10K users, 5K articles, 100K transactions to control training cost.  |
| CON-03   | No real user authentication infrastructure (Cognito, OAuth) is required for v1. `rr/rr` is the only valid credential. |
| CON-04   | ECS Fargate tasks must use 0.5 vCPU / 1.0 GB sizing for the unified application. SageMaker endpoints must use `ml.t3.medium` instances unless a performance NFR cannot be satisfied at that sizing. |
| CON-05   | The FAISS index must fit within Lambda's 10 GB memory limit (estimated < 300 MB for the full H&M dataset). |
| CON-07   | The system must use only AWS-native services. No third-party SaaS (Pinecone, Datadog, etc.) may be introduced in v1. |
| CON-08   | All infrastructure must be managed by Terraform. No CDK, SAM, or CloudFormation stacks are permitted alongside Terraform state. |

---

## 7. Assumptions

| ID       | Assumption                                                                                                   |
|----------|--------------------------------------------------------------------------------------------------------------|
| ASM-01   | The H&M dataset has been imported once into the S3 `raw/` zone. Re-importing is out of scope.               |
| ASM-02   | All customer identifiers in the H&M dataset are pre-hashed. No attempt to reverse-map identifiers is in scope. |
| ASM-03   | Traffic volume during active learning sessions will not exceed 60 RPS at the API Gateway stage throttle.    |
| ASM-04   | The FAISS `IndexFlatIP` index type is sufficient for the 5K-item development dataset. Migration to `IndexIVFFlat` for the full dataset is a configuration change only. |
| ASM-05   | Weekly model retraining cadence is sufficient for the recommendation quality objectives of v1.               |
| ASM-06   | A single Redis `cache.t3.micro` node provides sufficient memory and throughput for the development dataset at expected traffic levels. |
| ASM-07   | GitHub Actions free tier provides sufficient CI/CD compute minutes for the expected push frequency.         |
| ASM-08   | The unified FastAPI monolith architecture is acceptable for V1 learning objectives. Microservices decomposition is a documented V2 enhancement.         |

---

## 8. Out of Scope (V1)

The following items are explicitly deferred. They are documented here to prevent scope creep and to serve as an ordered backlog for v1.1.

| Item                              | Deferral Rationale                                                      | Target Version |
|-----------------------------------|-------------------------------------------------------------------------|----------------|
| `POST /events` endpoint           | Async event ingestion requires Kinesis Firehose architecture additions.  | v1.1           |
| Kinesis Firehose pipeline         | Depends on `POST /events`.                                               | v1.1           |
| SQS purchase-event consumer       | Depends on `POST /events`; enables real-time cache invalidation on purchase. | v1.1      |
| Engagement features (click/view)  | Requires the event pipeline to produce signals.                          | v1.1           |
| Cognito / real authentication     | Zero-cost `rr/rr` is sufficient for portfolio demonstration.             | v1.1           |
| LLM tag extraction                | Optional enrichment; separate HLD required.                              | v2             |
| RAG chatbot                       | Separate request path; separate HLD required.                            | v2             |
| Online learning / streaming retrain | Substantial complexity jump; weekly batch is sufficient.               | v2             |
| Date-scoped cache key             | Eliminates rare cross-midnight stale hit; low priority.                  | v2             |
| WAF on ALB or CloudFront          | Not required while access is restricted to portfolio reviewers.          | v2             |
| Multi-region active-active        | Cost/complexity not justified at this scale.                             | v3+            |
| Embedding cosine diversity (V2 reorder) | Needs embedding cache; swap-in compatible when ready.              | v2             |
| Private subnet + NAT Gateway      | ~$32/month; documented production hardening path.                        | Production     |
| Microservices decomposition       | Monolith is sufficient for v1; decomposition requires service mesh, inter-service auth. | v2+ |

---

## 9. Requirements Traceability Matrix

> Maps each requirement to the HLD section that addresses it, confirming full coverage.

| Requirement ID | Description (short)                         | HLD Section              |
|----------------|---------------------------------------------|--------------------------|
| FR-UI-01       | Login page (`rr/rr`)                        | §8.2, §6.2               |
| FR-UI-02       | User-picker page (6 cards)                  | §6.3                     |
| FR-UI-03       | Pre-warm badge on cards 1–3                 | §6.3, §12.4              |
| FR-UI-04       | HTMX partial recommendations fragment       | §6.1, §6.2               |
| FR-PIPE-01     | Five-stage online pipeline                  | §9.1                     |
| FR-PIPE-02     | Stage 0: cache check                        | §9.2                     |
| FR-PIPE-03     | Stage 1: retrieval                          | §9.3                     |
| FR-PIPE-04     | Stage 2: filter                             | §9.4                     |
| FR-PIPE-05     | Stage 3: CatBoost ranking                   | §9.5                     |
| FR-PIPE-06     | Stage 4: diversity reorder                  | §9.6                     |
| FR-PIPE-07     | Rate limiting (two layers)                  | §7.3                     |
| FR-PIPE-08     | Circuit breakers & fallbacks                | §9.8                     |
| FR-CACHE-01    | Recommendation result cache                 | §9.2, §10.3              |
| FR-CACHE-02    | Feature caches (user + item)                | §10.3, §10.4             |
| FR-CACHE-03    | SQS pre-warming work queue                  | §12.4                    |
| FR-CACHE-04    | Cold-start fallback Redis keys              | §9.4, §10.3              |
| FR-API-01      | `GET /health`                               | §8.2                     |
| FR-API-02      | `POST /login`                               | §8.2                     |
| FR-API-03      | `GET /users/active`                         | §8.2                     |
| FR-API-04      | `GET /recommendations/{id}`                 | §8.2, §9                 |
| FR-ML-01       | Two-tower SageMaker endpoint                | §11.1                    |
| FR-ML-02       | FAISS Lambda serving                        | §11.2                    |
| FR-ML-03       | CatBoost SageMaker endpoint                 | §11.3                    |
| FR-ML-04       | Zero-downtime FAISS index swap              | §11.2                    |
| FR-BATCH-01    | Data preparation Glue job                   | §12.2                    |
| FR-BATCH-02    | Feature engineering Glue job                | §12.2                    |
| FR-BATCH-03    | Cache warm-up Glue job                      | §12.2                    |
| FR-BATCH-04    | SageMaker ML training pipeline              | §12.3                    |
| FR-BATCH-05    | Drift monitoring                            | §13.3                    |
| FR-OBS-01      | CloudWatch alarms → SNS                     | §13.3                    |
| FR-OBS-02      | Custom business metrics                     | §13.3                    |
| FR-OBS-03      | X-Ray distributed tracing                  | §13.3                    |
| NFR-PERF-01    | Cache-hit latency < 15 ms p95              | §2.3, §9.7               |
| NFR-PERF-02    | Cache-miss latency < 250 ms p95            | §2.3, §9.7               |
| NFR-PERF-03    | FAISS warm latency < 20 ms p95             | §2.3, §11.2              |
| NFR-PERF-04    | Cold-start tolerance < 600 ms              | §11.2                    |
| NFR-PERF-05    | Diversity reorder < 5 ms                   | §9.6                     |
| NFR-REL-01     | 99.5% availability during active sessions  | §2.3                     |
| NFR-REL-02     | Graceful degradation under failure         | §9.8                     |
| NFR-REL-03     | Pre-warm reliability ≥ 95% of days         | §12.4                    |
| NFR-REL-04     | Idempotent SQS consumer                    | §12.4                    |
| NFR-SCALE-01   | Horizontal scaling                         | §13.4                    |
| NFR-SCALE-02   | Architecture parity dev vs. full dataset   | §3 (principles)          |
| NFR-SEC-01     | Encryption at rest                         | §13.2                    |
| NFR-SEC-02     | Encryption in transit (TLS 1.2+)           | §13.2                    |
| NFR-SEC-03     | Least-privilege IAM                        | §13.2                    |
| NFR-SEC-04     | Secrets in SSM Parameter Store             | §13.2                    |
| NFR-SEC-05     | VPC isolation for Redis + SageMaker        | §13.2                    |
| NFR-SEC-06     | Authentication scope (v1 gap documented)   | §13.2, §16.8             |
| NFR-SEC-07     | CloudTrail audit logging (30 days)         | §13.2                    |
| NFR-OBS-01     | Log retention (14 days)                    | §13.3                    |
| NFR-OBS-02     | Structured JSON logging                    | §13.3                    |
| NFR-OBS-03     | Per-stage latency histograms               | §13.3                    |
| NFR-OBS-04     | Cache hit ratio dashboard                  | §13.3                    |
| NFR-COST-01    | Active session cost < $60/month            | §15.1                    |
| NFR-COST-02    | One-command infrastructure teardown        | §2.1, §15               |
| NFR-COST-03    | No over-provisioning                       | §15.1                    |
| NFR-MAINT-01   | Unified codebase                           | §6.1, §8.2               |
| NFR-MAINT-02   | Environment parity (local / dev / prod)    | §14.3, CLAUDE.md         |
| NFR-MAINT-03   | Canary rollback within 5 minutes           | §14.4                    |
| NFR-MAINT-04   | 100% Terraform IaC coverage               | §14.1                    |
| NFR-MAINT-05   | Diversity weights via env var              | §9.6                     |
| NFR-PORT-01    | Local development at $0 AWS cost          | CLAUDE.md, §18.2         |
| NFR-PORT-02    | Single-command `terraform apply`           | §2.1                     |
| NFR-PORT-03    | CI/CD automation on every push             | §14.1                    |

---

*Document version controlled alongside source code. Changes to requirements must be reflected in `v1-hld.md` and vice versa.*
