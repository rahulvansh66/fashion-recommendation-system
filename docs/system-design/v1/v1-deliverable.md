Here's the complete v1 deliverable, organized by layer. After v1 ships, you'll have everything below — and *only* the items in Section 9 + Section 10 will remain unbuilt.

## 1. Unified Application Layer (Frontend + Backend on ECS Fargate)

| Item | Detail |
|---|---|
| **Architecture** | **Unified FastAPI monolith** on ECS Fargate — single application serving both frontend and backend |
| **Frontend tech stack** | **Jinja2** templates (server-rendered HTML) + **HTMX** (partial page updates) + **Tailwind CSS** (CDN, no build step) |
| **Backend framework** | **FastAPI** — serves both HTML templates and JSON API endpoints |
| **Why this stack** | Modern UX (HTMX) without SPA complexity; no Node.js build pipeline; production-proven server-rendered pattern; zero inter-service latency |
| Login page | `rr / rr` check, signed cookie session, Jinja2-rendered |
| User-picker page | **6 active-user cards**. Each card shows: `customer_id` and `age` (loaded from the test dataset via Redis), plus `current_date` (auto-computed on the server at every page load — *today*; not stored anywhere). Per-card "Show recommendations" button forwards all three values to the inference pipeline as direct features. First 3 cards show a `pre-warmed` badge (cache hit); last 3 are live (cache miss) — visible latency demo of the SQS pre-warm pattern |
| Recommendations display | HTMX partial page update — renders top-10 article cards in-place without full reload |
| API endpoints | `GET /health`, `GET /` (login), `POST /login`, `GET /picker` (user-picker), `GET /recommendations/{customer_id}?age={age}&date={YYYY-MM-DD}&k=10` (HTMX partial or full JSON) |
| **Deployment platform** | **ECS Fargate** — 0.5 vCPU / 1.0 GB, public subnet with public IP, desired count = 1, no cold starts |
| Container | Single Docker image; same image runs locally and on AWS; **uvicorn** ASGI server |
| CloudFront + ACM | TLS termination, edge caching of static assets |

## 2. API Gateway & Ingress (No ALB)

| Item | Detail |
|---|---|
| **API Gateway HTTP API** | Single HTTP API fronting the Fargate service (saves ~$16/mo vs. ALB) |
| **VPC Link + Cloud Map** | Routes from API Gateway to Fargate task via **service discovery** (no load balancer needed) |
| **Why no ALB** | Low-traffic v1 doesn't justify ~$16/mo; Cloud Map service discovery sufficient; HTTP API provides throttling + routing |
| Stage-level rate limiting | 60 RPS / burst 100 per stage (API Gateway throttling) |
| Application-level rate limit | Token bucket per `customer_id` in Redis (30 req/min default), implemented as FastAPI middleware |

## 3. Online serving pipeline (5 stages)

| Stage | What's live |
|---|---|
| Stage 0 — Cache check | Redis `reco:{customer_id}` GET, 12-h TTL |
| Stage 1 — Retrieve | User feature fetch (Redis → S3 fallback) → SageMaker user-tower endpoint → FAISS Lambda |
| Stage 2 — Filter | Redis `seen:{customer_id}` set; cold-start fallback to `popular:items:top100` |
| Stage 3 — Rank | Bulk item-feature read → SageMaker CatBoost endpoint |
| Stage 4 — Order | Categorical + price-bucket diversity reorder (positions 1–4 raw, 5–6 diverse, 7–10 raw) |
| Cache write | Redis SETEX 12 h |
| Circuit breakers | `pybreaker` on Redis / user-tower / FAISS / CatBoost with documented fallbacks for each |

## 4. ML inference (managed)

| Item | Detail |
|---|---|
| SageMaker endpoint — Two-Tower user-tower | `ml.t3.medium`, 256-dim embedding output |
| SageMaker endpoint — CatBoost ranker | `ml.t3.medium`, batch-scored candidates |
| FAISS Lambda | 2 GB memory, S3-backed `.index` file, top-100 retrieval |
| Production-variant scaffolding | Canary / A/B configurable on both endpoints |
| Model Monitor | Data-quality and model-drift baseline configured |

## 5. Data layer (the data lake + cache)

| Item | Detail |
|---|---|
| S3 zones | `raw/`, `clean/`, `features/users/`, `features/items/`, `features/interactions/`, `models/`, `embeddings/`, `indices/` |
| S3 reserved (empty) | `enriched/` (LLM tags, future), `events/` (v1.1) |
| ElastiCache Redis | `cache.t3.micro` |
| Redis keys | `reco:{cid}` (live OR pre-warmed), `user:{cid}:features`, `item:{aid}:features`, `seen:{cid}` (rebuilt nightly), `popular:items:top100`, `popular:items:by_category:{cat}`, `ratelimit:{cid}`, `embedding:user:{cid}` (fallback), `active:users:top6` (entries `{customer_id, age, prewarmed}` — drives picker cards + pre-warm; `current_date` set at render time), `prewarm:done:{cid}:{date}` (pre-warm idempotency) |

## 6. Offline batch pipelines

| Item | Detail |
|---|---|
| EventBridge cron | Triggers Step Functions on schedule (data + features weekly; cache warm-up daily 03:00 UTC; cache pre-warm daily 05:00 UTC) |
| Step Functions — data + feature pipeline | Orchestrates Glue jobs end-to-end |
| AWS Glue (PySpark) | Clean, feature-engineer (users, items, interactions), build popular-items keys, populate Redis, write `active:users:top6` list with `{customer_id, age, prewarmed}` entries (drives both user-picker cards and pre-warm; `current_date` is added by the frontend at render time) |
| SageMaker Pipelines — ML pipeline | Two-Tower training → CatBoost training → FAISS index build → Model Registry registration |
| SageMaker Training Jobs | On `ml.m5.large` spot (weekly) |
| SageMaker Model Registry | Approval-gated promotion |
| **Cache pre-warming work queue** | Daily 05:00 UTC: producer Lambda reads top-3 of `active:users:top6`, stamps each message with `run_date = today (UTC)` plus `age`, and sends to SQS Standard queue → consumer Lambda runs the full 5-stage pipeline with `(customer_id, age, current_date=run_date)` → writes `reco:{cid}` to Redis with 12 h TTL. Idempotent via `prewarm:done:{cid}:{run_date}` SETNX. DLQ + alarm on failure. Reserved concurrency = 5. |

## 7. Observability

| Item | Detail |
|---|---|
| CloudWatch metrics | Per-stage latency, fallback counters (`recommendation.fallback.{component}`), Fargate / SageMaker / Lambda built-ins |
| CloudWatch alarms + SNS | Pipeline failures, breaker trips, drift, error rate, p95 latency |
| AWS X-Ray | Distributed tracing across Fargate → SageMaker → FAISS → Redis |
| Structured JSON logs | All Fargate / Lambda output |
| (Optional) CloudWatch dashboard | Single pane for the full pipeline |

## 8. Security & ops

| Item | Detail |
|---|---|
| IAM least-privilege roles | Per Fargate task / Lambda / Glue / SageMaker |
| KMS encryption at rest | S3, Redis, SageMaker artifacts |
| TLS in transit | Everywhere (CloudFront / API Gateway / endpoints) |
| Secrets in SSM Parameter Store | No secrets in env vars or code |
| `rr/rr` placeholder auth | Documented as production gap |

## 9. CI/CD pipeline

| Item | Detail |
|---|---|
| GitHub Actions | Lint, test, build, push to ECR, terraform plan/apply, ECS update |
| Conditional ML pipeline | Triggers SageMaker Pipelines only when `models/` or `feature_pipeline/` paths change |
| Manual approval gates | (a) terraform apply to prod, (b) Model Registry approval before canary |
| Canary deployment | SageMaker production variants 10% → 50% → 100% |
| Auto-rollback | CloudWatch alarm → variant weights reset |
| Dual environments | `dev` (auto-apply on `main`) + `prod` (separate Terraform workspace, tag-gated) |
| LocalStack-driven local-dev workflow | Same code path as cloud |

## 10. Infrastructure as Code

| Item | Detail |
|---|---|
| Terraform | All AWS resources, modular |
| One-command apply / destroy | Fargate module separately destroyable for cost control |
| Public-subnet variant for Fargate | Documented swap path to private subnet + VPC endpoints |

## 11. Documentation

| Item | Detail |
|---|---|
| `v1-hld.md` | Complete high-level design |
| `v1-requirements.md` | Functional and non-functional requirements |
| `v1-deliverable.md` | This document |
| `project-description.md`, `infrastructure-layer.md`, `schema-info.md`, `project-structure.md` | Pre-existing context |
| Implementation plan | (next step from v1 design — to be written) |
| Cost-tracking notes | Actual vs. estimated for retrospective |

---

## What you will **NOT** have after v1

- **All of v1.1**: `POST /events`, Kinesis Firehose, SQS purchase queue, consumer Lambda, live cache invalidation, engagement features, CatBoost retraining on click signals
- **All of Section 17 (Future Enhancements)**: LLM tag extraction, RAG chatbot, embedding-cosine diversity (V2), online/streaming retrain, Cognito + JWT, private-subnet + VPC endpoints, multi-region, semantic caching, WAF managed rules, microservices decomposition, React/Vue/Svelte SPA

---

## What "v1 release" means concretely

When v1 ships you'll be able to:

1. **Open the web application** (FastAPI + Jinja2 + HTMX + Tailwind on ECS Fargate) during a learning session, log in as `rr/rr`, see 6 active-user cards rendered with Tailwind styling. Each card shows `customer_id` and `age` (loaded from the test dataset) plus `current_date` (auto-computed at page load — *today*). Click any card's button to trigger an HTMX request that loads top-10 recommendations inline (partial page update, no full reload). The card's three values flow as direct inputs to the inference pipeline.
2. **Visibly demonstrate the SQS pre-warm pattern:** clicking any of the **first 3 cards** returns recommendations in ~15 ms (cache hit, pre-warmed by the nightly SQS work queue) with **instant HTMX swap**. Clicking any of the **last 3 cards** takes ~190 ms (cache miss, runs the full pipeline) with **visible loading state**. The latency difference is a live talking point about idempotent consumers, DLQ, and work-queue patterns.
3. **Demonstrate production-grade server-rendered architecture:** unified FastAPI monolith on ECS Fargate (no cold starts), API Gateway HTTP API + VPC Link + Cloud Map (no ALB), modern UX with HTMX (no SPA complexity), all deployed via Terraform.
4. Run `terraform apply` to spin up the full stack (Fargate + SageMaker endpoints) for ~$45–50/mo of active use (6h/day weekdays), and `terraform destroy` for the same modules between sessions to drop to minimal idle cost.
5. Push a code change to GitHub → CI runs lint/test/build/IaC plan → manual approve → CD deploys Fargate service / ML pipeline as needed → canary deployment for ML model changes → auto-rollback on alarm.
6. Trigger weekly retraining via EventBridge → Step Functions → Glue → SageMaker Pipelines → Model Registry → manual approval → canary deploy.
7. Walk an interviewer through every diagram in `v1-hld.md` and point at the live AWS resource backing each box. Explain the architectural tradeoffs: **why FastAPI monolith over Lambda microservices, why HTMX over React SPA, why HTTP API + Cloud Map over ALB.**

That is the complete v1 deliverable.
