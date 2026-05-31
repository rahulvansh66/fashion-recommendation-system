Here's the complete v1 deliverable, organized by layer. After v1 ships, you'll have everything below — and *only* the items in Section 10 + Section 19 will remain unbuilt.

## 1. Frontend layer (live, user-facing)

| Item | Detail |
|---|---|
| FastAPI + Jinja2 + HTMX + Tailwind app | Server-rendered, no SPA, single Docker image |
| Login page | `rr / rr` check, signed cookie session |
| User-picker page | **6 active-user cards**. Each card shows: `customer_id` and `age` (loaded from the test dataset via Redis), plus `current_date` (auto-computed on the server at every page load — *today*; not stored anywhere). Per-card "Show recommendations" button forwards all three values to the inference pipeline as direct features. First 3 cards show a `pre-warmed` badge (cache hit); last 3 are live (cache miss) — visible latency demo of the SQS pre-warm pattern |
| Recommendations page | Renders top-10 article cards from backend API; HTMX swap from the picker page button |
| Health endpoint | `/health` |
| **Identical card UI on both deploy targets** | The 6-card user-picker page and the recommendations page render identically on ECS Fargate and Lambda + LWA — same Docker image, same Jinja templates, same HTMX behavior; only the deploy target differs |
| **Two simultaneous deployments** of the same image | ECS Fargate (primary, public subnet, public IP) + Lambda + LWA (always-on demo) |
| CloudFront + ACM | TLS termination, edge caching of static assets |

## 2. API gateway & edge

| Item | Detail |
|---|---|
| 3 separate API Gateway HTTP APIs | `frontend-fargate`, `frontend-lambda`, `api` |
| VPC Link + Cloud Map | Routes from API Gateway to Fargate task |
| Stage-level rate limiting | 60 RPS / burst 100 per stage |
| Application-level rate limit | Token bucket per `customer_id` in Redis (30 req/min default) |

## 3. Backend API

| Item | Detail |
|---|---|
| Lambda + AWS Lambda Web Adapter (FastAPI orchestrator) | Container packaging; same image as locally |
| Endpoints | `GET /health`, `GET /recommendations/{customer_id}?age={age}&date={YYYY-MM-DD}&k=10` (forwards card inputs to inference pipeline), `POST /login`, `GET /users/active` (returns top-6 `{customer_id, age, prewarmed}`) |
| Reserved concurrency | 100 (cost guardrail) |

## 4. Online serving pipeline (5 stages)

| Stage | What's live |
|---|---|
| Stage 0 — Cache check | Redis `reco:{customer_id}` GET, 12-h TTL |
| Stage 1 — Retrieve | User feature fetch (Redis → S3 fallback) → SageMaker user-tower endpoint → FAISS Lambda |
| Stage 2 — Filter | Redis `seen:{customer_id}` set; cold-start fallback to `popular:items:top100` |
| Stage 3 — Rank | Bulk item-feature read → SageMaker CatBoost endpoint |
| Stage 4 — Order | Categorical + price-bucket diversity reorder (positions 1–4 raw, 5–6 diverse, 7–10 raw) |
| Cache write | Redis SETEX 12 h |
| Circuit breakers | `pybreaker` on Redis / user-tower / FAISS / CatBoost with documented fallbacks for each |

## 5. ML inference (managed)

| Item | Detail |
|---|---|
| SageMaker endpoint — Two-Tower user-tower | `ml.t3.medium`, 256-dim embedding output |
| SageMaker endpoint — CatBoost ranker | `ml.t3.medium`, batch-scored candidates |
| FAISS Lambda | 10 GB memory, S3-backed `.index` file, top-100 retrieval |
| Production-variant scaffolding | Canary / A/B configurable on both endpoints |
| Model Monitor | Data-quality and model-drift baseline configured |

## 6. Data layer (the data lake + cache)

| Item | Detail |
|---|---|
| S3 zones | `raw/`, `clean/`, `features/users/`, `features/items/`, `features/interactions/`, `models/`, `embeddings/`, `indices/` |
| S3 reserved (empty) | `enriched/` (LLM tags, future), `events/` (v1.1) |
| ElastiCache Redis | `cache.t3.micro` |
| Redis keys | `reco:{cid}` (live OR pre-warmed), `user:{cid}:features`, `item:{aid}:features`, `seen:{cid}` (rebuilt nightly), `popular:items:top100`, `popular:items:by_category:{cat}`, `ratelimit:{cid}`, `embedding:user:{cid}` (fallback), `active:users:top6` (entries `{customer_id, age, prewarmed}` — drives picker cards + pre-warm; `current_date` set at render time), `prewarm:done:{cid}:{date}` (pre-warm idempotency) |

## 7. Offline batch pipelines

| Item | Detail |
|---|---|
| EventBridge cron | Triggers Step Functions on schedule (data + features weekly; cache warm-up daily 03:00 UTC; cache pre-warm daily 05:00 UTC) |
| Step Functions — data + feature pipeline | Orchestrates Glue jobs end-to-end |
| AWS Glue (PySpark) | Clean, feature-engineer (users, items, interactions), build popular-items keys, populate Redis, write `active:users:top6` list with `{customer_id, age, prewarmed}` entries (drives both user-picker cards and pre-warm; `current_date` is added by the frontend at render time) |
| SageMaker Pipelines — ML pipeline | Two-Tower training → CatBoost training → FAISS index build → Model Registry registration |
| SageMaker Training Jobs | On `ml.m5.large` spot (weekly) |
| SageMaker Model Registry | Approval-gated promotion |
| **Cache pre-warming work queue** | Daily 05:00 UTC: producer Lambda reads top-3 of `active:users:top6`, stamps each message with `run_date = today (UTC)` plus `age`, and sends to SQS Standard queue → consumer Lambda runs the full 5-stage pipeline with `(customer_id, age, current_date=run_date)` → writes `reco:{cid}` to Redis with 12 h TTL. Idempotent via `prewarm:done:{cid}:{run_date}` SETNX. DLQ + alarm on failure. Reserved concurrency = 5. |

## 8. Observability

| Item | Detail |
|---|---|
| CloudWatch metrics | Per-stage latency, fallback counters (`recommendation.fallback.{component}`), Lambda / SageMaker / Fargate built-ins |
| CloudWatch alarms + SNS | Pipeline failures, breaker trips, drift, error rate, p95 latency |
| AWS X-Ray | Distributed tracing across API Lambda → SageMaker → FAISS → Redis |
| Structured JSON logs | All Lambda / Fargate output |
| (Optional) CloudWatch dashboard | Single pane for the full pipeline |

## 9. Security & ops

| Item | Detail |
|---|---|
| IAM least-privilege roles | Per Lambda / Fargate task / Glue / SageMaker |
| KMS encryption at rest | S3, Redis, SageMaker artifacts |
| TLS in transit | Everywhere (CloudFront / API Gateway / endpoints) |
| Secrets in SSM Parameter Store | No secrets in env vars or code |
| `rr/rr` placeholder auth | Documented as production gap |

## 10. CI/CD pipeline

| Item | Detail |
|---|---|
| GitHub Actions | Lint, test, build, push to ECR, terraform plan/apply, Lambda/ECS update |
| Conditional ML pipeline | Triggers SageMaker Pipelines only when `models/` or `feature_pipeline/` paths change |
| Manual approval gates | (a) terraform apply to prod, (b) Model Registry approval before canary |
| Canary deployment | SageMaker production variants 10% → 50% → 100% |
| Auto-rollback | CloudWatch alarm → variant weights reset |
| Dual environments | `dev` (auto-apply on `main`) + `prod` (separate Terraform workspace, tag-gated) |
| LocalStack-driven local-dev workflow | Same code path as cloud |

## 11. Infrastructure as Code

| Item | Detail |
|---|---|
| Terraform | All AWS resources, modular |
| One-command apply / destroy | Fargate module separately destroyable for cost control |
| Public-subnet variant for Fargate | Documented swap path to private subnet + VPC endpoints |

## 12. Documentation

| Item | Detail |
|---|---|
| `hld.md` | This document |
| `project-description.md`, `infrastructure-layer.md`, `schema-info.md`, `project-structure.md` | Pre-existing context |
| Implementation plan | (next step from v1 design — to be written) |
| Cost-tracking notes | Actual vs. estimated for retrospective |

---

## What you will **NOT** have after v1

- **All of v1.1**: `POST /events`, Kinesis Firehose, SQS purchase queue, consumer Lambda, live cache invalidation, engagement features, CatBoost retraining on click signals
- **All of Section 19 (Future Enhancements)**: LLM tag extraction, RAG chatbot, embedding-cosine diversity (V2), online/streaming retrain, Cognito + JWT, private-subnet + VPC endpoints, multi-region, semantic caching, WAF managed rules

---

## What "v1 release" means concretely

When v1 ships you'll be able to:

1. Open the `frontend-fargate.<domain>` URL during a learning session, log in as `rr/rr`, see 6 active-user cards. Each card shows `customer_id` and `age` (loaded from the test dataset) plus `current_date` (auto-computed at page load — *today*). Click any card's button to see live top-10 recommendations served by the full 5-stage pipeline (cache → retrieve → filter → rank → order). The card's three values flow as direct inputs to the inference pipeline.
2. Open the `frontend-lambda.<domain>` URL **at any time** (including months later, with everything else torn down) and demo the same flow at $0 idle cost — this is the resume-shareable URL. The card UI is identical to the Fargate variant; same Docker image, same Jinja templates.
3. Visibly demonstrate the SQS pre-warm pattern: clicking any of the **first 3 cards** returns recommendations in ~15 ms (cache hit, pre-warmed by the nightly SQS work queue). Clicking any of the **last 3 cards** takes ~190 ms (cache miss, runs the full pipeline). The latency difference is a live talking point about idempotent consumers, DLQ, and work-queue patterns.
4. Run `terraform apply` to spin up the heavy stuff (Fargate + SageMaker endpoints) for ~$53/mo of active use, and `terraform destroy` for the same modules between sessions to drop to ~$3/mo idle.
5. Push a code change to GitHub → CI runs lint/test/build/IaC plan → manual approve → CD deploys frontend / backend / ML pipeline as needed → canary deployment for ML model changes → auto-rollback on alarm.
6. Trigger weekly retraining via EventBridge → Step Functions → Glue → SageMaker Pipelines → Model Registry → manual approval → canary deploy.
7. Walk an interviewer through every diagram in `hld.md` and point at the live AWS resource backing each box.

That is the complete v1 deliverable.