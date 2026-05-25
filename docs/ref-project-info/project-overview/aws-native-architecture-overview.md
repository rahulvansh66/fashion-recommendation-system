---
⚠️ **REFERENCE PROJECT DISCLAIMER** ⚠️

**THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION**

- **DO NOT USE** unless explicitly asked to reference old code
- **CURRENT IMPLEMENTATION** is in `system-design/` directory
- This file is for **REFERENCE ONLY** to understand legacy approaches
- All new development should follow current system design specifications

---

# AWS-Native Fashion Recommendation System: Master Architecture Overview

## Executive Summary

### Project Overview
A production-scale fashion recommendation system built on AWS serverless architecture, designed for learning ML engineering principles while providing patterns scalable to enterprise deployments. This system demonstrates end-to-end implementation of modern recommendation systems using the H&M fashion dataset with two-tower neural networks, vector databases, and 4-stage recommendation pipelines.

### Business Value Proposition
- **Personalization:** Individual style preferences with 15%+ CTR and 3-5% conversion rates
- **Discovery:** Advanced similarity search helping users find items they didn't know they wanted
- **Revenue Impact:** Target 20-25% of total sales generated through recommendation engine
- **Cost Efficiency:** Serverless-first architecture scaling from $50/month (learning) to production workloads

### Key Architectural Decisions
1. **Serverless-First Architecture:** Eliminates operational overhead while maintaining scalability
2. **Embedding-Based Hybrid Approach:** Combines collaborative filtering with content-based recommendations
3. **4-Stage Pipeline:** Optimized balance of quality, latency, and business requirements
4. **AWS-Native Integration:** Leverages managed services for rapid development and scaling

### Learning Outcomes
- **ML Engineering:** Two-tower models, embedding systems, vector databases, feature engineering
- **System Design:** Microservices, caching strategies, API development, monitoring patterns
- **AWS Ecosystem:** Serverless patterns, data lakes, ML orchestration, cost optimization
- **Production Patterns:** CI/CD, observability, security, incident response

## System Architecture Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL INTERFACES                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Mobile/Web Apps → API Gateway → CDN → Load Balancer                           │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│ │  Lambda API     │  │  Business Rules │  │  User Profile   │                  │
│ │  Orchestrator   │  │  Engine         │  │  Manager        │                  │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
│           │                    │                    │                          │
│  ┌─────────────────────────────────────────────────────────────┐               │
│  │        4-Stage Recommendation Pipeline                      │               │
│  │  Stage 1 → Stage 2 → Stage 3 → Stage 4                    │               │
│  │  Generate  Filter    Rank     Order                       │               │
│  └─────────────────────────────────────────────────────────────┘               │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────────────────┐
│                               ML LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│ │  SageMaker      │  │  Vector DB      │  │  Feature        │                  │
│ │  Training       │  │ (OpenSearch)    │  │  Engineering    │                  │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
│           │                    │                    │                          │
│  ┌─────────────────────────────────────────────────────────────┐               │
│  │          Two-Tower Neural Architecture                      │               │
│  │  User Tower ←→ Shared Embedding Space ←→ Item Tower        │               │
│  └─────────────────────────────────────────────────────────────┘               │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────────────────┐
│                              DATA LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│ │  S3 Data Lake   │  │  Glue ETL       │  │  Athena Query   │                  │
│ │  (Raw→Features) │  │  Pipelines      │  │  Engine         │                  │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
│           │                    │                    │                          │
│  ┌─────────────────────────────────────────────────────────────┐               │
│  │           Parquet-Optimized Data Pipeline                   │               │
│  │  Raw Data → Validation → Features → Quality Checks         │               │
│  └─────────────────────────────────────────────────────────────┘               │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│ │  VPC Network    │  │  Serverless     │  │  Storage        │                  │
│ │  Security       │  │  Compute        │  │  Foundation     │                  │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
│           │                    │                    │                          │
│  ┌─────────────────────────────────────────────────────────────┐               │
│  │         IAM, KMS, Cost Optimization                         │               │
│  └─────────────────────────────────────────────────────────────┘               │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────────────────┐
│                          OPERATIONS LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│ │  CloudWatch     │  │  GitHub Actions │  │  Security &     │                  │
│ │  Monitoring     │  │  CI/CD Pipeline │  │  Compliance     │                  │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
│           │                    │                    │                          │
│  ┌─────────────────────────────────────────────────────────────┐               │
│  │    Incident Response & Cost Optimization                    │               │
│  └─────────────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Relationships

**Data Flow Architecture:**
```
H&M Dataset (S3) → 
Glue ETL (Feature Engineering) → 
SageMaker Training (Two-Tower Models) → 
OpenSearch (Vector Embeddings) → 
Lambda API (4-Stage Pipeline) → 
JSON Recommendations
```

**Cross-Layer Integration:**
- **Infrastructure ↔ All Layers:** VPC, security groups, IAM policies
- **Data ↔ ML:** Feature engineering feeds model training and inference
- **ML ↔ Application:** Embedding similarity, ranking model predictions
- **Application ↔ Operations:** Custom metrics, performance monitoring

## Layer-by-Layer Summary

### 1. Project Description ([project-description.md](project-description.md))
**Purpose:** Foundational learning objectives and system architecture overview

**Key Components:**
- Two-tower neural architecture for embedding-based recommendations
- 4-stage pipeline: Generate → Filter → Rank → Order
- Learning vs. production trade-offs (10K users vs. 1.3M)
- Success metrics: 8-12% CTR, 3-5% conversion, <200ms latency

**Implementation Context:** Balances learning cost-effectiveness with production-ready patterns

### 2. Infrastructure Layer ([infrastructure-layer.md](infrastructure-layer.md))
**Purpose:** Serverless-first AWS foundation minimizing operational overhead

**Key Components:**
- **Compute:** Lambda functions, Glue serverless, AWS Batch for ML training
- **Network:** VPC with public/private subnets, security groups, NAT gateways
- **Storage:** S3 data lake, DynamoDB for user profiles, ElastiCache for hot data
- **Security:** IAM least-privilege policies, encryption at rest/transit

**Cost Optimization:** Spot instances, auto-scaling, lifecycle policies
**Learning Simplification:** Single AZ, reduced security complexity

### 3. Data Layer ([data-layer.md](data-layer.md))
**Purpose:** Scalable data lake with Parquet optimization and quality monitoring

**Key Components:**
- **Storage Architecture:** S3 with organized folder structure (raw→processed→features→embeddings)
- **ETL Pipelines:** Glue jobs for validation, feature engineering, quality monitoring
- **Query Engine:** Athena with partition pruning for 95% cost reduction
- **Data Catalog:** Glue catalog for schema discovery and lineage tracking

**Performance Benefits:** 86% storage reduction with Parquet, 200x query speedup with partitioning
**Learning Simplification:** Daily batch processing vs. real-time streaming

### 4. ML Layer ([ml-layer.md](ml-layer.md))
**Purpose:** Embedding-based hybrid recommendation using two-tower architecture

**Key Components:**
- **Neural Architecture:** Dual encoders (user/item towers) with shared embedding space
- **Vector Database:** OpenSearch with ANN indices for sub-50ms similarity search
- **SageMaker Integration:** Hyperparameter tuning, model registry, batch inference
- **4-Stage Pipeline:** Optimized for 110-200ms total latency

**Model Performance:** 95%+ recall with ANN search, 0.87+ similarity scores
**Alternative Approaches:** Matrix factorization, content-based, session-based options analyzed

### 5. Application Layer ([application-layer.md](application-layer.md))
**Purpose:** Serverless API orchestration with multi-level caching and business rules

**Key Components:**
- **API Gateway:** Rate limiting, authentication, CORS, throttling
- **Lambda Orchestration:** 4-stage pipeline execution with fallback strategies
- **Caching Strategy:** API Gateway → ElastiCache → DynamoDB → OpenSearch
- **Business Rules:** Diversity constraints, new arrivals promotion, A/B testing

**Performance Targets:** P50: 50ms, P95: 150ms, P99: 200ms
**Cache Hit Rates:** L1: 30%, L2: 60-70% for significant latency reduction

### 6. Operations Layer ([operations-layer.md](operations-layer.md))
**Purpose:** Production monitoring, deployment automation, and incident response

**Key Components:**
- **Monitoring:** CloudWatch dashboards, custom metrics, alerting
- **CI/CD Pipeline:** GitHub Actions with 10-stage deployment workflow
- **Blue-Green Deployment:** Canary testing, automatic rollback, traffic shifting
- **Security:** Encryption, IAM policies, audit logging, compliance

**Operational Metrics:** 99.9% availability, <15min MTTR, <5% change failure rate
**Cost Monitoring:** Daily budgets, service-level cost breakdown, optimization alerts

## End-to-End Data Flow Architecture

### Offline Processing (Daily/Weekly)

```
Raw H&M Data (S3 raw/) →
┌─────────────────────────────────────────────────────────────────┐
│                    Data Validation & ETL                        │
├─────────────────────────────────────────────────────────────────┤
│ AWS Glue Job: hm-data-validation-and-ingestion                 │
│ • Schema validation                                             │
│ • Data quality checks                                           │
│ • CSV → Parquet conversion                                      │
│ • Partitioning by date                                          │
└─────────────────┬───────────────────────────────────────────────┘
                  │
Processed Data (S3 processed/) →
┌─────────────────────────────────────────────────────────────────┐
│                   Feature Engineering                           │
├─────────────────────────────────────────────────────────────────┤
│ AWS Glue Job: hm-feature-engineering                           │
│ • User features (demographics + behavior)                       │
│ • Item features (attributes + popularity)                       │
│ • Interaction matrix (sparse user-item pairs)                   │
│ • Temporal features (seasonal, recency)                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
Features (S3 features/) →
┌─────────────────────────────────────────────────────────────────┐
│                    Model Training                               │
├─────────────────────────────────────────────────────────────────┤
│ SageMaker Training Job: two-tower-recommender                  │
│ • User tower: demographics + behavior → 256D embedding         │
│ • Item tower: attributes + popularity → 256D embedding         │
│ • Contrastive loss training                                     │
│ • Hyperparameter tuning                                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
Trained Model (S3 models/) →
┌─────────────────────────────────────────────────────────────────┐
│                 Batch Inference                                 │
├─────────────────────────────────────────────────────────────────┤
│ SageMaker Batch Transform: embedding-generation                │
│ • Generate user embeddings for all users                        │
│ • Generate item embeddings for all items                        │
│ • Store in S3 with versioning                                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
Embeddings (S3 embeddings/) →
┌─────────────────────────────────────────────────────────────────┐
│                Vector Database Update                           │
├─────────────────────────────────────────────────────────────────┤
│ Lambda Trigger: update-vector-database                         │
│ • Bulk index embeddings in OpenSearch                          │
│ • HNSW index for fast ANN search                               │
│ • Enable real-time similarity queries                           │
└─────────────────────────────────────────────────────────────────┘
```

### Online Serving (Real-time)

```
User Request (API Gateway) →
┌─────────────────────────────────────────────────────────────────┐
│                  Stage 1: Candidate Generation                 │
├─────────────────────────────────────────────────────────────────┤
│ Lambda Function: recommendation-orchestrator                   │
│ • Retrieve user embedding (DynamoDB/Cache)                      │
│ • k-NN search in OpenSearch (100 candidates)                   │
│ • Fallback to popular items for cold start                      │
│ • Latency Target: <50ms                                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
Candidates (~100 items) →
┌─────────────────────────────────────────────────────────────────┐
│                      Stage 2: Filtering                        │
├─────────────────────────────────────────────────────────────────┤
│ • Remove recently viewed items (ElastiCache lookup)            │
│ • Check inventory availability (DynamoDB)                       │
│ • Apply age/region restrictions                                 │
│ • Latency Target: <20ms                                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
Filtered Candidates (~50 items) →
┌─────────────────────────────────────────────────────────────────┐
│                       Stage 3: Ranking                         │
├─────────────────────────────────────────────────────────────────┤
│ • Extract user/item/contextual features                         │
│ • Batch prediction via SageMaker endpoint (XGBoost)            │
│ • Score candidates by purchase probability                      │
│ • Latency Target: <60ms                                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
Ranked Candidates →
┌─────────────────────────────────────────────────────────────────┐
│                   Stage 4: Ordering                            │
├─────────────────────────────────────────────────────────────────┤
│ • Apply diversity constraints (max 5 per category)             │
│ • Promote new arrivals (20% ratio)                             │
│ • Apply business rules (seasonality, user segment)             │
│ • Generate recommendation reasons                               │
│ • Latency Target: <20ms                                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
Final Recommendations (JSON Response) →
User Application
```

## Technology Stack

### Complete AWS Service Mapping

| Layer | Core Services | Supporting Services | Data Storage |
|-------|--------------|-------------------|--------------|
| **Application** | API Gateway, Lambda | ElastiCache, CloudFront | DynamoDB |
| **ML** | SageMaker, OpenSearch | Batch, ECR | S3 (models) |
| **Data** | Glue, Athena | EventBridge, Step Functions | S3 (data lake) |
| **Infrastructure** | VPC, IAM | KMS, CloudTrail | EBS, EFS |
| **Operations** | CloudWatch, Systems Manager | SNS, SQS | CloudWatch Logs |

### Integration Patterns

**Synchronous Integrations:**
- API Gateway ↔ Lambda (REST API)
- Lambda ↔ DynamoDB (user data lookup)
- Lambda ↔ SageMaker (model inference)
- Lambda ↔ OpenSearch (vector similarity)

**Asynchronous Integrations:**
- S3 Events → Lambda (data processing triggers)
- EventBridge → Glue (scheduled ETL jobs)
- SQS → Lambda (interaction processing)
- SNS → Multiple targets (alerting)

**Caching Layers:**
- API Gateway (response caching)
- ElastiCache (hot data)
- Lambda memory (connection pooling)
- DynamoDB DAX (microsecond latency)

## Key Design Decisions

### 1. Serverless-First Architecture

**Decision:** Use Lambda, Glue, and managed services instead of EC2/EKS
**Rationale:**
- Eliminates operational overhead (scaling, patching, monitoring)
- Cost-efficient for learning projects (pay-per-use model)
- Automatic scaling for production workloads
- Integrated security and compliance

**Trade-offs:**
- Cold start latency (mitigated with provisioned concurrency)
- Vendor lock-in (offset by learning AWS ecosystem deeply)
- Limited customization (sufficient for recommendation workloads)

### 2. Embedding-Based Hybrid Approach

**Decision:** Two-tower neural network generating user/item embeddings
**Rationale:**
- Handles both collaborative and content-based signals
- Enables fast similarity search with vector databases
- Scales to large catalogs with sub-linear query complexity
- Supports cold-start through content features

**Alternative Approaches Evaluated:**
- Matrix Factorization: Faster but limited personalization
- Deep Learning End-to-End: Higher accuracy but requires more data/compute
- Session-Based RNNs: Good for sequential patterns but complex implementation

### 3. 4-Stage Recommendation Pipeline

**Decision:** Separate candidate generation, filtering, ranking, and ordering
**Rationale:**
- Optimizes different constraints at each stage (efficiency → quality)
- Enables independent scaling and A/B testing
- Clear separation of ML logic and business rules
- Maintainable and debuggable architecture

**Stage Rationale:**
- **Stage 1:** Fast similarity search (computational efficiency)
- **Stage 2:** Business logic (inventory, user history)
- **Stage 3:** ML precision (purchase probability)
- **Stage 4:** User experience (diversity, freshness)

### 4. AWS-Native vs. Open Source

**Decision:** AWS managed services over self-managed alternatives
**Rationale:**
- Faster time-to-value for learning projects
- Production-grade reliability and security built-in
- Integrated monitoring and cost management
- Easier scaling from learning to production

**Comparison:**
| Component | AWS Choice | Open Source Alternative | Rationale |
|-----------|------------|------------------------|-----------|
| Vector DB | OpenSearch | Pinecone, Weaviate | Native AWS integration |
| ML Platform | SageMaker | MLflow, Kubeflow | Serverless model serving |
| Data Lake | S3 + Glue | HDFS + Spark | Managed scaling |
| API Layer | API Gateway + Lambda | Express.js + K8s | Serverless cost model |

## Implementation Roadmap

### Suggested Order for Implementation

#### Phase 1: Data Foundation (Weeks 1-2)
**Dependencies:** None
**Goal:** Establish data lake and feature engineering

1. **Set up S3 data lake structure**
   - Create buckets with proper folder organization
   - Configure lifecycle policies and encryption
   - Upload H&M dataset samples

2. **Build Glue ETL pipelines**
   - Data validation and ingestion job
   - Feature engineering pipeline
   - Data quality monitoring

3. **Configure Athena for querying**
   - Register tables in Glue Catalog
   - Test partition pruning and query performance

**Success Criteria:** Features available in S3, Athena queries working

#### Phase 2: Infrastructure Setup (Weeks 2-3)
**Dependencies:** AWS account setup
**Goal:** Core infrastructure and security

1. **VPC and networking**
   - Public/private subnets
   - Security groups and NACLs
   - NAT Gateway configuration

2. **IAM policies and roles**
   - Least-privilege service roles
   - Cross-service permissions
   - Key management (KMS)

3. **Database setup**
   - DynamoDB tables for user profiles
   - ElastiCache cluster configuration
   - Connection pooling setup

**Success Criteria:** Infrastructure tested, services communicating

#### Phase 3: ML Pipeline (Weeks 3-5)
**Dependencies:** Phase 1 (features ready)
**Goal:** Two-tower model training and inference

1. **SageMaker training pipeline**
   - Two-tower model implementation
   - Hyperparameter tuning jobs
   - Model evaluation framework

2. **Vector database setup**
   - OpenSearch cluster configuration
   - Index creation with HNSW settings
   - Embedding upload and testing

3. **Batch inference pipeline**
   - Embedding generation jobs
   - Model versioning and registry
   - Vector database update automation

**Success Criteria:** Trained models, embeddings in vector DB, similarity search working

#### Phase 4: Application Layer (Weeks 5-6)
**Dependencies:** Phase 3 (ML models ready)
**Goal:** REST API with 4-stage pipeline

1. **Core Lambda functions**
   - Recommendation orchestrator
   - User profile manager
   - Business rules engine

2. **API Gateway configuration**
   - Endpoint definitions
   - Rate limiting and throttling
   - Authentication setup

3. **Caching implementation**
   - Multi-level cache architecture
   - Cache invalidation strategies
   - Performance optimization

**Success Criteria:** API returning recommendations <200ms, cache hit rates >60%

#### Phase 5: Operations & Monitoring (Weeks 6-7)
**Dependencies:** Phase 4 (API working)
**Goal:** Production-ready monitoring and deployment

1. **CloudWatch monitoring**
   - Custom metrics implementation
   - Dashboard creation
   - Alerting configuration

2. **CI/CD pipeline**
   - GitHub Actions workflow
   - Blue-green deployment
   - Automated testing

3. **Security hardening**
   - Encryption verification
   - Audit logging
   - Compliance checks

**Success Criteria:** Full monitoring, automated deployments, security compliance

#### Phase 6: Testing & Optimization (Weeks 7-8)
**Dependencies:** Phase 5 (monitoring ready)
**Goal:** Performance testing and optimization

1. **Load testing**
   - API performance under load
   - Database scaling verification
   - Cost optimization

2. **A/B testing framework**
   - Experiment configuration
   - Statistical significance testing
   - Results analysis

3. **Documentation completion**
   - Runbooks for incident response
   - User guides and API documentation
   - Learning outcomes documentation

**Success Criteria:** System handles target load, A/B tests running, documentation complete

### Dependencies Between Layers

**Critical Path Dependencies:**
```
Data Layer → ML Layer → Application Layer → Operations Layer
     ↓         ↓            ↓                ↓
Infrastructure Layer (foundational for all)
```

**Parallel Development Opportunities:**
- Infrastructure + Data Layer setup can proceed simultaneously
- Operations monitoring setup can begin once Application Layer is functional
- Documentation can be written throughout development

## Cross-Layer Integration

### Data Interfaces

**S3 → SageMaker:**
- Features stored as Parquet files with standardized schema
- Metadata includes data types, null handling, feature descriptions
- Versioning enables reproducible model training

**SageMaker → OpenSearch:**
- Embeddings generated in batch and stored in S3
- Lambda trigger updates vector indices automatically
- Embedding versioning supports model rollback

**OpenSearch → Lambda API:**
- REST API for k-NN similarity search
- JSON request/response format with similarity scores
- Connection pooling for performance

### API Contracts

**Internal Service APIs:**

```python
# ML Layer → Application Layer
class MLServiceAPI:
    def get_user_embedding(user_id: str) -> List[float]
    def get_similar_items(embedding: List[float], k: int) -> List[ItemScore]
    def rank_candidates(features: List[Dict]) -> List[PredictionScore]

# Data Layer → ML Layer  
class DataServiceAPI:
    def get_user_features(user_id: str) -> Dict
    def get_item_features(item_ids: List[str]) -> Dict[str, Dict]
    def get_interaction_history(user_id: str, days: int) -> List[Interaction]

# Application Layer → Operations Layer
class MetricsAPI:
    def record_latency(operation: str, duration_ms: float)
    def record_error(operation: str, error_type: str)
    def record_business_metric(name: str, value: float)
```

**External API Contract:**

```yaml
RecommendationAPI:
  Endpoint: GET /v1/recommendations/{user_id}
  Request:
    Parameters:
      user_id: string (required)
      limit: integer (default: 20, max: 100)
      experiment: string (optional)
    Headers:
      Authorization: Bearer token
      X-Device-Type: mobile|web
  
  Response:
    Format: JSON
    Schema:
      recommendations: Array<Recommendation>
      request_id: string
      served_from: cache|realtime|fallback
      latency_ms: number
    
    Recommendation:
      article_id: string
      title: string
      category: string
      price: number
      confidence_score: number (0-1)
      image_url: string
      reason: string
```

## Performance & Scalability

### System-Wide Performance Characteristics

**Latency Breakdown:**
```
End-to-End Request: 110-200ms (P95)
├── API Gateway: 5-10ms
├── Lambda Cold Start: 0-100ms (mitigated with provisioned concurrency)
├── Stage 1 (Vector Search): 30-50ms
├── Stage 2 (Filtering): 10-20ms  
├── Stage 3 (Ranking): 40-60ms
├── Stage 4 (Ordering): 10-20ms
└── Response Formatting: 5-10ms
```

**Throughput Characteristics:**
```
Learning Project: 10-50 RPS sustained
Production Ready: 1000+ RPS with:
├── Lambda: 1000 concurrent executions
├── OpenSearch: 3-node cluster  
├── DynamoDB: 1000 read/write capacity units
└── ElastiCache: r6g.xlarge cluster
```

### Scaling Patterns

**Horizontal Scaling:**
- Lambda functions: Automatic scaling up to 10,000 concurrent executions
- DynamoDB: On-demand billing with automatic scaling
- OpenSearch: Multi-node clusters with shard distribution
- ElastiCache: Cluster mode with multiple shards

**Vertical Scaling:**
- Lambda memory: 128MB → 10GB based on workload
- DynamoDB capacity: On-demand → provisioned for predictable workloads
- OpenSearch nodes: t3.small → r5.xlarge for memory-intensive operations
- ElastiCache nodes: cache.t3.micro → cache.r6g.xlarge for larger datasets

**Auto-Scaling Triggers:**
- Lambda concurrency utilization > 70%
- DynamoDB consumed capacity > 80%
- ElastiCache CPU utilization > 70%
- OpenSearch query latency > 100ms

## Cost Analysis

### Complete Cost Breakdown

#### Learning Project (Monthly Costs)

| Service | Usage | Cost |
|---------|--------|------|
| **Lambda** | 1M requests, 512MB, 200ms avg | $8 |
| **API Gateway** | 1M requests | $3.50 |
| **DynamoDB** | 10K users, 100K interactions | $12 |
| **ElastiCache** | cache.t3.micro | $13 |
| **OpenSearch** | t3.small.search | $24 |
| **SageMaker** | Training 2hr/month, inference | $15 |
| **S3** | 10GB data, standard storage | $2 |
| **Glue** | 1 DPU-hour daily | $13 |
| **CloudWatch** | Basic monitoring | $5 |
| **Data Transfer** | Minimal cross-AZ | $3 |
| **Total** | | **~$98/month** |

#### Production Scale (Monthly Costs)

| Service | Usage | Cost |
|---------|--------|------|
| **Lambda** | 100M requests, 1GB memory | $300 |
| **API Gateway** | 100M requests | $350 |
| **DynamoDB** | 1M users, 10M interactions/day | $400 |
| **ElastiCache** | cache.r6g.xlarge cluster | $340 |
| **OpenSearch** | 3x m5.large.search | $450 |
| **SageMaker** | Daily training, real-time endpoints | $800 |
| **S3** | 1TB data, intelligent tiering | $50 |
| **Glue** | 50 DPU-hours daily | $650 |
| **CloudWatch** | Comprehensive monitoring | $150 |
| **Data Transfer** | Cross-AZ, internet egress | $200 |
| **Reserved Instances** | 1-year commitment savings | -$800 |
| **Total** | | **~$2,890/month** |

### Cost Optimization Economics

**Learning vs Production Scaling:**
- **30x user scale:** Learning (10K users) → Production (1M users)
- **30x cost scale:** $98 → $2,890 monthly
- **Linear scaling:** Cost per user remains ~$0.01/user/month
- **Optimization opportunity:** Reserved instances save ~22% at production scale

**Break-even Analysis:**
- **Development ROI:** 2-4 weeks implementation → production-ready patterns
- **Knowledge ROI:** Learning investment pays dividends in career advancement
- **Business ROI:** 20% revenue increase from recommendations covers infrastructure costs

## Success Metrics

### Business KPIs

**Recommendation Quality:**
- **Click-Through Rate:** Target 15% (industry avg: 8-12%)
- **Conversion Rate:** Target 4% from recommendations
- **Revenue Attribution:** 20-25% of total sales through recommendations
- **User Engagement:** 40% increase in session duration

**User Experience:**
- **Cold Start:** New users receive quality recommendations within 24 hours
- **Diversity:** >0.8 diversity score across categories
- **Freshness:** 20% new arrivals in top-10 recommendations
- **Personalization:** Different recommendations for 90%+ user pairs

### Technical Metrics

**Performance:**
- **API Latency:** P95 <150ms, P99 <200ms
- **Availability:** >99.9% uptime
- **Throughput:** Support 1000+ concurrent users
- **Error Rate:** <0.1% API errors

**ML Model Quality:**
- **Precision@10:** >0.15 for recommendation accuracy
- **Recall@100:** >0.80 for candidate coverage  
- **AUC Score:** >0.72 for ranking model
- **Embedding Quality:** Similarity correlates with user preferences

### Learning Objectives Measurement

**ML Engineering Skills:**
- ✅ Implement two-tower neural networks
- ✅ Build vector similarity search systems
- ✅ Design feature engineering pipelines
- ✅ Evaluate recommendation system metrics

**System Design Skills:**
- ✅ Design microservices architecture
- ✅ Implement multi-level caching strategies
- ✅ Build serverless API systems
- ✅ Design for scalability and reliability

**AWS Expertise:**
- ✅ Serverless compute patterns (Lambda, Glue)
- ✅ Data lake architecture (S3, Athena)
- ✅ ML platform usage (SageMaker)
- ✅ Operations and monitoring (CloudWatch)

**Production Thinking:**
- ✅ CI/CD pipeline implementation
- ✅ Monitoring and incident response
- ✅ Security and compliance patterns
- ✅ Cost optimization strategies

## Getting Started Guide

### Next Steps for Implementation

#### For New Learners
1. **Start with Project Description:** Understand the business problem and architecture overview
2. **Review H&M Dataset:** Familiarize yourself with the data structure and patterns
3. **Set up AWS Account:** Create account with billing alerts and basic security setup
4. **Begin with Data Layer:** Follow the implementation roadmap Phase 1

#### Recommended Reading Order
1. [project-description.md](project-description.md) - Business context and architecture
2. [infrastructure-layer.md](infrastructure-layer.md) - AWS setup and security
3. [data-layer.md](data-layer.md) - Data lake and ETL implementation  
4. [ml-layer.md](ml-layer.md) - Two-tower model and vector database
5. [application-layer.md](application-layer.md) - API development and business logic
6. [operations-layer.md](operations-layer.md) - Monitoring and deployment

#### Implementation Priorities

**Minimum Viable Product (Week 4):**
- Basic data pipeline (CSV → Parquet → Features)
- Simple similarity-based recommendations
- REST API returning JSON recommendations
- Basic monitoring dashboard

**Production-Ready System (Week 8):**
- Full ML pipeline with two-tower training
- 4-stage recommendation pipeline
- Multi-level caching and optimization
- Comprehensive monitoring and CI/CD

**Enterprise Scaling (Week 12+):**
- A/B testing framework
- Multi-region deployment
- Advanced ML techniques (ensemble models, online learning)
- Business intelligence and analytics

### Prerequisites

**Technical Skills:**
- Python programming (intermediate)
- Basic machine learning concepts
- REST API development
- AWS fundamentals (or willingness to learn)

**AWS Account Setup:**
- AWS account with programmatic access
- Billing alerts configured
- Basic IAM user with appropriate permissions
- AWS CLI configured locally

**Development Environment:**
- Python 3.11+ environment
- Git for version control
- Code editor (VS Code recommended)
- Jupyter notebooks for data exploration

### Quick Start Commands

```bash
# Clone repository and setup
git clone <repository-url>
cd fashion-recommendation-system
pip install -r requirements.txt

# Configure AWS credentials
aws configure

# Deploy infrastructure
aws cloudformation deploy --template-file infrastructure/cloudformation/main.yaml \
  --stack-name fashion-recommender --capabilities CAPABILITY_IAM

# Upload sample data
aws s3 sync dataset/ s3://fashion-recommender-data-dev/raw/

# Run data pipeline
aws glue start-job-run --job-name hm-data-validation-and-ingestion

# Test API endpoint
curl -X GET "https://api.fashion-recommender.example.com/v1/recommendations/user123" \
  -H "Authorization: Bearer <token>"
```

---

This master architecture overview provides a comprehensive guide to building an AWS-native fashion recommendation system that balances learning objectives with production-ready patterns. The system scales from cost-effective learning implementations to enterprise-grade deployments while teaching modern ML engineering practices and AWS cloud architecture patterns.

The documentation enables both immediate implementation for learning and future scaling to production workloads, making it valuable for career development and practical ML system development experience.
---
⚠️ **END OF REFERENCE PROJECT FILE** ⚠️

Remember: This is archived code. Use `system-design/` for current implementation.

---
