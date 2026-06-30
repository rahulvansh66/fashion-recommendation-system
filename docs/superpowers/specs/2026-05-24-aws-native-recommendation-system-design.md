# AWS-Native TikTok-like Recommendation System Design

**Date:** 2026-05-24  
**Status:** Design Specification  
**Context:** Learning project mimicking production-scale recommendation systems using H&M dataset

## Project Overview

### Learning Project Scope

This architecture documentation serves as a **learning exercise** to understand production-scale recommendation systems by designing AWS-native alternatives to Hopsworks-based implementations.

**Key Learning Goals:**
- **System Design:** Architect enterprise-grade ML systems using AWS services
- **ML Engineering:** Implement two-tower models, embedding systems, and 4-stage recommendation pipelines  
- **Production Patterns:** Design for scalability, reliability, and cost-efficiency
- **AWS Ecosystem:** Deep-dive into serverless ML, data lakes, and vector databases

**Implementation Reality:**
- **Dataset Size:** Use H&M sample data (~10K users, ~5K items vs. full 1.3M users, 105K items)
- **Cost Optimization:** Leverage free tiers, spot instances, and minimal resource allocation
- **Time Efficiency:** Focus on core functionality over production-grade monitoring/security
- **Learning Priority:** Understand architectural patterns over optimizing for massive scale

**Scale Simulation Strategy:**
- **Design for Production:** Architecture assumes full H&M dataset scale requirements
- **Implement for Learning:** Use representative sample data to validate concepts
- **Document Trade-offs:** Explain what changes when scaling from sample to production

### System Architecture Philosophy

**Selected Approach:** Data Lake + Batch-Optimized Architecture with Embedding-Based Hybrid Serving

**Core Principles:**
1. **Serverless-First:** Minimize operational overhead with managed services
2. **Cost-Effective:** Batch processing with spot instances and intelligent scaling
3. **Proven Pattern:** Implement the 4-stage recommendation pipeline from TikTok-like systems
4. **AWS-Native:** Leverage AWS ecosystem strengths rather than direct Hopsworks replacements

## Architecture Layers

### Layer 1: Infrastructure Layer

**Core Philosophy:** Minimize operational overhead with serverless-first approach

#### Compute Strategy
- **No Kubernetes/EKS** - eliminate cluster management entirely
- **AWS Glue** for ETL jobs (serverless Spark)
- **AWS Batch** for ML training workloads (managed compute environments)
- **Lambda** for lightweight orchestration and API serving
- **Step Functions** for workflow coordination

#### Network & Security
- **VPC** with private subnets for Glue/Batch jobs
- **NAT Gateway** for external ML library downloads during jobs
- **IAM roles** with least-privilege policies per service
- **Secrets Manager** for API keys and connection strings

#### Storage Foundation
- **S3** as primary data lake (raw → processed → feature → model artifacts)
- **DynamoDB** for pre-computed recommendations and user metadata
- **ElastiCache Redis** for hot recommendation caching

**Key Benefit:** Eliminates 90% of traditional DevOps overhead compared to container orchestration.

### Layer 2: Data Layer

#### Storage Architecture
**S3 Data Lake Structure:**
```
s3://fashion-recommender-data/
├── raw/           # Original H&M CSV files
├── processed/     # Cleaned, validated data (Parquet format)
├── features/      # Engineered features for ML
├── embeddings/    # Pre-computed user/item embeddings
└── recommendations/ # Final recommendation outputs
```

#### Data Processing Pipeline
- **AWS Glue Catalog** as metadata store (replaces traditional data warehouse schemas)
- **Glue ETL Jobs** for data transformations (Python/PySpark)
- **Athena** for ad-hoc querying and data validation
- **QuickSight** for data quality monitoring dashboards

#### Data Flow
1. **Ingestion:** H&M CSV → S3 raw/ (one-time or periodic uploads)
2. **Validation:** Glue job validates schema, data types, handles missing values
3. **Transformation:** Convert to Parquet, create time-based partitions
4. **Feature Engineering:** User profiles, item features, interaction matrices
5. **Storage:** Features stored in S3 features/ for ML consumption

**Key Optimization:** Parquet columnar format with date partitioning reduces query costs by 70% compared to CSV and enables efficient time-based filtering.

### Layer 3: ML Layer (Embedding-Based Hybrid)

#### Architecture Overview
**Core Philosophy:** Separate batch learning from real-time serving using embeddings as the bridge

**Components:**
1. **Batch Training Pipeline:** Generates user/item embeddings periodically  
2. **Vector Database:** Stores embeddings for fast similarity search
3. **Real-time API:** Orchestrates 4-stage recommendation process
4. **Feature Cache:** Stores user interaction history for filtering

#### Batch Training Pipeline (Daily/Weekly)
```
Raw Data → Feature Engineering → Two-Tower Training → Embedding Generation → Vector Index Update
```

**AWS Implementation:**
- **SageMaker Training Job:** Trains two-tower model on historical interactions
- **Batch Transform:** Generates embeddings for all users and items  
- **OpenSearch/Pinecone:** Stores embeddings in vector database with ANN indices
- **S3:** Backs up embedding snapshots for version control

**Two-Tower Model Details:**
- **User Tower Input:** Demographics, purchase history aggregates, seasonal preferences
- **Item Tower Input:** Product attributes, category hierarchy, popularity metrics
- **Shared Embedding Space:** 128-256 dimensions for efficient similarity computation
- **Training Objective:** Maximize similarity between user-item pairs that interacted

#### Real-time Serving (4-Stage Pipeline)

**Stage 1: Candidate Generation**
- **Implementation:** OpenSearch k-NN query of user embedding against item embedding index
- **Latency Target:** <50ms for similarity search
- **Fallback:** Popular items if user is new/embedding missing
- **Output:** ~100 candidate items

**Stage 2: Filtering**
- **Implementation:** DynamoDB for user interaction history + ElastiCache for recent interactions
- **Logic:** Remove already seen/purchased items, filter unavailable stock
- **Optimization:** Hybrid caching (ElastiCache for last 30 days, DynamoDB for complete history)
- **Output:** ~50 filtered candidates

**Stage 3: Ranking**
- **Implementation:** SageMaker Endpoint hosting trained ranking model
- **Features:** User features + item features + contextual signals
- **Model Options:** XGBoost for tabular features OR neural networks for complex interactions
- **Output:** Scored candidates with purchase probability

**Stage 4: Ordering**
- **Implementation:** Business rules service with Lambda functions
- **Logic:** Apply diversity constraints, promote new arrivals, A/B testing
- **Personalization:** User-specific ordering preferences
- **Output:** Final top-K personalized recommendations

#### Detailed ML Reasoning & Alternative Approaches

##### Why the 4-Stage Pipeline Works

**Stage 1: Candidate Generation - Deep Dive**
- **Computational Efficiency:** Computing similarity across 100K+ items in real-time is O(n) - too slow
- **ANN Index Solution:** Pre-built indices reduce search to O(log n) with 95%+ accuracy
- **Memory Optimization:** Vector databases use quantization and clustering to fit embeddings in RAM

**Alternative Approaches:**
1. **Matrix Factorization + Precomputed Similarities**
   - ✅ Simple implementation
   - ❌ Static, can't incorporate new items quickly
   - ❌ Memory explosion: O(users × items)

2. **Real-time Deep Learning Inference**
   - ✅ Most personalized
   - ❌ High latency (200-500ms)
   - ❌ Expensive GPU infrastructure

3. **Category-based Filtering + Collaborative Filtering**
   - ✅ Interpretable results
   - ❌ Limited personalization
   - ❌ Cold start problems

**When to Use Each:**
- **ANN Index:** >10K items, need <100ms latency, daily user base
- **Matrix Factorization:** <1K items, batch recommendations acceptable
- **Real-time DL:** Premium users, high-value transactions, latency <500ms OK

**Stage 2: Filtering - Critical for User Experience**
- **Immediate Relevance:** User bought shoes yesterday → don't show same shoes today
- **Inventory Reality:** Don't recommend out-of-stock items
- **Legal Compliance:** Age-restricted, region-blocked items

**Stage 3: Ranking - The Intelligence Layer**

Why similarity alone isn't enough:
- **Temporal Patterns:** User buys winter clothes in November, not July
- **Price Sensitivity:** Similar users might have different budgets  
- **Cross-category Preferences:** Fashion user might also like accessories
- **Business Logic:** Promote new arrivals, clear inventory

**Ranking Model Approaches:**
1. **Gradient Boosting (XGBoost/XGBoost)**
   - ✅ Excellent with tabular features
   - ✅ Fast training and inference
   - ✅ Interpretable feature importance
   - ❌ Limited ability to learn complex interactions

2. **Neural Networks (Deep & Wide, DeepFM)**
   - ✅ Learns feature interactions automatically
   - ✅ Better with high-dimensional sparse features
   - ❌ Longer training time
   - ❌ Less interpretable

**Feature Engineering Strategy:**
```python
# User Features
user_features = {
    'age_bucket': categorical,
    'purchase_frequency': numerical,
    'avg_price_range': numerical,
    'preferred_categories': multi_hot,
    'seasonal_activity': time_series
}

# Item Features  
item_features = {
    'category_hierarchy': categorical,
    'price_tier': categorical,
    'popularity_score': numerical,
    'inventory_velocity': numerical,
    'seasonal_relevance': numerical
}

# Contextual Features
context_features = {
    'time_of_day': categorical,
    'day_of_week': categorical,
    'season': categorical,
    'user_session_length': numerical,
    'device_type': categorical
}
```

##### When to Use Alternative Architectures

**Simple Content-Based (No ML)**
- **Use When:** <1K items, homogeneous catalog, limited user data

**Session-Based (RNNs/Transformers)**
- **Use When:** Strong sequential patterns, short session engagement critical

**Contextual Bandits**
- **Use When:** Heavy exploration needed, fast adaptation to new items critical

**Pure Collaborative Filtering**
- **Use When:** Rich interaction data, sparse content features

##### Why Our Hybrid Approach Wins for Fashion E-commerce

1. **Fashion-Specific Needs:**
   - **Visual Similarity:** Embeddings capture visual patterns well
   - **Seasonal Trends:** Batch retraining captures seasonal shifts
   - **Style Evolution:** Real-time filtering adapts to immediate preferences

2. **Operational Requirements:**
   - **Cost Control:** Batch training uses spot instances
   - **Reliability:** Precomputed embeddings provide fallbacks
   - **Scalability:** ANN indices handle user growth efficiently

### Layer 4: Application Layer

**Philosophy:** Orchestrate ML services into user-facing APIs with high availability and low latency

#### API Gateway Architecture
**Request Flow:**
```
User Request → API Gateway → Lambda Functions → ML/Data Services → Response
```

**AWS Implementation:**
- **API Gateway:** Rate limiting, authentication, request routing
- **Lambda Functions:** Recommendation logic, business rules, response formatting  
- **Application Load Balancer:** For high-throughput scenarios (>1000 RPS)

#### Core Application Components

**1. Recommendation API Service**
```python
# Endpoint: GET /recommendations/{user_id}
def get_recommendations(user_id, limit=20, context={}):
    # Stage 1: Candidate Generation
    candidates = vector_search_service.get_similar_items(user_id, top_k=100)
    
    # Stage 2: Filtering  
    filtered = filtering_service.remove_seen_items(candidates, user_id)
    
    # Stage 3: Ranking
    ranked = ranking_service.score_candidates(filtered, user_id, context)
    
    # Stage 4: Ordering + Business Logic
    final = business_rules_service.apply_rules(ranked, limit)
    
    return format_response(final)
```

**2. User Profile Management**
- **DynamoDB:** Persistent user profiles and interaction history
- **ElastiCache:** Hot cache for recent interactions
- **SQS:** Async processing queue for batch updates

**3. Content Management Integration**
- **Real-time inventory sync**
- **Vector database metadata updates**
- **Cache invalidation strategies**

#### Performance Targets
- **P95 Latency:** <200ms end-to-end
- **Availability:** 99.9% uptime
- **Throughput:** Support 1000 concurrent users

**Caching Strategy:**
- **L1 (API Gateway):** 15-minute user-level cache
- **L2 (ElastiCache):** User interactions and item metadata  
- **L3 (OpenSearch):** Embedding similarity results

### Layer 5: Operations Layer

**Philosophy:** Comprehensive observability and automated operations for a production ML system

#### Monitoring & Observability Strategy

**Application Performance Monitoring:**
- **CloudWatch Dashboards:** API Gateway, Lambda, DynamoDB, ElastiCache, OpenSearch metrics
- **Custom Business Metrics:** CTR, conversion rate, diversity score, cold start coverage
- **SageMaker Model Monitor:** Drift detection on input features

**Data Quality Monitoring:**
```python
# Glue Data Quality Rules
quality_checks = {
    'completeness': 'CHECK user_id IS NOT NULL',
    'uniqueness': 'CHECK COUNT(DISTINCT user_id) = COUNT(*)',
    'timeliness': 'CHECK t_dat >= CURRENT_DATE - INTERVAL 7 DAYS',
    'distribution': 'CHECK AVG(price) BETWEEN 10 AND 500'
}
```

#### Deployment & CI/CD Pipeline

**GitHub Actions Workflow:**
```yaml
stages:
  - data_validation: Validate new data quality
  - feature_engineering: Run Glue ETL jobs
  - model_training: Trigger SageMaker training if needed
  - model_evaluation: Compare against baseline metrics
  - embedding_update: Update vector database
  - integration_testing: Test full recommendation pipeline
  - deployment: Blue-green deployment strategy
```

#### Security & Compliance
- **Data Protection:** S3 encryption, DynamoDB encryption at rest, VPC configuration
- **Privacy Compliance:** Data anonymization, GDPR deletion workflows, audit logging
- **Cost Optimization:** Spot instances, intelligent tiering, usage-based scaling

## Implementation Scope for Learning Project

### Full-Scale Design vs. Learning Implementation

**What We Design For (Production Scale):**
- 1.3M users, 105K items, 31M transactions
- 1000+ concurrent users
- Sub-100ms recommendation latency
- 99.9% availability requirements

**What We Implement (Learning Scale):**
- 10K users, 5K items, 100K transactions (sample dataset)
- 10-50 concurrent users
- Sub-500ms recommendation latency (acceptable for learning)
- Development-grade availability

### Learning-Focused Simplifications

**Infrastructure:**
- Single AWS region deployment
- Minimal monitoring dashboards
- Basic security configurations
- Development-grade networking

**Data Processing:**
- Smaller Glue DPU allocations
- Simplified data validation rules
- Basic partitioning strategies

**ML Training:**
- Smaller model architectures
- CPU-based training (avoid GPU costs)
- Simplified hyperparameter tuning

**Application Layer:**
- Single Lambda function deployments
- Basic caching strategies
- Minimal business rule complexity

### What Remains Production-Ready

**Architectural Patterns:**
- 4-stage recommendation pipeline
- Embedding-based hybrid approach
- Proper separation of concerns across layers

**AWS Service Integration:**
- Correct service selection and configuration
- Scalable data lake design
- Vector database implementation

**ML Engineering:**
- Two-tower model architecture
- Feature engineering best practices
- Model evaluation and monitoring foundations

## Documentation Structure

This design will be documented across 6 separate files in `docs/project-info/`:

1. **`project-description.md`** - System goals, ML methodology, learning objectives
2. **`infrastructure-layer.md`** - VPC, compute services, networking, security foundation
3. **`data-layer.md`** - S3 data lake, ETL pipelines, data processing workflows
4. **`ml-layer.md`** - Two-tower architecture, 4-stage pipeline, embedding systems
5. **`application-layer.md`** - APIs, user management, business logic, performance
6. **`operations-layer.md`** - Monitoring, deployment, security, cost optimization

Plus one master document:
7. **`aws-native-architecture-overview.md`** - Connects all layers with system-wide perspective

Each document will include both production-scale design rationale and learning-project implementation guidance.

## Success Criteria

**Learning Objectives Met:**
- ✅ Understand production-scale ML system architecture
- ✅ Implement AWS-native alternatives to MLOps platforms
- ✅ Build end-to-end recommendation pipeline
- ✅ Apply vector databases and embedding systems
- ✅ Create portfolio-worthy ML engineering project

**Technical Implementation:**
- ✅ Working 4-stage recommendation API
- ✅ Batch training pipeline with embeddings
- ✅ Real-time filtering and ranking
- ✅ Cost-effective AWS resource utilization
- ✅ Documented trade-offs between learning and production scale

This comprehensive design provides both the theoretical foundation for production-scale systems and practical implementation guidance for a cost-effective learning project.