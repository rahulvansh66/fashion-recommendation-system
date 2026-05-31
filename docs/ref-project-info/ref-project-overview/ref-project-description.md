---
⚠️ **REFERENCE PROJECT DISCLAIMER** ⚠️

**THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION**

- **DO NOT USE** unless explicitly asked to reference old code
- **CURRENT IMPLEMENTATION** is in `system-design/` directory
- This file is for **REFERENCE ONLY** to understand legacy approaches
- All new development should follow current system design specifications

---

# Fashion Recommendation System

## Learning Project Overview

### System Purpose
Production-scale recommendation system for fashion e-commerce using H&M dataset, designed to mimic real-world scalable systems while optimizing for learning costs and time.

### Key Learning Objectives
- **ML Engineering:** Implement two-tower models, embedding systems, 4-stage recommendation pipelines
- **System Design:** Architect enterprise-grade ML systems using AWS serverless services
- **Production Patterns:** Design for scalability, reliability, cost-efficiency at scale
- **AWS Ecosystem:** Deep-dive into data lakes, vector databases, ML orchestration

### Implementation Context
- **Dataset:** H&M fashion data (105K products, 1.3M customers, 31M transactions)
- **Learning Scale:** 10K users, 5K items for cost-effective implementation
- **Production Design:** Architecture assumes full dataset scale requirements
- **Time Optimization:** 2-4 weeks implementation timeline

## System Architecture Summary

### Core Approach: Embedding-Based Hybrid Recommendation

**Philosophy:** Separate batch learning from real-time serving using embeddings as the bridge between offline training and online inference.

**Key Components:**
1. **Batch Training Pipeline:** Daily/weekly embedding generation using two-tower models
2. **Vector Database:** Fast similarity search with approximate nearest neighbors (ANN)
3. **Real-time API:** 4-stage recommendation pipeline (Generate → Filter → Rank → Order)
4. **AWS-Native Stack:** Serverless-first architecture minimizing operational overhead

### Original vs. AWS-Native Implementation

**Original Reference System:**
- **Hopsworks:** Feature store, model registry, serving infrastructure
- **Kubernetes:** Container orchestration and scaling
- **Apache Kafka:** Real-time streaming and event processing

**Our AWS-Native Approach:**
- **S3 + Glue:** Serverless data lake and ETL pipelines
- **SageMaker:** Model training, batch inference, endpoints
- **Lambda + API Gateway:** Serverless application layer
- **OpenSearch:** Vector similarity search with managed scaling
- **DynamoDB + ElastiCache:** User state and recommendation caching

### Business Value Proposition

**For Fashion E-commerce:**
- **Personalization:** Individual style preferences and seasonal trends
- **Discovery:** Help users find items they didn't know they wanted
- **Conversion:** Increase purchase probability through relevant recommendations
- **Engagement:** Keep users browsing longer with fresh, interesting suggestions

**Success Metrics:**
- **Click-through Rate:** 8-12% (industry benchmark: 5-8%)
- **Conversion Rate:** 3-5% from recommendations (typical e-commerce: 1-3%)
- **Revenue Impact:** 15-25% of total sales from recommendation engine
- **User Engagement:** 40% increase in session duration

## Technical Methodology

### Two-Tower Neural Architecture

**User Tower Processing:**
```
Demographics + Purchase History + Behavioral Signals
    ↓
Feature Engineering (age groups, price preferences, category affinity)
    ↓
Dense Neural Network (3-4 layers, ReLU activation)
    ↓
128-256 Dimensional User Embedding
```

**Item Tower Processing:**
```
Product Attributes + Category Hierarchy + Popularity Metrics
    ↓
Feature Engineering (brand encoding, seasonal relevance, visual features)
    ↓
Dense Neural Network (parallel to user tower)
    ↓
128-256 Dimensional Item Embedding
```

**Training Objective:**
Maximize cosine similarity between user-item pairs that actually interacted, minimize similarity for negative samples using contrastive loss.

### 4-Stage Recommendation Pipeline

**Stage 1: Candidate Generation (Recall)**
- Query user embedding against item embedding index
- Retrieve ~100 most similar items using ANN search
- Latency target: <50ms
- Fallback: Popular items for new users

**Stage 2: Filtering (Business Logic)**
- Remove previously purchased/viewed items
- Filter unavailable inventory
- Apply age/region restrictions
- Output: ~50 qualified candidates

**Stage 3: Ranking (Precision)**
- Score candidates using rich feature model
- Incorporate contextual signals (time, device, season)
- Predict purchase probability
- Output: Probability-ranked items

**Stage 4: Ordering (Optimization)**
- Apply diversity constraints
- Promote business objectives (new arrivals, clearance)
- Personalize ordering based on user preferences
- Output: Final top-K recommendations

### Data Flow Architecture

**Offline Processing (Daily/Weekly):**
```
Raw H&M Data (S3) → 
Feature Engineering (Glue ETL) → 
Model Training (SageMaker) → 
Embedding Generation (Batch Transform) → 
Vector Index Update (OpenSearch) → 
Performance Monitoring (CloudWatch)
```

**Online Serving (Real-time):**
```
User Request (API Gateway) → 
Lambda Orchestration → 
Embedding Lookup (OpenSearch) → 
Filtering (DynamoDB + ElastiCache) → 
Ranking (SageMaker Endpoint) → 
Business Rules (Lambda) → 
JSON Response
```

## Learning vs. Production Trade-offs

### What We Design For (Production Scale)
- 1.3M users, 105K items, 31M transactions
- 1000+ concurrent users, <100ms P95 latency
- 99.9% availability, multi-region deployment
- Real-time inventory integration
- Advanced A/B testing framework

### What We Implement (Learning Scale)  
- 10K users, 5K items, 100K transactions
- 10-50 concurrent users, <500ms latency acceptable
- Single region, development-grade monitoring
- Simulated inventory management
- Basic recommendation comparison

### Skills Developed Regardless of Scale
- **ML Engineering:** Feature engineering, model evaluation, embedding systems
- **System Architecture:** Microservices design, caching strategies, API development
- **AWS Expertise:** Serverless patterns, data lake design, cost optimization
- **Production Thinking:** Monitoring, testing, deployment automation

This foundation enables scaling to production requirements while learning core concepts cost-effectively.

---
⚠️ **END OF REFERENCE PROJECT FILE** ⚠️

Remember: This is archived code. Use `system-design/` for current implementation.

---
