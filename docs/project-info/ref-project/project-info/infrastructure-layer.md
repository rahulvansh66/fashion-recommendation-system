# Infrastructure Layer: AWS-Native Foundation

## Overview

**Philosophy:** Serverless-first architecture minimizing operational overhead while maintaining production-grade scalability patterns.

**Key Decision:** Eliminate container orchestration (EKS/Kubernetes) in favor of managed services to reduce DevOps complexity for learning project scope.

## Compute Strategy

### Serverless Compute Services

**AWS Lambda**
- **Use Case:** API orchestration, business logic, lightweight data processing
- **Configuration:**
  ```yaml
  Runtime: Python 3.11
  Memory: 512MB - 3008MB (auto-scaling based on workload)
  Timeout: 15 minutes maximum
  Concurrent Executions: 100 (can scale to 10,000+ in production)
  ```
- **Cost Optimization:** Pay only for compute time used, automatic scaling
- **Learning Benefits:** No server management, integrated with AWS services

**AWS Glue (Serverless Apache Spark)**
- **Use Case:** ETL jobs, feature engineering, data transformations
- **Configuration:**
  ```yaml
  Worker Type: G.1X (4 vCPU, 16GB RAM)
  Number of Workers: 2-10 (auto-scaling)
  Max Capacity: 100 DPU (Data Processing Units)
  Job Timeout: 2880 minutes (48 hours)
  ```
- **Advantages:** Managed Spark environment, automatic scaling, pay-per-minute
- **Data Catalog Integration:** Schema discovery, lineage tracking

**AWS Batch (Managed Compute Environments)**
- **Use Case:** ML model training, large-scale batch inference
- **Configuration:**
  ```yaml
  Compute Environment: EC2 Spot Instances
  Instance Types: [m5.large, m5.xlarge, c5.xlarge]
  Min/Desired/Max vCPUs: 0/16/1000  
  Spot Fleet Request: 70% cost savings vs On-Demand
  ```
- **Queue Configuration:** High-priority queue for training, normal queue for inference
- **Learning Benefits:** Significant cost savings, automatic resource provisioning

### Orchestration Services

**AWS Step Functions**
- **Use Case:** Workflow coordination for ML pipelines
- **State Machine Types:**
  - **Standard:** Long-running workflows (model training pipelines)
  - **Express:** High-volume, short-duration (real-time inference coordination)
- **Integration:** Native connectors to Lambda, SageMaker, Glue, Batch
- **Error Handling:** Built-in retry logic, dead letter queues, monitoring

**Amazon EventBridge**
- **Use Case:** Event-driven architecture, decoupled service communication
- **Events:** Model training completion, data pipeline status, batch job results
- **Rules:** Route events to appropriate services (Lambda, SNS, SQS)

## Network Architecture

### VPC Design

**Production-Ready Network Topology:**
```
Fashion-Recommender-VPC (10.0.0.0/16)
├── Public Subnets (10.0.1.0/24, 10.0.2.0/24)
│   ├── NAT Gateways
│   └── Application Load Balancers (if needed)
├── Private Subnets (10.0.11.0/24, 10.0.12.0/24)  
│   ├── Lambda Functions (VPC-enabled)
│   ├── Glue Jobs
│   └── Batch Compute Environments
└── Database Subnets (10.0.21.0/24, 10.0.22.0/24)
    ├── RDS instances (if used)
    └── ElastiCache clusters
```

**Learning Project Simplification:**
- Single Availability Zone deployment
- Minimal subnets (1 public, 1 private)
- Basic security groups without advanced networking

### Security Groups Configuration

**Lambda Security Group:**
```yaml
Inbound Rules:
  - HTTPS (443) from API Gateway Security Group
  - Custom TCP (5432) to RDS (if using PostgreSQL)

Outbound Rules:  
  - HTTPS (443) to internet (for AWS API calls)
  - Custom ports to ElastiCache, OpenSearch
```

**Data Processing Security Group (Glue/Batch):**
```yaml  
Inbound Rules:
  - None (no direct access required)

Outbound Rules:
  - HTTPS (443) to internet (for library downloads)
  - S3 traffic (443) to data bucket
  - Custom ports to other AWS services
```

## Storage Foundation

### Amazon S3 Data Lake

**Bucket Structure:**
```
fashion-recommender-data-[environment]
├── raw/
│   ├── articles/
│   ├── customers/  
│   └── transactions/
├── processed/
│   ├── year=2024/month=01/day=15/
│   └── year=2024/month=01/day=16/
├── features/
│   ├── user_features/
│   └── item_features/
├── embeddings/
│   ├── user_embeddings/
│   └── item_embeddings/
└── models/
    ├── training_artifacts/
    └── inference_models/
```

**S3 Configuration:**
- **Storage Classes:** S3 Standard for active data, S3 IA for older features, S3 Glacier for long-term model storage
- **Lifecycle Policies:** Automatic tiering after 30/90 days
- **Versioning:** Enabled for model artifacts and critical datasets
- **Cross-Region Replication:** For production disaster recovery (learning: single region)

**Security:**
- **Encryption:** SSE-S3 for learning, SSE-KMS for production
- **Access Policies:** Least-privilege IAM roles per service
- **VPC Endpoints:** Direct access from private subnets without internet routing

### Database Services

**Amazon DynamoDB**
- **Use Case:** User interaction history, recommendation cache, real-time lookups
- **Table Design:**
  ```yaml
  UserInteractions:
    Partition Key: user_id (String)
    Sort Key: timestamp (Number)
    Attributes: item_id, interaction_type, context
    
  RecommendationCache:
    Partition Key: user_id (String)  
    Sort Key: cache_key (String)
    TTL: recommendation_expires (Number)
  ```
- **Billing:** On-demand for learning (predictable costs), Provisioned for production
- **Performance:** Single-digit millisecond latency, automatic scaling

**Amazon ElastiCache (Redis)**
- **Use Case:** Hot data caching, session management, recent user interactions
- **Configuration:**
  ```yaml
  Node Type: cache.r6g.large (learning), cache.r6g.xlarge+ (production)
  Engine Version: Redis 7.0
  Cluster Mode: Disabled (learning), Enabled (production)
  Backup: Daily snapshots, 7-day retention
  ```
- **Data Structures:** Sorted sets for recent interactions, hash maps for user profiles

## Identity and Access Management (IAM)

### Service Roles

**Lambda Execution Role:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream", 
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/fashion-recommender-*"
    }
  ]
}
```

**SageMaker Execution Role:**
- S3 access to data and model buckets
- CloudWatch logging permissions
- ECR access for custom containers
- VPC permissions for network-isolated training

**Glue Service Role:**  
- S3 read/write permissions for data lake
- Glue Catalog access for schema management
- CloudWatch logging and metrics

### Security Best Practices

**Least Privilege Access:**
- Each service role only includes required permissions
- Resource-specific ARNs instead of wildcards
- Time-limited credentials using AWS STS

**Network Security:**
- Private subnets for compute resources
- VPC endpoints for AWS service communication
- Security groups with minimal required ports

**Data Protection:**
- Encryption in transit (HTTPS/TLS 1.2+)
- Encryption at rest for all storage services
- Key rotation policies for production environments

## Cost Optimization Strategies

### Learning Project Optimizations

**Compute:**
- Spot instances for Batch jobs (70% savings)
- Lambda memory right-sizing based on actual usage
- Glue job optimization (minimal DPU allocation)

**Storage:**
- S3 Intelligent Tiering for automatic cost optimization
- DynamoDB On-Demand billing to avoid over-provisioning  
- ElastiCache reserved instances for predictable workloads

**Development Practices:**
- Automated resource shutdown outside business hours
- Environment-specific resource sizing
- Cost alerts and budgets for spending control

### Production Scaling Considerations

**Reserved Instances:**
- 1-3 year commitments for predictable workloads
- Compute Savings Plans for flexible instance usage
- RDS Reserved Instances for database workloads

**Auto-Scaling:**
- CloudWatch-based scaling triggers
- Predictive scaling for known traffic patterns
- Load testing to determine optimal scaling thresholds

This infrastructure foundation provides production-grade patterns while maintaining cost-effectiveness for learning implementations.
