# Operations Layer: Production Monitoring & Deployment

## Overview

**Architecture Philosophy:** Comprehensive observability and automated operations for a production ML system using AWS-native tools, enabling reliable deployments, rapid incident response, and cost optimization while maintaining development-grade simplicity for learning projects.

**Core Approach:** Serverless operations leveraging CloudWatch for unified monitoring, GitHub Actions for CI/CD automation, and AWS native security services for compliance while scaling from learning projects to production workloads.

**Key Innovation:** Decoupled monitoring (custom metrics + infrastructure metrics), automated incident response with runbooks, and progressive deployment strategies enabling safe releases with quick rollback capabilities.

## Monitoring & Observability Strategy

### CloudWatch Architecture

**Three-Layer Observability Stack:**

```
┌─────────────────────────────────────────────────────────┐
│ Logs (Raw Events)                                       │
│ - Lambda function logs                                  │
│ - Glue ETL job logs                                     │
│ - API Gateway access logs                               │
│ - DynamoDB operation logs                               │
└─────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Metrics (Aggregated Signals)                            │
│ - Infrastructure metrics (CPU, memory, network)         │
│ - Application metrics (latency, errors)                 │
│ - Business metrics (recommendations quality)            │
└─────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────┐
│ Dashboards & Alarms (Actionable Intelligence)           │
│ - Real-time dashboards for monitoring                   │
│ - Alarms for critical conditions                        │
│ - Incident response triggers                            │
└─────────────────────────────────────────────────────────┘
```

### Custom Metrics Implementation

**Recommendation Service Metrics:**

```python
import boto3
import json
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

class MetricsCollector:
    """Collect and publish custom metrics to CloudWatch"""
    
    namespace = 'FashionRecommender'
    
    @staticmethod
    def record_recommendation_latency(user_id: str, latency_ms: float, 
                                     source: str = 'realtime'):
        """
        Record recommendation request latency
        
        Dimensions:
        - Source: 'realtime' | 'cache' | 'fallback'
        - Stage: 'total' | 'stage1' | 'stage2' | 'stage3' | 'stage4'
        """
        cloudwatch.put_metric_data(
            Namespace=MetricsCollector.namespace,
            MetricData=[
                {
                    'MetricName': 'RecommendationLatency',
                    'Value': latency_ms,
                    'Unit': 'Milliseconds',
                    'Dimensions': [
                        {'Name': 'Source', 'Value': source},
                        {'Name': 'Environment', 'Value': 'production'}
                    ],
                    'Timestamp': datetime.utcnow(),
                    'StorageResolution': 1  # High-resolution metrics
                }
            ]
        )
    
    @staticmethod
    def record_recommendations_quality(user_id: str, metrics: dict):
        """
        Record recommendation quality metrics
        
        Metrics:
        - diversity_score: 0-1 (1 = diverse recommendations)
        - freshness_score: 0-1 (1 = new recommendations)
        - cold_start_handled: 0|1 (if cold-start user)
        """
        metric_data = [
            {
                'MetricName': 'DiversityScore',
                'Value': metrics['diversity_score'],
                'Unit': 'None',
                'Dimensions': [
                    {'Name': 'RecommendationType', 'Value': 'Personalized'}
                ]
            },
            {
                'MetricName': 'FreshnessScore',
                'Value': metrics['freshness_score'],
                'Unit': 'None'
            }
        ]
        
        if metrics.get('cold_start_handled'):
            metric_data.append({
                'MetricName': 'ColdStartHandled',
                'Value': 1,
                'Unit': 'Count'
            })
        
        cloudwatch.put_metric_data(
            Namespace=MetricsCollector.namespace,
            MetricData=metric_data
        )
    
    @staticmethod
    def record_ml_model_performance(model_name: str, metrics: dict):
        """
        Record ML model performance metrics
        
        Metrics for ranking model:
        - inference_time_ms: SageMaker endpoint latency
        - batch_size: Number of items ranked
        - error_rate: Inference failures
        """
        cloudwatch.put_metric_data(
            Namespace=MetricsCollector.namespace,
            MetricData=[
                {
                    'MetricName': 'ModelInferenceTime',
                    'Value': metrics['inference_time_ms'],
                    'Unit': 'Milliseconds',
                    'Dimensions': [
                        {'Name': 'ModelName', 'Value': model_name},
                        {'Name': 'BatchSize', 'Value': str(metrics.get('batch_size', 1))}
                    ]
                },
                {
                    'MetricName': 'ModelInferenceErrorRate',
                    'Value': metrics['error_rate'],
                    'Unit': 'Percent',
                    'Dimensions': [
                        {'Name': 'ModelName', 'Value': model_name}
                    ]
                }
            ]
        )
    
    @staticmethod
    def record_data_pipeline_event(pipeline_name: str, status: str, 
                                  records_processed: int, duration_seconds: float):
        """
        Record data pipeline execution metrics
        
        Status: 'success' | 'failure' | 'partial'
        """
        cloudwatch.put_metric_data(
            Namespace=MetricsCollector.namespace,
            MetricData=[
                {
                    'MetricName': 'PipelineRecordsProcessed',
                    'Value': records_processed,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Pipeline', 'Value': pipeline_name},
                        {'Name': 'Status', 'Value': status}
                    ]
                },
                {
                    'MetricName': 'PipelineDuration',
                    'Value': duration_seconds,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'Pipeline', 'Value': pipeline_name}
                    ]
                }
            ]
        )
```

### CloudWatch Dashboards

**Main Operations Dashboard:**

```json
{
  "name": "FashionRecommender-Operations",
  "description": "Real-time operations monitoring dashboard",
  "widgets": [
    {
      "type": "metric",
      "title": "API Latency (P50, P95, P99)",
      "metrics": [
        ["FashionRecommender", "RecommendationLatency", {"stat": "p50"}],
        ["...", "RecommendationLatency", {"stat": "p95"}],
        ["...", "RecommendationLatency", {"stat": "p99"}]
      ],
      "yAxis": {"left": {"label": "Milliseconds", "min": 0, "max": 500}}
    },
    {
      "type": "metric",
      "title": "API Error Rate",
      "metrics": [
        ["AWS/ApiGateway", "5XXError"],
        ["AWS/ApiGateway", "4XXError"]
      ],
      "yAxis": {"left": {"label": "Count", "min": 0}}
    },
    {
      "type": "metric",
      "title": "Lambda Concurrent Executions",
      "metrics": [
        ["AWS/Lambda", "ConcurrentExecutions", {"dimensions": {"FunctionName": "recommendation-orchestrator"}}],
        ["...", "Throttles"]
      ]
    },
    {
      "type": "metric",
      "title": "ML Model Inference Time",
      "metrics": [
        ["FashionRecommender", "ModelInferenceTime", {"dimensions": {"ModelName": "ranking-model-v1"}}]
      ]
    },
    {
      "type": "metric",
      "title": "Cache Hit Rate",
      "metrics": [
        ["AWS/ElastiCache", "CacheHits", {"dimensions": {"CacheClusterId": "recommendations-cache"}}],
        ["...", "CacheMisses"]
      ]
    },
    {
      "type": "metric",
      "title": "DynamoDB Consumed Capacity",
      "metrics": [
        ["AWS/DynamoDB", "ConsumedReadCapacityUnits", {"dimensions": {"TableName": "UserProfiles"}}],
        ["...", "ConsumedWriteCapacityUnits"]
      ]
    },
    {
      "type": "metric",
      "title": "Data Pipeline Status",
      "metrics": [
        ["FashionRecommender", "PipelineRecordsProcessed", {"dimensions": {"Pipeline": "feature-engineering"}}],
        ["...", "PipelineDuration"]
      ]
    },
    {
      "type": "log",
      "title": "Recent Errors",
      "query": "fields @timestamp, @message | filter @message like /ERROR/ | stats count() by @message | sort count() desc"
    }
  ]
}
```

### Alerting Configuration

**Critical Alerts:**

```yaml
Alerts:

  # API Performance
  APILatencyHigh:
    Metric: RecommendationLatency
    Threshold: 
      p95: 250ms  # Alert if P95 > 250ms
      p99: 500ms  # Alert if P99 > 500ms
    EvaluationPeriods: 2
    DatapointsToAlarm: 2
    Action: Page on-call engineer, send Slack notification
    Runbook: "docs/runbooks/api-latency-high.md"

  APIErrorRate:
    Metric: AWS/ApiGateway/5XXError
    Threshold: 
      Rate: 1%  # Alert if error rate > 1%
    EvaluationPeriods: 1
    DatapointsToAlarm: 1
    Action: Page on-call engineer, trigger incident
    Runbook: "docs/runbooks/api-errors.md"

  # ML Model Issues
  ModelInferenceFailure:
    Metric: ModelInferenceErrorRate
    Threshold: 5%  # Alert if inference error rate > 5%
    EvaluationPeriods: 2
    Action: Page on-call engineer, disable model, fallback to v0
    Runbook: "docs/runbooks/model-inference-failure.md"

  # Infrastructure
  LambdaThrottling:
    Metric: AWS/Lambda/Throttles
    Threshold: >0
    EvaluationPeriods: 1
    Action: Auto-scale Lambda concurrency, page engineer
    Runbook: "docs/runbooks/lambda-throttling.md"

  # Data Pipeline
  DataPipelineFailure:
    Metric: AWS/Glue/JobStatus
    Threshold: FAILED
    EvaluationPeriods: 1
    Action: Page on-call engineer, alert data team
    Runbook: "docs/runbooks/data-pipeline-failure.md"

  # Cost Overruns
  DailyAWSCostHigh:
    Metric: EstimatedCharges
    Threshold: 
      Daily: $100  # Alert if daily spend > $100
      Weekly: $500
    EvaluationPeriods: 1
    Action: Slack alert to team, pause non-critical jobs
    Runbook: "docs/runbooks/cost-overrun.md"

  # Database Performance
  DynamoDBThrottling:
    Metric: UserErrors
    Threshold: >0
    EvaluationPeriods: 1
    Action: Alert database team, auto-scale if applicable
    Runbook: "docs/runbooks/dynamodb-throttling.md"
```

## Application Performance Monitoring (APM)

### API Gateway Metrics

**Key Metrics:**

```python
class APIGatewayMonitoring:
    """Monitor API Gateway performance"""
    
    @staticmethod
    def get_api_metrics():
        """
        Key API Gateway Metrics:
        
        1. Count: Total API requests
        2. 4XXError: Client errors (bad requests, auth failures)
        3. 5XXError: Server errors (lambda failures, timeouts)
        4. Latency: End-to-end request duration
        5. CacheDiffer: Cache miss detection
        """
        cloudwatch = boto3.client('cloudwatch')
        
        # Fetch metrics for past 1 hour
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/ApiGateway',
            MetricName='Count',
            Dimensions=[
                {'Name': 'ApiName', 'Value': 'fashion-recommender-api'},
                {'Name': 'Stage', 'Value': 'prod'}
            ],
            StartTime=datetime.utcnow() - timedelta(hours=1),
            EndTime=datetime.utcnow(),
            Period=60,
            Statistics=['Sum', 'Average', 'Maximum']
        )
        
        return response['Datapoints']
    
    @staticmethod
    def calculate_error_rate():
        """Calculate API error rate"""
        cloudwatch = boto3.client('cloudwatch')
        
        total = cloudwatch.get_metric_statistics(
            Namespace='AWS/ApiGateway',
            MetricName='Count',
            StartTime=datetime.utcnow() - timedelta(hours=1),
            EndTime=datetime.utcnow(),
            Period=3600,
            Statistics=['Sum']
        )
        
        errors = cloudwatch.get_metric_statistics(
            Namespace='AWS/ApiGateway',
            MetricName='5XXError',
            StartTime=datetime.utcnow() - timedelta(hours=1),
            EndTime=datetime.utcnow(),
            Period=3600,
            Statistics=['Sum']
        )
        
        total_count = total['Datapoints'][0]['Sum'] if total['Datapoints'] else 1
        error_count = errors['Datapoints'][0]['Sum'] if errors['Datapoints'] else 0
        
        return (error_count / total_count) * 100
```

### Lambda Performance Monitoring

**Lambda-Specific Metrics:**

```yaml
Lambda Metrics:

  Duration:
    Description: "Function execution time in milliseconds"
    Target: P95 < 100ms for recommendation functions
    Alert: > 200ms

  Errors:
    Description: "Number of function invocations that resulted in an error"
    Target: < 0.1%
    Alert: > 1%

  Throttles:
    Description: "Number of invocation requests throttled by Lambda"
    Target: 0
    Alert: > 0

  ConcurrentExecutions:
    Description: "Number of function instances currently active"
    Reserved: 100 (production should reserve capacity)
    Max: 1000 (configurable)

  UnreservedConcurrentExecutions:
    Description: "Available concurrent executions for burst"
    Target: > 50 reserved for production

  ProvisionedConcurrentExecutions:
    Description: "Pre-initialized execution environments"
    Target: 20 for recommendation-orchestrator (cold start optimization)
```

### SageMaker Model Monitoring

**Model Endpoint Metrics:**

```python
class SageMakerMonitoring:
    """Monitor SageMaker endpoint performance"""
    
    @staticmethod
    def setup_model_monitoring():
        """
        SageMaker Model Monitor Configuration
        
        Tracks:
        1. Model Performance: Accuracy, latency, throughput
        2. Data Quality: Input feature distribution drift
        3. Bias: Fairness across user segments
        4. Feature Importance: Which features influence predictions most
        """
        sagemaker = boto3.client('sagemaker')
        
        # Enable data capture for monitoring
        baseline_job_name = 'ranking-model-baseline'
        
        # Create baselining job to establish metrics baseline
        sagemaker.create_processing_job(
            ProcessingJobName=baseline_job_name,
            ProcessingInputs=[
                {
                    'InputName': 'input_data',
                    'S3Input': {
                        'S3Uri': 's3://fashion-recommender-data/training-data/',
                        'LocalPath': '/opt/ml/processing/input',
                        'S3DataType': 'S3Prefix',
                        'S3DataDistributionType': 'FullyReplicated',
                        'S3CompressionType': 'None'
                    }
                }
            ],
            ProcessingOutputConfig={
                'Outputs': [
                    {
                        'OutputName': 'baseline',
                        'S3Output': {
                            'S3Uri': 's3://fashion-recommender-data/baselines/',
                            'LocalPath': '/opt/ml/processing/output',
                            'S3UploadMode': 'EndOfJob'
                        }
                    }
                ]
            },
            ProcessingResources={
                'InstanceCount': 1,
                'InstanceType': 'ml.m5.xlarge',
                'VolumeSizeInGB': 100
            },
            RoleArn='arn:aws:iam::ACCOUNT:role/SageMakerRole'
        )
    
    @staticmethod
    def create_model_quality_monitor(endpoint_name: str):
        """Create ongoing model quality monitoring"""
        sagemaker = boto3.client('sagemaker')
        
        # Schedule daily model evaluation
        sagemaker.create_model_quality_job_definition(
            JobDefinitionName='ranking-model-quality-monitor',
            ModelQualityBaselineConfig={
                'BaseliningJobName': 'ranking-model-baseline'
            },
            ModelQualityAppSpecification={
                'ImageUri': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-model-monitor-analyzer:latest'
            },
            ModelQualityJobInput={
                'EndpointInput': {
                    'EndpointName': endpoint_name,
                    'LocalPath': '/opt/ml/processing/input'
                }
            },
            ModelQualityJobOutputConfig={
                'S3Output': {
                    'S3Uri': 's3://fashion-recommender-data/model-monitoring/'
                }
            }
        )
        
        # Create schedule
        sagemaker.create_monitoring_schedule(
            MonitoringScheduleName='ranking-model-daily-monitor',
            MonitoringScheduleConfig={
                'ScheduleExpression': 'cron(0 2 ? * * *)',  # 2 AM daily
                'MonitoringJobDefinitionName': 'ranking-model-quality-monitor'
            }
        )
```

## Data Quality Monitoring

### Glue Data Quality Framework

**Data Quality Rules:**

```python
class DataQualityMonitor:
    """Monitor data pipeline quality"""
    
    @staticmethod
    def setup_glue_data_quality_checks():
        """
        AWS Glue Data Quality Rules
        
        Checks completeness, accuracy, consistency
        """
        glue = boto3.client('glue')
        
        quality_rules = {
            # Completeness checks
            'user_id_not_null': 'ColumnValues > "user_id" where "user_id" is not null > 0.99',
            'article_id_not_null': 'ColumnValues > "article_id" where "article_id" is not null > 0.99',
            'timestamp_not_null': 'ColumnValues > "t_dat" where "t_dat" is not null > 0.99',
            
            # Uniqueness checks
            'user_id_uniqueness': 'ColumnValues > "user_id" > 0.95',  # 95% unique
            'article_id_valid': 'ColumnValues > "article_id" > 0.95',
            
            # Timeliness checks
            'recent_data': 'ColumnValues > "t_dat" where "t_dat" >= current_date() - interval 7 days > 0.80',
            
            # Accuracy/Distribution checks
            'price_in_range': 'ColumnValues > "price" where "price" between 5 and 500 > 0.99',
            'quantity_positive': 'ColumnValues > "quantity" where "quantity" > 0 > 0.99',
            
            # Consistency checks
            'no_duplicate_transactions': 'ColumnValues > "transaction_id" > 1.0'
        }
        
        return quality_rules
    
    @staticmethod
    def publish_data_quality_metrics():
        """Publish data quality results to CloudWatch"""
        cloudwatch = boto3.client('cloudwatch')
        
        quality_scores = {
            'completeness': 0.98,    # 98% of required fields present
            'accuracy': 0.97,         # 97% of values pass validation
            'consistency': 0.99,      # 99% of values are consistent
            'timeliness': 0.95        # 95% of data is recent (< 7 days)
        }
        
        metric_data = [
            {
                'MetricName': f'DataQuality_{metric_name.title()}',
                'Value': score,
                'Unit': 'Percent',
                'Dimensions': [
                    {'Name': 'Dataset', 'Value': 'transactions_train'},
                    {'Name': 'Environment', 'Value': 'production'}
                ]
            }
            for metric_name, score in quality_scores.items()
        ]
        
        cloudwatch.put_metric_data(
            Namespace='FashionRecommender',
            MetricData=metric_data
        )
```

## Deployment & CI/CD Pipeline

### GitHub Actions Workflow

**Main Deployment Workflow:**

```yaml
name: Deploy Fashion Recommender System

on:
  push:
    branches:
      - main
    paths:
      - 'src/**'
      - 'infrastructure/**'
      - '.github/workflows/deploy.yml'
  workflow_dispatch:

env:
  AWS_REGION: us-east-1
  AWS_ACCOUNT_ID: ${{ secrets.AWS_ACCOUNT_ID }}
  REGISTRY: ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-east-1.amazonaws.com

jobs:
  # Stage 1: Validation & Testing
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest boto3 moto
      
      - name: Run Unit Tests
        run: pytest tests/unit/ -v --cov=src
      
      - name: Validate Infrastructure
        run: |
          # Validate CloudFormation templates
          aws cloudformation validate-template \
            --template-body file://infrastructure/cloudformation/api-gateway.yaml
          aws cloudformation validate-template \
            --template-body file://infrastructure/cloudformation/lambda-functions.yaml
          aws cloudformation validate-template \
            --template-body file://infrastructure/cloudformation/databases.yaml

  # Stage 2: Data Validation
  data-validation:
    runs-on: ubuntu-latest
    needs: validate
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Validate Training Data
        run: |
          python scripts/validate_data.py \
            --data-path s3://fashion-recommender-data/raw/ \
            --output-report data-validation-report.json
      
      - name: Check Data Quality
        run: |
          aws glue start-job-run \
            --job-name data-quality-check \
            --arguments '{"--job_name": "data-quality-check"}'

  # Stage 3: Feature Engineering
  feature-engineering:
    runs-on: ubuntu-latest
    needs: data-validation
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Run Feature Engineering Pipeline
        run: |
          aws glue start-job-run \
            --job-name feature-engineering-pipeline \
            --arguments '{"--job_name": "feature-engineering-pipeline"}'
      
      - name: Wait for Pipeline Completion
        run: |
          python scripts/wait_for_glue_job.py \
            --job-name feature-engineering-pipeline \
            --timeout 3600

  # Stage 4: Model Training (Optional - only if model changed)
  model-training:
    runs-on: ubuntu-latest
    needs: feature-engineering
    if: contains(github.event.head_commit.modified, 'src/ml_layer/')
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Submit SageMaker Training Job
        run: |
          python scripts/submit_training_job.py \
            --job-name ranking-model-$(date +%Y%m%d-%H%M%S) \
            --instance-type ml.m5.xlarge \
            --instance-count 1
      
      - name: Wait for Training Completion
        run: |
          python scripts/wait_for_training_job.py \
            --job-name ranking-model-${{ github.sha }} \
            --timeout 7200

  # Stage 5: Model Evaluation
  model-evaluation:
    runs-on: ubuntu-latest
    needs: model-training
    if: success() && contains(github.event.head_commit.modified, 'src/ml_layer/')
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Evaluate Model Performance
        run: |
          python scripts/evaluate_model.py \
            --model-name ranking-model-v1 \
            --test-data s3://fashion-recommender-data/test-features/
      
      - name: Compare Against Baseline
        run: |
          python scripts/compare_models.py \
            --current-model ranking-model-v1 \
            --baseline-model ranking-model-v0 \
            --metrics accuracy,auc,latency
      
      - name: Quality Gate Check
        run: |
          python scripts/quality_gate.py \
            --model ranking-model-v1 \
            --min-accuracy 0.72 \
            --max-latency 50ms

  # Stage 6: Vector Database Update (Embedding Generation)
  embedding-update:
    runs-on: ubuntu-latest
    needs: [model-evaluation, feature-engineering]
    if: always() && needs.feature-engineering.result == 'success'
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Generate User/Item Embeddings
        run: |
          aws sagemaker create-processing-job \
            --processing-job-name embedding-generation-$(date +%s) \
            --processing-inputs file://infrastructure/sagemaker/embedding-config.json
      
      - name: Update Vector Database
        run: |
          python scripts/update_vector_db.py \
            --embeddings-path s3://fashion-recommender-data/embeddings/ \
            --opensearch-endpoint ${{ secrets.OPENSEARCH_ENDPOINT }}

  # Stage 7: Integration Testing
  integration-tests:
    runs-on: ubuntu-latest
    needs: embedding-update
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Test Dependencies
        run: pip install pytest-asyncio boto3 requests
      
      - name: Run Integration Tests
        env:
          API_ENDPOINT: https://api-staging.fashion-recommender.example.com
        run: pytest tests/integration/ -v --tb=short
      
      - name: Run Smoke Tests
        env:
          API_ENDPOINT: https://api-staging.fashion-recommender.example.com
        run: |
          python tests/smoke/test_recommendation_api.py \
            --endpoint ${{ env.API_ENDPOINT }} \
            --test-users 10

  # Stage 8: Blue-Green Deployment
  deploy-blue-green:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: success()
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Prepare Blue-Green Deployment
        run: |
          # Create new Lambda version (Green)
          aws lambda publish-version \
            --function-name recommendation-orchestrator \
            --description "Deployment ${{ github.sha }}"
          
          # Store new version
          echo "GREEN_VERSION=$(aws lambda describe-function | jq -r '.Version')" >> $GITHUB_ENV
      
      - name: Deploy Lambda Functions (Green)
        run: |
          # Update Lambda code
          aws lambda update-function-code \
            --function-name recommendation-orchestrator-green \
            --s3-bucket fashion-recommender-deployments \
            --s3-key lambda/${{ github.sha }}/recommendation-orchestrator.zip
          
          # Wait for deployment
          aws lambda wait function-updated \
            --function-name recommendation-orchestrator-green
      
      - name: Run Canary Tests (Green)
        env:
          CANARY_TRAFFIC_PERCENTAGE: 5
        run: |
          # Route 5% traffic to Green version
          python scripts/canary_deployment.py \
            --traffic-percentage ${{ env.CANARY_TRAFFIC_PERCENTAGE }} \
            --green-version ${{ env.GREEN_VERSION }}
          
          # Monitor for 5 minutes
          python scripts/monitor_canary.py \
            --duration 300 \
            --error-threshold 1.0
      
      - name: Promote to Production (Full Traffic)
        if: success()
        run: |
          # Route 100% traffic to Green
          python scripts/shift_traffic.py \
            --green-percentage 100 \
            --duration 600
      
      - name: Rollback on Failure
        if: failure()
        run: |
          # Route traffic back to Blue (old version)
          python scripts/shift_traffic.py \
            --green-percentage 0 \
            --duration 60
          
          # Notify team
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -d '{"text": "Deployment failed, rolled back to previous version"}'

  # Stage 9: Smoke Tests (Post-Deployment)
  smoke-tests:
    runs-on: ubuntu-latest
    needs: deploy-blue-green
    if: success()
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Run Post-Deployment Smoke Tests
        env:
          API_ENDPOINT: https://api.fashion-recommender.example.com
        run: |
          python tests/smoke/post_deployment_tests.py \
            --endpoint ${{ env.API_ENDPOINT }} \
            --iterations 100
      
      - name: Verify Metrics
        run: |
          python scripts/verify_deployment_health.py \
            --function-name recommendation-orchestrator \
            --duration 300

  # Stage 10: Notification
  notify-deployment:
    runs-on: ubuntu-latest
    needs: [deploy-blue-green, smoke-tests]
    if: always()
    steps:
      - name: Send Slack Notification
        run: |
          STATUS="${{ job.status }}"
          COMMIT_URL="https://github.com/${{ github.repository }}/commit/${{ github.sha }}"
          
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -H 'Content-Type: application/json' \
            -d "{
              \"text\": \"Deployment Complete\",
              \"blocks\": [
                {
                  \"type\": \"section\",
                  \"text\": {
                    \"type\": \"mrkdwn\",
                    \"text\": \"*Deployment Status:* $STATUS\n*Commit:* <$COMMIT_URL|${{ github.sha }}>\n*Author:* ${{ github.actor }}\"
                  }
                }
              ]
            }"
```

### Infrastructure as Code (CloudFormation)

**Lambda Deployment Template:**

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Fashion Recommender System - Lambda Functions'

Parameters:
  EnvironmentName:
    Type: String
    Default: prod
    AllowedValues: [dev, staging, prod]
  
  CodeBucket:
    Type: String
    Description: S3 bucket containing Lambda function code
  
  CodeVersion:
    Type: String
    Description: Version/commit hash of the code

Resources:

  # IAM Role for Lambda Functions
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole
      Policies:
        - PolicyName: LambdaPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:GetItem
                  - dynamodb:Query
                  - dynamodb:UpdateItem
                  - dynamodb:PutItem
                Resource: arn:aws:dynamodb:*:*:table/UserProfiles*
              
              - Effect: Allow
                Action:
                  - opensearchserverless:*
                Resource: arn:aws:opensearchserverless:*:*:collection/*
              
              - Effect: Allow
                Action:
                  - sagemaker:InvokeEndpoint
                Resource: arn:aws:sagemaker:*:*:endpoint/ranking-model-*
              
              - Effect: Allow
                Action:
                  - kms:Decrypt
                  - kms:GenerateDataKey
                Resource: arn:aws:kms:*:*:key/*
              
              - Effect: Allow
                Action:
                  - logs:CreateLogGroup
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource: arn:aws:logs:*:*:*
              
              - Effect: Allow
                Action:
                  - cloudwatch:PutMetricData
                Resource: '*'

  # Recommendation Orchestrator Function
  RecommendationOrchestratorFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: recommendation-orchestrator
      Runtime: python3.11
      Handler: index.lambda_handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Timeout: 30
      MemorySize: 1024
      ReservedConcurrentExecutions: 100
      EphemeralStorage:
        Size: 512
      
      Code:
        S3Bucket: !Ref CodeBucket
        S3Key: !Sub 'lambda/${CodeVersion}/recommendation-orchestrator.zip'
      
      Environment:
        Variables:
          ENVIRONMENT: !Ref EnvironmentName
          OPENSEARCH_ENDPOINT: !ImportValue OpenSearchEndpoint
          DYNAMODB_TABLE_PROFILES: UserProfiles
          DYNAMODB_TABLE_INTERACTIONS: UserInteractions
          SAGEMAKER_ENDPOINT_NAME: ranking-model-v1
          CACHE_ENDPOINT: !ImportValue ElastiCacheEndpoint
      
      VpcConfig:
        SecurityGroupIds:
          - !ImportValue LambdaSecurityGroup
        SubnetIds:
          - !ImportValue PrivateSubnet1
          - !ImportValue PrivateSubnet2
      
      Layers:
        - !Sub 'arn:aws:lambda:${AWS::Region}:${AWS::AccountId}:layer:fashion-recommender-dependencies:1'
      
      TracingConfig:
        Mode: Active

  # Lambda Function Version (Blue-Green Deployment)
  RecommendationOrchestratorVersion:
    Type: AWS::Lambda::Version
    Properties:
      FunctionName: !Ref RecommendationOrchestratorFunction
      Description: !Sub 'Version ${CodeVersion}'

  # Alias for Blue-Green Switching
  RecommendationOrchestratorAlias:
    Type: AWS::Lambda::Alias
    Properties:
      AliasName: live
      FunctionName: !Ref RecommendationOrchestratorFunction
      FunctionVersion: !GetAtt RecommendationOrchestratorVersion.Version
      RoutingConfig:
        AdditionalVersionWeights:
          - FunctionVersion: !GetAtt RecommendationOrchestratorVersion.Version
            FunctionWeight: 0

  # CloudWatch Log Group
  RecommendationLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub '/aws/lambda/recommendation-orchestrator'
      RetentionInDays: 30

  # Lambda Auto-Scaling
  LambdaConcurrencyScaling:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      PolicyName: LambdaConcurrencyScaling
      PolicyType: TargetTrackingScaling
      TargetTrackingScalingPolicyConfiguration:
        TargetValue: 0.7
        PredefinedMetricSpecification:
          PredefinedMetricType: LambdaProvisionedConcurrencyUtilization
        ScaleOutCooldown: 60
        ScaleInCooldown: 300

Outputs:
  FunctionArn:
    Description: ARN of the Lambda function
    Value: !GetAtt RecommendationOrchestratorFunction.Arn
    Export:
      Name: RecommendationFunctionArn
  
  FunctionVersion:
    Description: Version of deployed function
    Value: !Ref RecommendationOrchestratorVersion
    Export:
      Name: RecommendationFunctionVersion
```

## Security & Compliance

### Data Protection

**Encryption Strategy:**

```yaml
Encryption at Rest:
  S3 Data Lake:
    Enabled: true
    Algorithm: AES-256
    Key Management: AWS KMS
    KeyRotation: Annual
  
  DynamoDB:
    Enabled: true
    Algorithm: AES-256
    Key Management: AWS KMS
    PointInTimeRecovery: Enabled
  
  ElastiCache:
    Enabled: true
    Algorithm: AES-256
    Key Management: AWS KMS
  
  OpenSearch:
    Enabled: true
    Algorithm: AES-256
    Key Management: AWS KMS

Encryption in Transit:
  TLS Version: 1.3 minimum
  API Communications:
    - API Gateway ↔ Lambda: TLS 1.3
    - Lambda ↔ DynamoDB: TLS 1.3
    - Lambda ↔ SageMaker: TLS 1.3
  Certificate Management: AWS Certificate Manager
  Certificate Rotation: Automatic
```

### IAM Policies

**Least Privilege Configuration:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RecommendationAPIAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:123456789:table/UserProfiles",
        "arn:aws:dynamodb:us-east-1:123456789:table/UserInteractions"
      ],
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        }
      }
    },
    {
      "Sid": "VectorDatabaseAccess",
      "Effect": "Allow",
      "Action": [
        "opensearchserverless:APIAccessAll"
      ],
      "Resource": [
        "arn:aws:opensearchserverless:us-east-1:123456789:collection/recommendations-index"
      ]
    },
    {
      "Sid": "ModelInference",
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": [
        "arn:aws:sagemaker:us-east-1:123456789:endpoint/ranking-model-*"
      ]
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "cloudwatch:namespace": "FashionRecommender"
        }
      }
    }
  ]
}
```

### Audit Logging

**Compliance & Audit Trail:**

```python
class AuditLogging:
    """Comprehensive audit logging for compliance"""
    
    @staticmethod
    def log_api_request(request_id: str, user_id: str, endpoint: str, 
                       status_code: int, latency_ms: float):
        """Log all API requests for audit trail"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_id,
            'user_id': user_id,
            'endpoint': endpoint,
            'status_code': status_code,
            'latency_ms': latency_ms,
            'source_ip': get_source_ip(),
            'user_agent': get_user_agent()
        }
        
        # Write to CloudWatch Logs
        logger.info(json.dumps(audit_entry))
        
        # Also write to S3 for long-term audit storage
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket='fashion-recommender-audit',
            Key=f'api-logs/{datetime.now().strftime("%Y/%m/%d")}/{request_id}.json',
            Body=json.dumps(audit_entry)
        )
    
    @staticmethod
    def log_data_access(user_id: str, data_type: str, access_type: str):
        """Log all sensitive data access"""
        access_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'data_type': data_type,
            'access_type': access_type,
            'accessor': get_caller_identity(),
            'arn': get_caller_arn()
        }
        
        logger.info(json.dumps(access_entry))
```

## Cost Optimization

### Resource Optimization

**Cost Monitoring & Alerts:**

```python
class CostOptimization:
    """Monitor and optimize AWS costs"""
    
    @staticmethod
    def setup_cost_monitoring():
        """Setup cost tracking and alerts"""
        ce = boto3.client('ce')  # Cost Explorer
        
        # Create budget alert for daily costs
        budgets = boto3.client('budgets')
        budgets.create_budget(
            AccountId='123456789',
            Budget={
                'BudgetName': 'FashionRecommender-Daily',
                'BudgetLimit': {
                    'Amount': '100',
                    'Unit': 'USD'
                },
                'TimeUnit': 'DAILY',
                'BudgetType': 'COST',
                'CostFilters': {
                    'Service': [
                        'AWS Lambda',
                        'Amazon DynamoDB',
                        'Amazon OpenSearch Service',
                        'Amazon SageMaker'
                    ]
                }
            }
        )
    
    @staticmethod
    def analyze_cost_breakdown():
        """Get daily cost breakdown by service"""
        ce = boto3.client('ce')
        
        response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'End': datetime.now().strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Filter={
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon Elastic Compute Cloud - Compute',
                              'AWS Lambda',
                              'Amazon DynamoDB']
                }
            },
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                {'Type': 'DIMENSION', 'Key': 'PURCHASE_TYPE'}
            ]
        )
        
        return response
```

**Cost Optimization Strategies:**

```yaml
Lambda Optimization:
  - Use provisioned concurrency for predictable workloads
  - Adjust memory allocation (more memory = faster execution = lower cost)
  - Enable x86_64 architecture for cost efficiency
  - Use Lambda Insights for monitoring

DynamoDB Optimization:
  - Enable on-demand billing for variable workloads
  - Use global secondary indexes judiciously
  - Implement TTL for automatic data cleanup
  - Use point-in-time recovery selectively

OpenSearch Optimization:
  - Use warm storage tier for infrequently accessed data
  - Configure index lifecycle policies for automatic archiving
  - Right-size node types based on actual usage
  - Enable domain logging only for critical environments

SageMaker Optimization:
  - Use spot instances for training jobs (70% savings)
  - Batch inference instead of real-time endpoints
  - Use Autopilot for hyperparameter tuning efficiency
  - Regular model evaluation to avoid unnecessary retraining
```

## Incident Response

### Runbooks

**API Latency High:**

```markdown
# Runbook: API Latency High

## Severity: High
## Detection: P95 latency > 250ms

### Immediate Actions (0-5 min)
1. Check CloudWatch dashboard for recent changes
2. Verify Lambda concurrent execution limits
   - If throttled: Request increase or optimize code
3. Check DynamoDB read/write throttling
   - If throttled: Enable auto-scaling
4. Verify SageMaker endpoint health
   - If unhealthy: Route to previous version

### Investigation (5-15 min)
1. Review recent deployments
   - If recent: Consider rollback
2. Check data volume changes
   - If increased: Scale infrastructure
3. Analyze slow queries in CloudWatch Logs
4. Check cache hit rates
   - If low: Warm cache or increase TTL

### Resolution
- Scale Lambda concurrency
- Enable DynamoDB auto-scaling
- Optimize slow query paths
- Consider caching strategy improvements
```

**Model Inference Failure:**

```markdown
# Runbook: Model Inference Failure

## Severity: Critical
## Detection: Model error rate > 5%

### Immediate Actions (0-2 min)
1. Check SageMaker endpoint status
2. Verify model version is deployed
3. Check input data shape matches expectations
4. Switch to previous model version

### Investigation (2-10 min)
1. Review SageMaker CloudWatch logs
2. Check data quality issues
3. Verify preprocessing pipeline
4. Review recent model updates

### Resolution
- Redeploy previous model version
- Fix input data issues
- Retrain model with corrected data
- Test in staging before production redeployment
```

## Learning vs Production Considerations

### Learning Project Simplifications

**What's Simplified:**
- Single region deployment
- Basic monitoring dashboards (core metrics only)
- Development-grade alerting
- Minimal runbooks

**What Remains Production-Ready:**
- Proper encryption and IAM policies
- CloudWatch integration
- CI/CD pipeline structure
- Cost monitoring framework

### Production Scaling Considerations

**As Operational Complexity Increases:**

```
Users: 10K → 1M+
├── Monitoring: Add regional dashboards, anomaly detection
├── Deployment: Multi-region blue-green, canary analysis
├── Incident Response: 24/7 on-call rotations, detailed runbooks
└── Cost: Reserved instances, savings plans, RI marketplace

Throughput: 10 RPS → 1000+ RPS
├── Alerts: Add multi-metric correlation, ML-based anomaly detection
├── Dashboards: Add predictive scaling recommendations
├── SLAs: Implement SLO tracking and error budgets
└── Monitoring: Add distributed tracing (X-Ray)

Availability: 99.9% → 99.99% required
├── Deployment: Canary analysis, automatic rollback
├── Testing: Chaos engineering, failure injection
├── Recovery: Automated failover procedures
└── Monitoring: Real-time alerting, predictive incident detection
```

## Success Metrics

**Operational Excellence:**
- Deployment frequency: Multiple deployments per day
- Lead time for changes: < 1 hour from commit to production
- Mean time to recovery (MTTR): < 15 minutes
- Change failure rate: < 5%

**Reliability:**
- System availability: > 99.9%
- P99 API latency: < 200ms
- Error rate: < 0.1%
- Data loss: Zero (point-in-time recovery available)

**Cost Efficiency:**
- Cost per recommendation: < $0.01
- Infrastructure utilization: > 70%
- Reserved capacity utilization: > 85%
- Monthly cost trend: Stable or decreasing

**Security & Compliance:**
- Security incidents: Zero (undetected)
- Audit trail completeness: 100%
- Compliance violations: Zero
- Average incident detection time: < 5 minutes

This operations layer documentation provides both learning project foundations and production-ready patterns for monitoring, deployment, security, and operational excellence in AWS-native ML systems.
