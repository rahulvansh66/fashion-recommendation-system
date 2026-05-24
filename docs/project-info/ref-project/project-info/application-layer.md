# Application Layer: Serverless API & Business Logic

## Overview

**Architecture Philosophy:** Orchestrate ML layer services into user-facing REST APIs with high availability, low latency, and business rule customization while maintaining cost-efficiency through serverless compute patterns.

**Core Approach:** API Gateway + Lambda-based microservices implementing the 4-stage recommendation pipeline with multi-level caching, user interaction tracking, and business rules engine for personalization and experimentation.

**Key Innovation:** Decoupled recommendation logic (candidate generation, filtering, ranking, ordering) enables independent scaling and A/B testing while maintaining sub-200ms end-to-end latency through strategic caching and pre-computation.

## API Gateway Architecture

### Request Flow & Routing

**Design Pattern:** API Gateway as the centralized entry point with routing to Lambda functions based on resource paths and HTTP methods.

```
User Request → API Gateway → Authorization → Throttling → Route → Lambda Function → Response
                                                                        ↓
                                                          ML/Data Services Orchestration
```

**API Gateway Configuration:**

```yaml
RestApi:
  Name: "fashion-recommender-api"
  Description: "Serverless fashion recommendation API"
  
  Stage Configuration:
    Dev:
      ThrottleSettings:
        RateLimit: 100  # requests per second per IP
        BurstLimit: 200
      LoggingLevel: INFO
      CacheTTL: 300  # 5 minutes for non-personalized endpoints
      
    Prod:
      ThrottleSettings:
        RateLimit: 1000  # Support scale for learning project expansion
        BurstLimit: 2000
      LoggingLevel: ERROR
      CacheTTL: 60  # Shorter cache for freshness

  CORS Configuration:
    AllowedOrigins: ["https://fashion-recommender-ui.example.com"]
    AllowedMethods: [GET, POST, OPTIONS]
    AllowedHeaders: [Content-Type, Authorization, X-API-Key]
    ExposeHeaders: [X-RateLimit-Remaining, X-Cache-Status]
    MaxAge: 3600

  Authentication:
    Type: "AWS_IAM + API_KEY"
    Authorizers:
      - LambdaAuthorizer:
          FunctionName: "authorize-api-request"
          IdentitySource: "method.request.header.Authorization"
          CacheTTL: 300  # Cache authorization decisions
```

### API Endpoint Definitions

**Endpoint Structure:**

```yaml
Base URL: https://api.fashion-recommender.example.com/v1

Endpoints:

  # Core Recommendation API
  GET /recommendations/{user_id}
    Query Parameters:
      - limit: int (default: 20, max: 100)
      - filter_seen: boolean (default: true)
      - context: string (optional, JSON encoded)
      - experiment: string (optional, A/B test variant)
    Headers:
      - Authorization: Bearer <user_token>
      - X-Device-Type: mobile|web|app
      - X-Session-ID: <session_identifier>
    Response: 200 OK
      Body:
        {
          "recommendations": [
            {
              "article_id": "0748717001",
              "title": "Black Blazer",
              "category": "Blazers",
              "price": 49.99,
              "confidence_score": 0.87,
              "image_url": "https://...",
              "reason": "Based on your style preferences"
            }
          ],
          "request_id": "req-12345-abc",
          "served_from": "cache|realtime",
          "latency_ms": 42
        }

  # User Profile Management
  GET /users/{user_id}/profile
    Response: 200 OK
      Body:
        {
          "user_id": "user_123",
          "age_group": "25-34",
          "fashion_style": "Casual",
          "purchase_frequency": "Monthly",
          "favorite_categories": ["Dresses", "Shoes"],
          "price_sensitivity": "Mid-range",
          "seasonal_preferences": {
            "summer": ["Shorts", "T-shirts"],
            "winter": ["Coats", "Sweaters"]
          },
          "last_updated": "2026-05-24T14:30:00Z"
        }

  POST /users/{user_id}/interactions
    Body:
      {
        "interaction_type": "view|click|purchase|add_to_cart|remove_from_cart",
        "article_id": "0748717001",
        "timestamp": "2026-05-24T14:30:00Z",
        "session_id": "session-abc-123",
        "context": {
          "device": "mobile",
          "page": "product_detail",
          "referrer": "recommendations"
        }
      }
    Response: 202 Accepted
      Body:
        {
          "interaction_id": "int-456-def",
          "status": "queued_for_processing"
        }

  # Content Management
  GET /articles/{article_id}
    Response: 200 OK
      Body:
        {
          "article_id": "0748717001",
          "title": "Black Blazer",
          "description": "Professional black blazer for work and casual wear",
          "category": "Blazers",
          "subcategory": "Single-Breasted",
          "price": 49.99,
          "stock_status": "in_stock|low_stock|out_of_stock",
          "attributes": {
            "color": "Black",
            "size_range": ["XS", "S", "M", "L", "XL"],
            "material": "Polyester/Cotton",
            "fit": "Regular"
          },
          "images": ["url1", "url2"],
          "popularity_score": 0.82,
          "trending": false
        }

  # Analytics & Feedback
  POST /recommendations/{recommendation_id}/feedback
    Body:
      {
        "user_id": "user_123",
        "article_id": "0748717001",
        "feedback_type": "helpful|not_helpful|purchased|ignored",
        "rating": 4,
        "timestamp": "2026-05-24T15:00:00Z"
      }
    Response: 204 No Content

  # Search & Discovery
  GET /search
    Query Parameters:
      - q: string (search query)
      - category: string (filter by category)
      - price_min: float
      - price_max: float
      - limit: int (default: 20)
    Response: 200 OK
      Body:
        {
          "results": [
            {
              "article_id": "0748717001",
              "title": "Black Blazer",
              "relevance_score": 0.95,
              "category": "Blazers",
              "price": 49.99
            }
          ],
          "total_results": 42,
          "query_time_ms": 15
        }

  # Health Check
  GET /health
    Response: 200 OK
      Body:
        {
          "status": "healthy",
          "timestamp": "2026-05-24T14:30:00Z",
          "services": {
            "ml_layer": "healthy",
            "data_layer": "healthy",
            "cache": "healthy"
          }
        }
```

### Rate Limiting & Throttling

**Implementation Strategy:**

```python
# API Gateway Stage Throttle Settings (via CloudFormation/Terraform)
rate_limit_policy = {
    "RateLimit": 1000,      # Requests per second per user
    "BurstLimit": 2000,     # Allow temporary spikes
    "QuotaLimit": 10000,    # Total requests per day
}

# Lambda Authorizer for Fine-grained Control
def rate_limit_authorizer(event, context):
    user_id = extract_user_id(event)
    client = boto3.client('dynamodb')
    
    # Check user-specific rate limits
    response = client.get_item(
        TableName='RateLimitQuota',
        Key={'user_id': {'S': user_id}}
    )
    
    quota = response.get('Item', {})
    requests_today = int(quota.get('requests_today', {}).get('N', 0))
    
    if requests_today > 10000:  # Daily quota
        return {
            'principalId': user_id,
            'policyDocument': {
                'Version': '2012-10-17',
                'Statement': [
                    {
                        'Action': 'execute-api:Invoke',
                        'Effect': 'Deny',
                        'Resource': event['methodArn']
                    }
                ]
            }
        }
    
    return {
        'principalId': user_id,
        'policyDocument': {
            'Version': '2012-10-17',
            'Statement': [
                {
                    'Action': 'execute-api:Invoke',
                    'Effect': 'Allow',
                    'Resource': event['methodArn']
                }
            ]
        },
        'context': {
            'userRole': quota.get('user_role', 'free'),
            'remainingRequests': str(10000 - requests_today)
        }
    }
```

## Core Application Components

### 1. Recommendation API Service

**Primary Lambda Function: `recommendation-orchestrator`**

```python
import json
import boto3
import time
from typing import Dict, List
from datetime import datetime

opensearch_client = boto3.client('opensearchserverless')
dynamodb = boto3.client('dynamodb')
elasticache_client = boto3.client('elasticache')  # Via connection pool
sagemaker_runtime = boto3.client('sagemaker-runtime')

class RecommendationOrchestrator:
    """
    Orchestrates 4-stage recommendation pipeline:
    1. Candidate Generation (Vector DB similarity)
    2. Filtering (Remove seen items, check availability)
    3. Ranking (ML model scoring)
    4. Ordering (Business rules + personalization)
    """
    
    def __init__(self):
        self.cache = {}  # ElastiCache connection (simplified)
        self.openSearch_index = 'item-embeddings'
        self.ranking_endpoint = 'fashion-ranker-v1'
        
    def get_recommendations(self, user_id: str, limit: int = 20, 
                           context: Dict = None, experiment: str = None) -> Dict:
        """
        Main entry point for recommendation requests
        
        Target Latency:
        - P50: 50ms
        - P95: 150ms
        - P99: 200ms
        
        Breakdown:
        - Stage 1 (Candidate Gen): 30-50ms
        - Stage 2 (Filtering): 10-20ms
        - Stage 3 (Ranking): 40-60ms
        - Stage 4 (Ordering): 10-20ms
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        
        try:
            # Stage 1: Candidate Generation via Vector DB
            candidates = self._stage1_candidate_generation(user_id, top_k=100)
            
            # Stage 2: Filtering
            filtered = self._stage2_filtering(candidates, user_id)
            
            # Stage 3: Ranking
            ranked = self._stage3_ranking(filtered, user_id, context)
            
            # Stage 4: Ordering with Business Rules
            final = self._stage4_ordering(ranked, user_id, limit, experiment)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Track metrics
            self._record_metrics(request_id, elapsed_ms, len(final))
            
            return {
                'recommendations': final,
                'request_id': request_id,
                'served_from': 'realtime',
                'latency_ms': round(elapsed_ms, 2)
            }
            
        except Exception as e:
            return self._fallback_recommendations(user_id, request_id, str(e))
    
    def _stage1_candidate_generation(self, user_id: str, top_k: int = 100) -> List[Dict]:
        """
        Stage 1: Candidate Generation
        
        Approach: k-NN search on user embedding against item embeddings
        
        Implementation:
        - Get user embedding (cached if recent)
        - Query OpenSearch vector index for similar items
        - Fallback to popular items if user is new
        
        Latency Target: <50ms
        """
        # Get user embedding (check cache first)
        user_embedding = self._get_user_embedding(user_id)
        
        if user_embedding is None:
            # New user - return popular items
            return self._get_popular_items(top_k)
        
        # Query OpenSearch for k-NN
        query = {
            "size": top_k,
            "query": {
                "knn": {
                    "item_embedding": {
                        "vector": user_embedding,
                        "k": top_k
                    }
                }
            }
        }
        
        response = opensearch_client.search(
            body=json.dumps(query),
            index=self.openSearch_index
        )
        
        candidates = [
            {
                'article_id': hit['_source']['article_id'],
                'title': hit['_source']['title'],
                'similarity_score': hit['_score'],
                'category': hit['_source']['category']
            }
            for hit in response['hits']['hits']
        ]
        
        return candidates
    
    def _stage2_filtering(self, candidates: List[Dict], user_id: str) -> List[Dict]:
        """
        Stage 2: Filtering
        
        Removes:
        1. Items user has already seen/purchased
        2. Out-of-stock items
        3. Age-restricted items (if applicable)
        
        Approach:
        - ElastiCache for recent (last 30 days) interactions
        - DynamoDB for complete history
        - Check inventory in real-time
        
        Latency Target: <20ms
        """
        # Get user interaction history (cache + DB)
        seen_items = self._get_user_seen_items(user_id)
        
        # Filter out seen items
        filtered = [
            item for item in candidates
            if item['article_id'] not in seen_items
        ]
        
        # Check stock status
        available_items = []
        for item in filtered:
            stock_status = self._get_inventory_status(item['article_id'])
            if stock_status in ['in_stock', 'low_stock']:
                available_items.append(item)
        
        return available_items
    
    def _stage3_ranking(self, candidates: List[Dict], user_id: str, 
                       context: Dict = None) -> List[Dict]:
        """
        Stage 3: Ranking
        
        Scores candidates using ML ranking model
        
        Features per candidate:
        - User features: demographics, purchase history
        - Item features: category, price, popularity
        - Contextual features: time of day, device, session length
        - Interaction signals: CTR in similar category
        
        Model: XGBoost trained endpoint on SageMaker
        
        Latency Target: <60ms
        """
        if not candidates:
            return []
        
        # Build feature vectors for ranking
        ranking_features = []
        for candidate in candidates:
            features = self._build_ranking_features(user_id, candidate, context)
            ranking_features.append(features)
        
        # Batch inference via SageMaker endpoint
        batch_payload = json.dumps({
            'instances': ranking_features
        })
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=self.ranking_endpoint,
            ContentType='application/json',
            Body=batch_payload
        )
        
        predictions = json.loads(response['Body'].read())
        scores = predictions['predictions']
        
        # Attach scores to candidates
        for i, candidate in enumerate(candidates):
            candidate['ranking_score'] = scores[i]
        
        # Sort by score
        ranked = sorted(candidates, key=lambda x: x['ranking_score'], reverse=True)
        
        return ranked
    
    def _stage4_ordering(self, ranked: List[Dict], user_id: str, 
                        limit: int, experiment: str = None) -> List[Dict]:
        """
        Stage 4: Ordering + Business Logic
        
        Applies personalized ordering rules:
        1. Diversity constraints (don't show too many from same category)
        2. New arrivals promotion
        3. Inventory velocity (move slow-selling items)
        4. A/B testing variants
        5. User-specific preferences
        
        Latency Target: <20ms
        """
        # Get business rules for this user
        rules = self._get_user_business_rules(user_id, experiment)
        
        # Apply diversity constraint
        ordered = self._apply_diversity_constraint(ranked, rules['max_same_category'])
        
        # Apply new arrivals boost
        ordered = self._apply_new_arrivals_boost(ordered, rules['new_arrivals_ratio'])
        
        # Apply personalization rules
        ordered = self._apply_user_preferences(ordered, user_id, rules)
        
        # Enforce A/B test variant if applicable
        if experiment:
            ordered = self._apply_experiment_treatment(ordered, user_id, experiment)
        
        # Return top-k
        final = ordered[:limit]
        
        # Add metadata
        for i, item in enumerate(final):
            item['rank'] = i + 1
            item['reason'] = self._generate_recommendation_reason(item, user_id)
        
        return final
    
    def _generate_request_id(self) -> str:
        import uuid
        return str(uuid.uuid4())[:16]
    
    def _get_user_embedding(self, user_id: str) -> List[float]:
        """Get user embedding from cache or DynamoDB"""
        try:
            # Try cache first
            cache_key = f"user_emb:{user_id}"
            if cache_key in self.cache:
                return self.cache[cache_key]['embedding']
            
            # Check DynamoDB
            response = dynamodb.get_item(
                TableName='UserEmbeddings',
                Key={'user_id': {'S': user_id}}
            )
            
            if 'Item' in response:
                embedding = response['Item']['embedding']['BS']
                # Cache it
                self.cache[cache_key] = {
                    'embedding': embedding,
                    'timestamp': datetime.now()
                }
                return embedding
            
            return None
        except Exception as e:
            print(f"Error getting user embedding: {e}")
            return None
    
    def _get_user_seen_items(self, user_id: str) -> set:
        """Get items user has already interacted with"""
        try:
            # Recent interactions from ElastiCache (last 30 days)
            recent_key = f"interactions:recent:{user_id}"
            recent = self.cache.get(recent_key, set())
            
            # Full history from DynamoDB for older interactions
            response = dynamodb.query(
                TableName='UserInteractions',
                KeyConditionExpression='user_id = :uid',
                ExpressionAttributeValues={
                    ':uid': {'S': user_id}
                },
                ProjectionExpression='article_id'
            )
            
            all_seen = recent | set(
                item['article_id']['S'] for item in response.get('Items', [])
            )
            
            return all_seen
        except Exception as e:
            print(f"Error getting seen items: {e}")
            return set()
    
    def _get_inventory_status(self, article_id: str) -> str:
        """Check if item is in stock"""
        try:
            response = dynamodb.get_item(
                TableName='Inventory',
                Key={'article_id': {'S': article_id}}
            )
            
            if 'Item' in response:
                return response['Item']['stock_status']['S']
            
            return 'out_of_stock'
        except Exception as e:
            print(f"Error checking inventory: {e}")
            return 'out_of_stock'
    
    def _build_ranking_features(self, user_id: str, item: Dict, 
                               context: Dict = None) -> Dict:
        """Build feature vector for ranking model"""
        user_features = self._get_user_features(user_id)
        item_features = self._get_item_features(item['article_id'])
        
        features = {
            # User features
            'user_age': user_features.get('age_bucket', 0),
            'user_purchase_freq': user_features.get('purchase_frequency', 0),
            'user_avg_price': user_features.get('avg_price_range', 0),
            
            # Item features
            'item_price': item_features.get('price', 0),
            'item_popularity': item_features.get('popularity_score', 0),
            'item_stock_velocity': item_features.get('inventory_velocity', 0),
            
            # Contextual features
            'time_of_day': context.get('time_of_day', 12) if context else 12,
            'day_of_week': context.get('day_of_week', 3) if context else 3,
            
            # Interaction signals
            'similarity_score': item.get('similarity_score', 0)
        }
        
        return features
    
    def _apply_diversity_constraint(self, items: List[Dict], 
                                   max_same_category: int = 5) -> List[Dict]:
        """Ensure diversity across categories"""
        category_count = {}
        result = []
        
        for item in items:
            category = item['category']
            if category_count.get(category, 0) < max_same_category:
                result.append(item)
                category_count[category] = category_count.get(category, 0) + 1
        
        return result
    
    def _apply_new_arrivals_boost(self, items: List[Dict], 
                                 new_arrivals_ratio: float = 0.2) -> List[Dict]:
        """Promote new arrivals to top positions"""
        new_arrivals = [i for i in items if i.get('is_new_arrival', False)]
        existing = [i for i in items if not i.get('is_new_arrival', False)]
        
        # Calculate split
        num_new = int(len(items) * new_arrivals_ratio)
        
        # Interleave: new arrivals come up in ranking
        reordered = new_arrivals[:num_new] + existing
        
        return reordered
    
    def _apply_user_preferences(self, items: List[Dict], user_id: str, 
                               rules: Dict) -> List[Dict]:
        """Apply user-specific personalization preferences"""
        preferences = self._get_user_preferences(user_id)
        
        # Boost items matching preferences
        for item in items:
            if item['category'] in preferences.get('preferred_categories', []):
                item['ranking_score'] *= 1.1  # 10% boost
            
            if item.get('price', 0) in self._get_user_price_range(user_id):
                item['ranking_score'] *= 1.05  # 5% boost
        
        return sorted(items, key=lambda x: x['ranking_score'], reverse=True)
    
    def _apply_experiment_treatment(self, items: List[Dict], user_id: str, 
                                   experiment: str) -> List[Dict]:
        """Apply A/B test variant treatment"""
        # Get experiment configuration
        exp_config = self._get_experiment_config(experiment)
        
        if exp_config.get('treatment') == 'shuffle':
            import random
            random.shuffle(items)
        elif exp_config.get('treatment') == 'reverse_rank':
            items = items[::-1]
        
        return items
    
    def _generate_recommendation_reason(self, item: Dict, user_id: str) -> str:
        """Generate human-readable reason for recommendation"""
        reasons = [
            "Based on your style preferences",
            "Trending in your favorite category",
            "Similar to items you've liked",
            "Popular with customers like you",
            "New arrival you might like",
            "Matches your size preferences"
        ]
        
        import random
        return random.choice(reasons)
    
    def _record_metrics(self, request_id: str, latency_ms: float, 
                       num_recommendations: int):
        """Record CloudWatch metrics"""
        cloudwatch = boto3.client('cloudwatch')
        
        cloudwatch.put_metric_data(
            Namespace='FashionRecommender',
            MetricData=[
                {
                    'MetricName': 'RecommendationLatency',
                    'Value': latency_ms,
                    'Unit': 'Milliseconds'
                },
                {
                    'MetricName': 'RecommendationsReturned',
                    'Value': num_recommendations,
                    'Unit': 'Count'
                }
            ]
        )
    
    def _fallback_recommendations(self, user_id: str, request_id: str, 
                                 error: str) -> Dict:
        """Return pre-computed popular items on error"""
        popular = self._get_popular_items(20)
        
        return {
            'recommendations': popular,
            'request_id': request_id,
            'served_from': 'fallback',
            'error': error,
            'latency_ms': 0
        }
    
    # Helper methods (abbreviated for space)
    def _get_popular_items(self, limit: int) -> List[Dict]:
        pass
    
    def _get_user_features(self, user_id: str) -> Dict:
        pass
    
    def _get_item_features(self, article_id: str) -> Dict:
        pass
    
    def _get_user_preferences(self, user_id: str) -> Dict:
        pass
    
    def _get_user_price_range(self, user_id: str) -> List[float]:
        pass
    
    def _get_user_business_rules(self, user_id: str, experiment: str = None) -> Dict:
        pass
    
    def _get_experiment_config(self, experiment: str) -> Dict:
        pass

def lambda_handler(event, context):
    """AWS Lambda entry point"""
    orchestrator = RecommendationOrchestrator()
    
    # Extract request parameters
    user_id = event['pathParameters']['user_id']
    query_params = event.get('queryStringParameters', {}) or {}
    
    limit = int(query_params.get('limit', 20))
    filter_seen = query_params.get('filter_seen', 'true').lower() == 'true'
    context_param = json.loads(query_params.get('context', '{}'))
    experiment = query_params.get('experiment')
    
    # Get recommendations
    result = orchestrator.get_recommendations(user_id, limit, context_param, experiment)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result),
        'headers': {
            'Content-Type': 'application/json',
            'X-Request-ID': result['request_id'],
            'Cache-Control': 'private, max-age=60'
        }
    }
```

### 2. User Profile Management Service

**Lambda Function: `user-profile-manager`**

```python
import boto3
import json
from datetime import datetime, timedelta

dynamodb = boto3.client('dynamodb')

class UserProfileManager:
    """
    Manages user profiles including:
    - Profile creation and updates
    - Preference learning
    - Interaction tracking
    - User segmentation
    """
    
    def __init__(self):
        self.profiles_table = 'UserProfiles'
        self.interactions_table = 'UserInteractions'
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Retrieve complete user profile"""
        response = dynamodb.get_item(
            TableName=self.profiles_table,
            Key={'user_id': {'S': user_id}}
        )
        
        if 'Item' not in response:
            # Create new profile
            return self._create_default_profile(user_id)
        
        return self._deserialize_profile(response['Item'])
    
    def update_user_profile(self, user_id: str, profile_updates: Dict) -> Dict:
        """Update user profile with new data"""
        current_profile = self.get_user_profile(user_id)
        
        # Merge updates
        updated_profile = {**current_profile, **profile_updates}
        updated_profile['last_updated'] = datetime.now().isoformat()
        
        # Persist
        dynamodb.put_item(
            TableName=self.profiles_table,
            Item=self._serialize_profile(updated_profile)
        )
        
        return updated_profile
    
    def track_interaction(self, user_id: str, interaction: Dict) -> str:
        """Track user interaction (view, click, purchase, etc.)"""
        interaction_id = self._generate_interaction_id()
        
        item = {
            'interaction_id': {'S': interaction_id},
            'user_id': {'S': user_id},
            'article_id': {'S': interaction['article_id']},
            'interaction_type': {'S': interaction['interaction_type']},
            'timestamp': {'S': datetime.now().isoformat()},
            'ttl': {'N': str(int((datetime.now() + timedelta(days=90)).timestamp()))}
        }
        
        # Add optional fields
        if 'rating' in interaction:
            item['rating'] = {'N': str(interaction['rating'])}
        if 'context' in interaction:
            item['context'] = {'S': json.dumps(interaction['context'])}
        
        dynamodb.put_item(TableName=self.interactions_table, Item=item)
        
        # Update interaction counts in profile
        self._update_interaction_counts(user_id, interaction['interaction_type'])
        
        return interaction_id
    
    def _create_default_profile(self, user_id: str) -> Dict:
        """Create default profile for new user"""
        profile = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'age_group': 'unknown',
            'purchase_frequency': 0,
            'favorite_categories': [],
            'price_sensitivity': 'unknown',
            'interaction_count': 0,
            'total_spent': 0.0
        }
        
        # Persist
        dynamodb.put_item(
            TableName=self.profiles_table,
            Item=self._serialize_profile(profile)
        )
        
        return profile
    
    def _update_interaction_counts(self, user_id: str, interaction_type: str):
        """Increment interaction counts in profile"""
        dynamodb.update_item(
            TableName=self.profiles_table,
            Key={'user_id': {'S': user_id}},
            UpdateExpression='ADD interaction_count :inc SET last_updated = :now',
            ExpressionAttributeValues={
                ':inc': {'N': '1'},
                ':now': {'S': datetime.now().isoformat()}
            }
        )
    
    def _serialize_profile(self, profile: Dict) -> Dict:
        """Convert Python dict to DynamoDB format"""
        return {
            'user_id': {'S': profile['user_id']},
            'created_at': {'S': profile['created_at']},
            'last_updated': {'S': profile['last_updated']},
            'age_group': {'S': profile.get('age_group', 'unknown')},
            'purchase_frequency': {'N': str(profile.get('purchase_frequency', 0))},
            'favorite_categories': {'SS': profile.get('favorite_categories', [])},
            'price_sensitivity': {'S': profile.get('price_sensitivity', 'unknown')},
            'interaction_count': {'N': str(profile.get('interaction_count', 0))},
            'total_spent': {'N': str(profile.get('total_spent', 0.0))}
        }
    
    def _deserialize_profile(self, item: Dict) -> Dict:
        """Convert DynamoDB format to Python dict"""
        return {
            'user_id': item['user_id']['S'],
            'created_at': item['created_at']['S'],
            'last_updated': item['last_updated']['S'],
            'age_group': item.get('age_group', {}).get('S', 'unknown'),
            'purchase_frequency': int(item.get('purchase_frequency', {}).get('N', 0)),
            'favorite_categories': item.get('favorite_categories', {}).get('SS', []),
            'price_sensitivity': item.get('price_sensitivity', {}).get('S', 'unknown'),
            'interaction_count': int(item.get('interaction_count', {}).get('N', 0)),
            'total_spent': float(item.get('total_spent', {}).get('N', 0.0))
        }
    
    def _generate_interaction_id(self) -> str:
        import uuid
        return str(uuid.uuid4())[:16]
```

### 3. Business Rules Engine

**Lambda Function: `business-rules-engine`**

```python
import json
from datetime import datetime, timedelta
from typing import Dict, List

class BusinessRulesEngine:
    """
    Applies business logic rules to recommendations:
    - Inventory management (promote slow-moving items)
    - New arrivals promotion
    - Cross-sell/upsell logic
    - Category diversity
    - Seasonal adjustments
    - A/B testing framework
    """
    
    def __init__(self, config_path: str = 'business_rules.json'):
        self.rules = self._load_rules(config_path)
    
    def apply_ordering_rules(self, candidates: List[Dict], user_id: str, 
                            experiment: str = None) -> List[Dict]:
        """Apply all business logic rules in sequence"""
        
        # 1. Apply diversity constraint
        candidates = self._enforce_category_diversity(candidates)
        
        # 2. Promote new arrivals
        candidates = self._promote_new_arrivals(candidates)
        
        # 3. Apply inventory velocity rules
        candidates = self._apply_inventory_velocity(candidates)
        
        # 4. Apply user segment rules
        candidates = self._apply_user_segment_rules(candidates, user_id)
        
        # 5. Apply seasonal rules
        candidates = self._apply_seasonal_adjustments(candidates)
        
        # 6. Apply A/B test treatment if applicable
        if experiment:
            candidates = self._apply_experiment(candidates, experiment)
        
        return candidates
    
    def _enforce_category_diversity(self, candidates: List[Dict]) -> List[Dict]:
        """
        Constraint: Don't show >N items from same category
        
        Example: Max 5 from 'Shoes', max 4 from 'Blazers'
        """
        max_per_category = self.rules['diversity'].get('max_per_category', 5)
        category_count = {}
        result = []
        
        for item in candidates:
            category = item['category']
            if category_count.get(category, 0) < max_per_category:
                result.append(item)
                category_count[category] = category_count.get(category, 0) + 1
        
        return result
    
    def _promote_new_arrivals(self, candidates: List[Dict]) -> List[Dict]:
        """
        Business Rule: Promote new arrivals to increase visibility
        
        Logic:
        - Items added in last 7 days get boost
        - Appears in top 5 positions
        - Ratio: new arrivals should be 20% of top-10
        """
        new_arrivals = [
            i for i in candidates
            if self._is_new_arrival(i)
        ]
        existing = [
            i for i in candidates
            if not self._is_new_arrival(i)
        ]
        
        ratio = self.rules['promotions'].get('new_arrivals_ratio', 0.2)
        num_new = max(1, int(len(candidates) * ratio))
        
        # Reorder: new arrivals come first
        reordered = new_arrivals[:num_new] + existing
        
        return reordered
    
    def _apply_inventory_velocity(self, candidates: List[Dict]) -> List[Dict]:
        """
        Business Rule: Move slow-selling inventory
        
        Logic:
        - Items with high inventory but low sales get boost
        - Items with low inventory get slight penalty
        - Helps clear old stock
        """
        for item in candidates:
            velocity = item.get('inventory_velocity', 0.5)
            
            if velocity < 0.3:  # Slow-selling
                item['business_score_boost'] = 0.15  # 15% boost
            elif velocity > 0.8:  # Fast-selling
                item['business_score_boost'] = -0.05  # 5% penalty
            else:
                item['business_score_boost'] = 0.0
        
        # Re-rank with boost applied
        for item in candidates:
            item['final_score'] = (
                item.get('ranking_score', 0) * (1 + item.get('business_score_boost', 0))
            )
        
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)
    
    def _apply_user_segment_rules(self, candidates: List[Dict], 
                                 user_id: str) -> List[Dict]:
        """
        Business Rule: Apply rules based on user segment
        
        Segments:
        - VIP: Premium customers, show exclusive items first
        - High-value: Regular purchasers, show trending items
        - New: Recently joined, show popular/diverse items
        - Inactive: Haven't purchased in 30+ days, offer discounts/promotions
        """
        user_segment = self._determine_user_segment(user_id)
        
        if user_segment == 'vip':
            # VIP: Exclusive items first, premium pricing acceptable
            candidates = self._filter_by_exclusivity(candidates, 'exclusive')
            
        elif user_segment == 'high_value':
            # High-value: Boost trending items
            for item in candidates:
                if item.get('trending', False):
                    item['final_score'] = item.get('final_score', 0) * 1.1
            
        elif user_segment == 'new':
            # New users: Diverse, popular items
            candidates = self._enforce_diversity(candidates, strictness=1.5)
            
        elif user_segment == 'inactive':
            # Inactive: Show promotions, discounted items
            candidates = self._boost_promotional_items(candidates)
        
        return candidates
    
    def _apply_seasonal_adjustments(self, candidates: List[Dict]) -> List[Dict]:
        """
        Business Rule: Seasonal promotion logic
        
        Logic:
        - Season-appropriate items get boost
        - Off-season items get penalty
        - Clearance items at end of season
        """
        current_month = datetime.now().month
        season = self._get_season(current_month)
        
        season_boost = {
            'spring': ['Dresses', 'Sandals', 'Light Jackets'],
            'summer': ['Shorts', 'T-shirts', 'Sunglasses'],
            'fall': ['Sweaters', 'Boots', 'Blazers'],
            'winter': ['Coats', 'Scarves', 'Gloves']
        }
        
        boosted_categories = season_boost.get(season, [])
        
        for item in candidates:
            if item['category'] in boosted_categories:
                item['seasonal_boost'] = 0.1
            else:
                item['seasonal_boost'] = -0.05
            
            item['final_score'] = (
                item.get('final_score', 0) * (1 + item.get('seasonal_boost', 0))
            )
        
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)
    
    def _apply_experiment(self, candidates: List[Dict], 
                         experiment: str) -> List[Dict]:
        """
        A/B Testing: Apply experimental treatment
        
        Experiments can modify:
        - Ranking (reverse, shuffle, multi-armed bandit)
        - Diversity (increase/decrease)
        - Diversity (increase/decrease)
        - Filter logic (add/remove constraints)
        """
        exp_config = {
            'control': lambda x: x,
            'reverse_rank': lambda x: x[::-1],
            'shuffle': lambda x: self._shuffle(x),
            'boost_categories': lambda x: self._boost_specific_categories(x, ['Shoes']),
            'high_diversity': lambda x: self._enforce_diversity(x, strictness=2.0)
        }
        
        treatment = exp_config.get(experiment, lambda x: x)
        return treatment(candidates)
    
    def _is_new_arrival(self, item: Dict) -> bool:
        """Check if item was added recently"""
        date_added = datetime.fromisoformat(item.get('date_added', '2020-01-01'))
        days_old = (datetime.now() - date_added).days
        return days_old < 7
    
    def _determine_user_segment(self, user_id: str) -> str:
        """Determine user segment based on purchase history"""
        # Simplified - would query DynamoDB in real implementation
        return 'high_value'
    
    def _get_season(self, month: int) -> str:
        """Get current season"""
        if 3 <= month <= 5:
            return 'spring'
        elif 6 <= month <= 8:
            return 'summer'
        elif 9 <= month <= 11:
            return 'fall'
        else:
            return 'winter'
    
    def _load_rules(self, config_path: str) -> Dict:
        """Load business rules from configuration"""
        return {
            'diversity': {'max_per_category': 5},
            'promotions': {'new_arrivals_ratio': 0.2},
            'inventory': {'slow_velocity_threshold': 0.3}
        }
    
    def _shuffle(self, items: List[Dict]) -> List[Dict]:
        import random
        random.shuffle(items)
        return items
    
    def _filter_by_exclusivity(self, items: List[Dict], level: str) -> List[Dict]:
        return [i for i in items if i.get('exclusivity', 'standard') == level]
    
    def _enforce_diversity(self, items: List[Dict], strictness: float = 1.0) -> List[Dict]:
        return items  # Simplified
    
    def _boost_promotional_items(self, items: List[Dict]) -> List[Dict]:
        return items  # Simplified
    
    def _boost_specific_categories(self, items: List[Dict], categories: List[str]) -> List[Dict]:
        return items  # Simplified
```

## Multi-Level Caching Strategy

### Cache Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Request                          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│ L1: API Gateway Cache (15 min TTL)                      │
│ - Caches full recommendation responses                  │
│ - Key: user_id + query_params                           │
│ - Hit rate: ~30% (varies per user)                      │
└────────────────────┬────────────────────────────────────┘
                     │ (L1 Miss)
┌────────────────────▼────────────────────────────────────┐
│ L2: ElastiCache (Redis) - Hot Layer                     │
│ - Recent user interactions (30 days)                    │
│ - User embeddings (current)                             │
│ - Popular items ranking                                │
│ TTL: 1 hour to 30 days                                  │
│ Hit rate: ~60-70%                                       │
└────────────────────┬────────────────────────────────────┘
                     │ (L2 Miss)
┌────────────────────▼────────────────────────────────────┐
│ L3: DynamoDB - Warm Layer                               │
│ - Complete user interaction history                     │
│ - User profiles                                         │
│ - Item metadata                                         │
│ - Pre-computed recommendations                          │
└────────────────────┬────────────────────────────────────┘
                     │ (L3 Miss)
┌────────────────────▼────────────────────────────────────┐
│ L4: OpenSearch Vector DB - Vector Storage               │
│ - User/item embeddings for similarity search            │
│ - ANN indices for fast k-NN queries                     │
└────────────────────┬────────────────────────────────────┘
                     │ (L4 Miss - Full Computation)
┌────────────────────▼────────────────────────────────────┐
│ Full Pipeline Execution                                 │
│ - Candidate Generation (if no cache)                    │
│ - Filtering                                             │
│ - Ranking via SageMaker                                 │
│ - Business Rules                                        │
└─────────────────────────────────────────────────────────┘
```

### Cache Key Patterns

```python
# Cache keys for different data types

# Recommendation cache
f"rec:{user_id}:{limit}:{experiment}"           # Full recommendation
f"rec_candidates:{user_id}"                      # Candidate generation stage
f"rec_filtered:{user_id}"                        # After filtering stage
f"rec_ranked:{user_id}"                          # After ranking stage

# User data cache
f"user_emb:{user_id}"                            # User embedding
f"user_profile:{user_id}"                        # User profile
f"user_interactions:recent:{user_id}"            # Last 30 days interactions
f"user_interactions:all:{user_id}"               # Complete history

# Item data cache
f"item_features:{article_id}"                    # Item metadata
f"item_stock:{article_id}"                       # Inventory status
f"popular_items:{category}:{period}"             # Popular items per category

# ML model cache
f"ranking_model:v1:schema"                       # Model input schema
f"embedding_model:v1:config"                     # Model configuration
```

### Cache Invalidation Strategies

```python
class CacheInvalidationStrategy:
    """
    Cache invalidation for consistency
    """
    
    def invalidate_user_profile(self, user_id: str):
        """Invalidate when user profile changes"""
        keys_to_delete = [
            f"user_profile:{user_id}",
            f"rec:{user_id}:*",  # Also invalidate recommendations
            f"rec_candidates:{user_id}"
        ]
        self._delete_cache_keys(keys_to_delete)
    
    def invalidate_user_interactions(self, user_id: str):
        """Invalidate after interaction (purchase, view, etc.)"""
        keys_to_delete = [
            f"user_interactions:recent:{user_id}",
            f"rec:{user_id}:*"
        ]
        self._delete_cache_keys(keys_to_delete)
    
    def invalidate_item_data(self, article_id: str):
        """Invalidate when item data changes"""
        keys_to_delete = [
            f"item_features:{article_id}",
            f"item_stock:{article_id}",
            f"popular_items:*"  # Invalidate all popular items
        ]
        self._delete_cache_keys(keys_to_delete)
    
    def invalidate_model_version(self, model_name: str, version: str):
        """Invalidate when ML model is updated"""
        keys_to_delete = [
            f"*:{model_name}:{version}:*",
            f"rec:*"  # Invalidate all recommendations
        ]
        self._delete_cache_keys(keys_to_delete)
    
    def _delete_cache_keys(self, patterns: List[str]):
        """Delete cache keys matching patterns"""
        redis_client = self._get_redis_connection()
        
        for pattern in patterns:
            if '*' in pattern:
                # Scan for matching keys
                for key in redis_client.scan_iter(match=pattern):
                    redis_client.delete(key)
            else:
                redis_client.delete(pattern)
```

## User Interaction Tracking

### Event Collection & Processing

```python
class InteractionTracker:
    """
    Tracks user interactions with real-time and batch processing
    """
    
    def __init__(self):
        self.sqs = boto3.client('sqs')
        self.queue_url = 'https://sqs.us-east-1.amazonaws.com/123/interactions-queue'
    
    def track_interaction(self, user_id: str, interaction: Dict):
        """
        Track interaction asynchronously
        
        Interaction types:
        - view: Item viewed
        - click: Item clicked
        - add_to_cart: Added to shopping cart
        - remove_from_cart: Removed from cart
        - purchase: Item purchased
        - return: Item returned
        - rate: User rated item
        """
        message = {
            'user_id': user_id,
            'interaction_type': interaction['interaction_type'],
            'article_id': interaction['article_id'],
            'timestamp': datetime.now().isoformat(),
            'session_id': interaction.get('session_id'),
            'context': interaction.get('context', {}),
            'value': interaction.get('value')  # Price, rating, quantity
        }
        
        # Send to SQS for async processing
        self.sqs.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(message)
        )
        
        # Also update user profile in real-time
        self._update_user_profile_realtime(user_id, interaction)
    
    def _update_user_profile_realtime(self, user_id: str, interaction: Dict):
        """Update user profile with immediate feedback"""
        dynamodb = boto3.client('dynamodb')
        
        # Increment interaction counts
        dynamodb.update_item(
            TableName='UserProfiles',
            Key={'user_id': {'S': user_id}},
            UpdateExpression=(
                'ADD interaction_count :inc, '
                f'{interaction["interaction_type"]}_count :inc '
                'SET last_activity = :now'
            ),
            ExpressionAttributeValues={
                ':inc': {'N': '1'},
                ':now': {'S': datetime.now().isoformat()}
            }
        )

# SQS Handler for batch processing interactions
def process_interactions_batch(event, context):
    """Process interaction batch from SQS"""
    for record in event['Records']:
        interaction = json.loads(record['body'])
        
        # Process interaction:
        # 1. Update user profile
        # 2. Update item popularity
        # 3. Update embeddings if needed
        # 4. Store for model training
        
        # Update cached interaction history
        cache.append_recent_interaction(
            interaction['user_id'],
            interaction['article_id'],
            interaction['interaction_type']
        )
```

## Performance Optimization

### Latency Breakdown & Optimization

**Current Targets:**
- P50: 50ms
- P95: 150ms
- P99: 200ms

**Latency Breakdown:**

```
50-100ms  Stage 1: Candidate Generation (OpenSearch k-NN)
10-20ms   Stage 2: Filtering (DynamoDB + Cache)
40-60ms   Stage 3: Ranking (SageMaker Batch Transform)
10-20ms   Stage 4: Ordering (Business Rules)
────────────────────────────────────
110-200ms TOTAL (realistic end-to-end)
```

**Optimization Strategies:**

```python
class PerformanceOptimization:
    """Performance tuning techniques"""
    
    @staticmethod
    def parallelize_stages():
        """
        Run stages in parallel where possible instead of sequential
        
        Example: While waiting for ranking model, pre-fetch item details
        """
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # These can run in parallel
            future_candidates = executor.submit(stage1_candidate_generation)
            future_items = executor.submit(fetch_item_features)
            future_user = executor.submit(get_user_features)
            
            candidates = future_candidates.result()
            items = future_items.result()
            user = future_user.result()
    
    @staticmethod
    def batch_predictions():
        """
        Batch multiple ranking requests to SageMaker
        
        Instead of: 1 request per recommendation session
        Do: Batch 20-50 requests in single invoke
        """
        pass
    
    @staticmethod
    def use_connection_pooling():
        """
        Reuse connections to avoid connection overhead
        """
        # Lambda initialization (outside handler)
        opensearch_connection = boto3.client('opensearchserverless')
        redis_connection = redis.StrictRedis(connection_pool=pool)
    
    @staticmethod
    def implement_circuit_breaker():
        """
        Fail fast if downstream services are slow
        """
        pass
```

## Security & Authentication

### API Security

```yaml
Authentication:
  Type: "IAM + API Key"
  
  # IAM for service-to-service
  IAM_Roles:
    - RecommendationAPIRole
    - LambdaExecutionRole
    - SageMakerInvokeRole
  
  # API Key for client applications
  API_Keys:
    - Mobile App
    - Web Frontend
    - Partner Integrations
  
  Authorization:
    - Verify user_id matches authenticated user
    - Rate limit by user_id
    - Log all API access

Rate Limiting:
  By_User_ID: 1000 requests/day
  By_IP: 10000 requests/day
  Burst: 100 requests/minute per user

Data Protection:
  Encryption_In_Transit: TLS 1.3
  Encryption_At_Rest: KMS encryption for all data
  Field_Level_Encryption: User IDs and recommendations
```

## Learning vs Production Considerations

### Learning Project Simplifications

**What's Simplified:**
- Single Lambda deployment (no auto-scaling config)
- Basic error handling (rely on Lambda retries)
- Minimal business rules (core logic only)
- Development-grade database capacity

**What Remains Production-Ready:**
- Proper API Gateway configuration
- Cache invalidation logic
- Stage separation (dev/prod)
- Monitoring integration

### Production Scaling Considerations

**As Scale Increases:**

```
Users: 10K → 1M+
- Add Lambda concurrency reservations
- Implement request queuing with SQS
- Add Application Load Balancer for request distribution

Throughput: 10 RPS → 1000+ RPS
- Batch ranking inference (10-20 requests per invoke)
- Cache pre-warming strategy
- Multi-region deployments

Latency: 200ms → <100ms required
- Add regional caching
- Implement streaming responses
- Use synchronous OpenSearch indices

Complexity: Simple rules → Complex personalization
- A/B testing framework (already in place)
- Multi-model ensemble ranking
- Real-time feature computation
```

## Error Handling & Resilience

### Fallback Strategies

```python
def get_recommendations_with_fallback(user_id: str, limit: int = 20) -> Dict:
    """
    Three-tier fallback strategy
    """
    try:
        # Attempt full pipeline
        return full_recommendation_pipeline(user_id, limit)
    
    except StageGenerationError:
        # Stage 1 failed: Return pre-computed recommendations
        return get_precomputed_recommendations(user_id, limit)
    
    except RankingModelError:
        # Stage 3 failed: Skip ranking, use filtering scores
        return recommendations_without_ranking(user_id, limit)
    
    except Exception as e:
        # Complete failure: Return popular items
        print(f"Recommendation failed: {e}")
        return get_popular_items(limit)
```

## Success Metrics

**Recommendations Quality:**
- CTR (Click-Through Rate): Target 15%
- Conversion Rate: Target 3-5%
- Diversity Score: >0.8 (different categories)

**API Performance:**
- P95 Latency: <150ms
- Availability: >99.9%
- Error Rate: <0.1%

**Business Impact:**
- Recommendations contribution to revenue: 20%+
- New user onboarding: Cold-start recommendations within 24 hours
- Inventory health: Slow-movers sold 10%+ faster

This application layer documentation provides both the learning project foundation and production-ready patterns for serverless recommendation APIs.