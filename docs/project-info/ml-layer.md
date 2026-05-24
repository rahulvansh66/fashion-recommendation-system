# ML Layer: Embedding-Based Hybrid Recommendation System

## Overview

**Architecture Philosophy:** Separate batch learning from real-time serving using embeddings as the bridge between offline model training and online recommendation generation.

**Core Approach:** Two-tower neural network generating user and item embeddings, stored in vector databases for fast similarity search, orchestrated through a 4-stage recommendation pipeline optimized for sub-200ms latency.

**Key Innovation:** Hybrid embedding system combining collaborative filtering signals with content-based features, enabling both personalization and cold-start handling in a unified architecture.

## Two-Tower Neural Architecture

### Model Design Philosophy

**Architectural Pattern:** Dual encoder approach where user and item features are processed through separate neural networks that output embeddings in a shared vector space.

**Core Insight:** By learning shared representations, we can efficiently compute user-item similarities while maintaining the flexibility to incorporate rich feature sets for both users and items.

### Network Architecture

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class TwoTowerRecommender(tf.keras.Model):
    def __init__(self, 
                 embedding_dim=256,
                 user_vocab_sizes=None,
                 item_vocab_sizes=None,
                 hidden_units=[512, 256, 128]):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # User Tower Components
        self.user_categorical_embeddings = {}
        for feature_name, vocab_size in user_vocab_sizes.items():
            self.user_categorical_embeddings[feature_name] = layers.Embedding(
                input_dim=vocab_size,
                output_dim=min(64, int(vocab_size**0.25 * 4))  # Embedding size heuristic
            )
        
        self.user_dense_layers = [
            layers.Dense(units, activation='relu', name=f'user_dense_{i}')
            for i, units in enumerate(hidden_units)
        ]
        
        self.user_output_layer = layers.Dense(
            embedding_dim, 
            activation='l2_normalize',  # L2 normalization for cosine similarity
            name='user_embedding'
        )
        
        # Item Tower Components
        self.item_categorical_embeddings = {}
        for feature_name, vocab_size in item_vocab_sizes.items():
            self.item_categorical_embeddings[feature_name] = layers.Embedding(
                input_dim=vocab_size,
                output_dim=min(64, int(vocab_size**0.25 * 4))
            )
        
        self.item_dense_layers = [
            layers.Dense(units, activation='relu', name=f'item_dense_{i}')
            for i, units in enumerate(hidden_units)
        ]
        
        self.item_output_layer = layers.Dense(
            embedding_dim,
            activation='l2_normalize',
            name='item_embedding'
        )
        
        # Dropout for regularization
        self.dropout = layers.Dropout(0.3)
        
    def call(self, inputs, training=None):
        user_features = inputs['user_features']
        item_features = inputs['item_features']
        
        # User Tower Forward Pass
        user_embeddings = []
        for feature_name, values in user_features.items():
            if feature_name in self.user_categorical_embeddings:
                emb = self.user_categorical_embeddings[feature_name](values)
                user_embeddings.append(layers.Flatten()(emb))
            else:
                user_embeddings.append(tf.expand_dims(values, -1))
        
        user_concat = layers.Concatenate()(user_embeddings)
        
        user_hidden = user_concat
        for dense_layer in self.user_dense_layers:
            user_hidden = dense_layer(user_hidden)
            user_hidden = self.dropout(user_hidden, training=training)
        
        user_embedding = self.user_output_layer(user_hidden)
        
        # Item Tower Forward Pass
        item_embeddings = []
        for feature_name, values in item_features.items():
            if feature_name in self.item_categorical_embeddings:
                emb = self.item_categorical_embeddings[feature_name](values)
                item_embeddings.append(layers.Flatten()(emb))
            else:
                item_embeddings.append(tf.expand_dims(values, -1))
        
        item_concat = layers.Concatenate()(item_embeddings)
        
        item_hidden = item_concat
        for dense_layer in self.item_dense_layers:
            item_hidden = dense_layer(item_hidden)
            item_hidden = self.dropout(item_hidden, training=training)
        
        item_embedding = self.item_output_layer(item_hidden)
        
        return {
            'user_embedding': user_embedding,
            'item_embedding': item_embedding,
            'similarity_score': tf.reduce_sum(user_embedding * item_embedding, axis=1)
        }

# Training Objective: Maximize similarity for positive pairs, minimize for negatives
def contrastive_loss(y_true, similarity_scores, margin=1.0, temperature=0.1):
    """
    Contrastive loss for two-tower training with temperature scaling
    """
    # Scale similarities by temperature for better gradient flow
    scaled_scores = similarity_scores / temperature
    
    # Binary cross-entropy loss with temperature scaling
    positive_scores = tf.boolean_mask(scaled_scores, tf.cast(y_true, tf.bool))
    negative_scores = tf.boolean_mask(scaled_scores, tf.logical_not(tf.cast(y_true, tf.bool)))
    
    positive_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(positive_scores), logits=positive_scores))
    negative_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(negative_scores), logits=negative_scores))
    
    return positive_loss + negative_loss
```

### Training Configuration

**Optimization Strategy:**
```python
# SageMaker Training Configuration
training_config = {
    "instance_type": "ml.m5.xlarge",  # CPU-based for cost efficiency
    "instance_count": 1,
    "framework_version": "2.8.0",
    "python_version": "py39",
    "training_parameters": {
        "epochs": 50,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "embedding_dim": 256,
        "hidden_units": [512, 256, 128],
        "dropout_rate": 0.3,
        "l2_regularization": 0.001,
        "negative_sampling_rate": 4,  # 4 negative samples per positive
        "temperature": 0.1
    }
}
```

## 4-Stage Recommendation Pipeline

### Pipeline Architecture Overview

The 4-stage pipeline balances computational efficiency with recommendation quality by progressively narrowing candidates through specialized operations.

**Core Insight:** Each stage optimizes for different constraints:
1. **Generation:** Computational efficiency (similarity search)
2. **Filtering:** Business logic (availability, history)
3. **Ranking:** Prediction accuracy (purchase probability)
4. **Ordering:** User experience (diversity, freshness)

### Stage 1: Candidate Generation

**Purpose:** Efficiently identify ~100 potentially relevant items from the full catalog using embedding similarity.

**Implementation Details:**

```python
# OpenSearch Vector Database Configuration
opensearch_config = {
    "cluster": {
        "instance_type": "t3.small.search",  # Learning project sizing
        "instance_count": 1,
        "dedicated_master": False,
        "zone_awareness": False
    },
    "index_settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "knn": {
            "space_type": "cosinesimil",
            "engine": "nmslib",  # Fastest for cosine similarity
            "method": {
                "name": "hnsw",  # Hierarchical NSW for ANN search
                "parameters": {
                    "ef_construction": 200,  # Build-time search depth
                    "m": 16  # Max connections per node
                }
            }
        }
    }
}

class CandidateGenerator:
    def __init__(self, opensearch_client, embedding_service):
        self.opensearch_client = opensearch_client
        self.embedding_service = embedding_service
    
    async def generate_candidates(self, user_id: str, top_k: int = 100) -> List[str]:
        """
        Stage 1: Generate candidate items using vector similarity search
        """
        # Retrieve user embedding from cache or compute on-demand
        user_embedding = await self.embedding_service.get_user_embedding(user_id)
        
        if user_embedding is None:
            # Cold start fallback: popular items
            return await self._get_popular_items(top_k)
        
        # Vector similarity search in OpenSearch
        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "item_embedding": {
                        "vector": user_embedding.tolist(),
                        "k": top_k,
                        "ef_search": 200  # Runtime search depth
                    }
                }
            },
            "_source": ["article_id", "category", "price", "availability"]
        }
        
        response = await self.opensearch_client.search(
            index="fashion_items",
            body=search_body
        )
        
        candidates = [
            hit["_source"]["article_id"] 
            for hit in response["hits"]["hits"]
        ]
        
        return candidates
    
    async def _get_popular_items(self, top_k: int) -> List[str]:
        """Fallback for cold start users"""
        # Query pre-computed popularity scores from DynamoDB
        pass  # Implementation details...
```

**Performance Characteristics:**
- **Target Latency:** <50ms for similarity search
- **Accuracy:** 95%+ recall compared to exact search
- **Scalability:** O(log N) search complexity with HNSW index
- **Memory Efficiency:** Quantized embeddings reduce RAM usage by 75%

**Alternative Approaches Analysis:**

| Approach | Latency | Accuracy | Memory Usage | When to Use |
|----------|---------|----------|--------------|-------------|
| **ANN Index (Our Choice)** | <50ms | 95%+ | Low | >10K items, <100ms requirement |
| **Exact Similarity** | 200-500ms | 100% | High | <1K items, accuracy critical |
| **Matrix Factorization** | <10ms | 80% | Very High | Static catalogs, batch OK |
| **Category Filtering** | <5ms | 60% | Minimal | Simple catalogs, cold start |

### Stage 2: Filtering

**Purpose:** Remove irrelevant candidates based on user history and business constraints.

**Implementation Strategy:**

```python
class CandidateFilter:
    def __init__(self, dynamodb_client, elasticache_client):
        self.dynamodb = dynamodb_client
        self.elasticache = elasticache_client
    
    async def filter_candidates(self, candidates: List[str], user_id: str) -> List[str]:
        """
        Stage 2: Filter candidates based on user history and business rules
        """
        # Parallel data retrieval for performance
        user_history, inventory_status = await asyncio.gather(
            self._get_user_interaction_history(user_id),
            self._check_inventory_availability(candidates)
        )
        
        filtered_candidates = []
        
        for item_id in candidates:
            # Skip already purchased items (last 30 days)
            if item_id in user_history.get('purchased_last_30d', set()):
                continue
                
            # Skip out-of-stock items
            if not inventory_status.get(item_id, {}).get('available', False):
                continue
            
            # Apply user-specific filters (age restrictions, etc.)
            if not self._passes_user_constraints(item_id, user_id):
                continue
                
            filtered_candidates.append(item_id)
        
        return filtered_candidates[:50]  # Limit for ranking stage
    
    async def _get_user_interaction_history(self, user_id: str) -> dict:
        """
        Hybrid caching strategy: ElastiCache (hot) + DynamoDB (complete)
        """
        # Try ElastiCache first (last 30 days)
        cache_key = f"user_history:{user_id}"
        cached_history = await self.elasticache.get(cache_key)
        
        if cached_history:
            return json.loads(cached_history)
        
        # Fallback to DynamoDB for complete history
        response = await self.dynamodb.query(
            TableName="user_interactions",
            KeyConditionExpression="user_id = :user_id",
            FilterExpression="interaction_date > :cutoff_date",
            ExpressionAttributeValues={
                ":user_id": user_id,
                ":cutoff_date": datetime.now() - timedelta(days=30)
            }
        )
        
        history = self._process_interaction_history(response['Items'])
        
        # Cache for future requests (TTL: 1 hour)
        await self.elasticache.setex(
            cache_key, 3600, json.dumps(history)
        )
        
        return history
```

**Filtering Logic Priorities:**
1. **Immediate Relevance:** Remove recent purchases (avoid fatigue)
2. **Inventory Constraints:** Only show available items
3. **User Preferences:** Respect explicit filtering preferences
4. **Business Rules:** Age restrictions, regional availability
5. **Quality Thresholds:** Minimum rating, return rate limits

### Stage 3: Ranking

**Purpose:** Score remaining candidates using rich features to predict purchase probability.

**Model Architecture Choice:**

For the ranking stage, we implement both XGBoost and Neural Network approaches, allowing A/B testing to determine optimal performance:

```python
# XGBoost Ranking Model (Primary choice for tabular features)
import xgboost as xgb
from sagemaker.xgboost import XGBoost

class XGBoostRanker:
    def __init__(self, model_endpoint_name: str):
        self.endpoint_name = model_endpoint_name
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    async def rank_candidates(self, candidates: List[str], user_id: str, 
                             context: dict) -> List[Tuple[str, float]]:
        """
        Stage 3: Rank candidates using XGBoost model
        """
        # Feature engineering
        features = await self._extract_ranking_features(
            candidates, user_id, context
        )
        
        # Batch prediction for efficiency
        prediction_input = self._format_for_inference(features)
        
        response = await self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='text/csv',
            Body=prediction_input
        )
        
        purchase_probabilities = self._parse_predictions(response['Body'])
        
        # Return candidates sorted by purchase probability
        ranked_candidates = list(zip(candidates, purchase_probabilities))
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_candidates
    
    async def _extract_ranking_features(self, candidates: List[str], 
                                       user_id: str, context: dict) -> pd.DataFrame:
        """
        Comprehensive feature engineering for ranking model
        """
        # Parallel feature extraction
        user_features, item_features, contextual_features = await asyncio.gather(
            self._get_user_features(user_id),
            self._get_item_features(candidates),
            self._get_contextual_features(context)
        )
        
        # Combine features into model input format
        feature_matrix = []
        
        for item_id in candidates:
            features = {
                # User Features (static per request)
                'user_age_bucket': user_features.get('age_bucket', 0),
                'user_avg_price': user_features.get('avg_price_range', 0),
                'user_purchase_frequency': user_features.get('purchase_frequency', 0),
                'user_preferred_categories': user_features.get('category_preferences', []),
                
                # Item Features (varies per candidate)
                'item_price': item_features[item_id].get('price', 0),
                'item_category': item_features[item_id].get('category_id', 0),
                'item_popularity_score': item_features[item_id].get('popularity', 0),
                'item_seasonal_relevance': item_features[item_id].get('seasonal_score', 0),
                'item_inventory_velocity': item_features[item_id].get('velocity', 0),
                
                # Interaction Features (user-item specific)
                'price_affordability_ratio': self._calculate_price_ratio(
                    item_features[item_id].get('price'), user_features.get('avg_price_range')
                ),
                'category_preference_match': self._calculate_category_match(
                    item_features[item_id].get('category_id'), 
                    user_features.get('category_preferences')
                ),
                
                # Contextual Features
                'time_of_day': contextual_features.get('hour_bucket', 0),
                'day_of_week': contextual_features.get('day_of_week', 0),
                'season': contextual_features.get('season', 0),
                'device_type': contextual_features.get('device_type', 0)
            }
            
            feature_matrix.append(features)
        
        return pd.DataFrame(feature_matrix)

# SageMaker XGBoost Training Configuration
xgboost_training_config = {
    "training_job_name": "fashion-ranking-xgboost",
    "algorithm_specification": {
        "training_image": "246618743249.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest",
        "training_input_mode": "File"
    },
    "role_arn": "arn:aws:iam::ACCOUNT:role/SageMakerRole",
    "input_data_config": [{
        "channel_name": "training",
        "data_source": {
            "s3_data_source": {
                "s3_data_type": "S3Prefix",
                "s3_uri": "s3://fashion-recommender-data/features/ranking_training/",
                "s3_data_distribution_type": "FullyReplicated"
            }
        },
        "content_type": "text/csv",
        "compression_type": "None"
    }],
    "output_data_config": {
        "s3_output_path": "s3://fashion-recommender-data/models/xgboost/"
    },
    "resource_config": {
        "instance_type": "ml.m5.2xlarge",
        "instance_count": 1,
        "volume_size_in_gb": 20
    },
    "stopping_condition": {
        "max_runtime_in_seconds": 3600
    },
    "hyperparameters": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "num_round": 100,
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "lambda": 1.0,
        "alpha": 0.0
    }
}
```

### Stage 4: Ordering

**Purpose:** Apply final business logic, diversity constraints, and personalization to create the optimal user experience.

```python
class RecommendationOrderer:
    def __init__(self, business_rules_config: dict):
        self.rules_config = business_rules_config
    
    async def order_recommendations(self, ranked_candidates: List[Tuple[str, float]], 
                                   user_id: str, limit: int = 20) -> List[str]:
        """
        Stage 4: Apply business logic and ordering optimization
        """
        # Get user preferences for ordering
        user_preferences = await self._get_user_ordering_preferences(user_id)
        
        ordered_recommendations = []
        used_categories = set()
        used_brands = set()
        
        for item_id, score in ranked_candidates:
            if len(ordered_recommendations) >= limit:
                break
                
            item_metadata = await self._get_item_metadata(item_id)
            
            # Diversity constraint: max 3 items per category
            category = item_metadata.get('category')
            category_count = sum(1 for rec in ordered_recommendations 
                               if self._get_category(rec) == category)
            if category_count >= 3:
                continue
            
            # Brand diversity: max 2 items per brand
            brand = item_metadata.get('brand')
            brand_count = sum(1 for rec in ordered_recommendations 
                            if self._get_brand(rec) == brand)
            if brand_count >= 2:
                continue
                
            # Promotion boost for new arrivals (first 5 positions)
            if (len(ordered_recommendations) < 5 and 
                item_metadata.get('is_new_arrival', False)):
                # Boost new arrivals to top positions
                ordered_recommendations.insert(
                    min(len(ordered_recommendations), 2), item_id
                )
            else:
                ordered_recommendations.append(item_id)
                
            used_categories.add(category)
            used_brands.add(brand)
        
        # Apply final user-specific ordering preferences
        ordered_recommendations = self._apply_user_ordering_preferences(
            ordered_recommendations, user_preferences
        )
        
        return ordered_recommendations

    def _apply_user_ordering_preferences(self, recommendations: List[str], 
                                       preferences: dict) -> List[str]:
        """
        Apply user-specific ordering preferences (price-first, brand loyalty, etc.)
        """
        if preferences.get('sort_by_price', False):
            # Re-sort by price within each group of 5 recommendations
            grouped = [recommendations[i:i+5] for i in range(0, len(recommendations), 5)]
            sorted_groups = []
            for group in grouped:
                sorted_group = sorted(group, 
                                    key=lambda x: self._get_item_price(x),
                                    reverse=preferences.get('price_descending', False))
                sorted_groups.extend(sorted_group)
            return sorted_groups
        
        return recommendations
```

## AWS SageMaker Implementation

### Training Pipeline Architecture

**Philosophy:** Fully automated training pipeline with hyperparameter optimization and model versioning.

```python
# Complete SageMaker Training Pipeline
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

class RecommenderTrainingPipeline:
    def __init__(self, role_arn: str, bucket_name: str):
        self.sagemaker_session = sagemaker.Session()
        self.role_arn = role_arn
        self.bucket = bucket_name
        
    def create_training_pipeline(self):
        """
        Create complete training pipeline with hyperparameter tuning
        """
        # Two-Tower Model Training Job
        tensorflow_estimator = TensorFlow(
            entry_point='two_tower_train.py',
            source_dir='src/ml/training/',
            role=self.role_arn,
            instance_count=1,
            instance_type='ml.m5.2xlarge',  # Cost-optimized for learning
            framework_version='2.8.0',
            py_version='py39',
            hyperparameters={
                'embedding-dim': 256,
                'hidden-units': '512,256,128',
                'learning-rate': 0.001,
                'dropout-rate': 0.3,
                'batch-size': 1024,
                'epochs': 50
            },
            metric_definitions=[
                {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
                {'Name': 'validation:auc', 'Regex': 'Validation AUC: ([0-9\\.]+)'},
                {'Name': 'validation:loss', 'Regex': 'Validation Loss: ([0-9\\.]+)'}
            ]
        )
        
        # Hyperparameter Tuning Configuration
        hyperparameter_ranges = {
            'learning-rate': ContinuousParameter(0.0001, 0.01),
            'embedding-dim': IntegerParameter(128, 512),
            'dropout-rate': ContinuousParameter(0.1, 0.5),
            'batch-size': IntegerParameter(512, 2048)
        }
        
        objective_metric_name = 'validation:auc'
        objective_type = 'Maximize'
        
        tuner = HyperparameterTuner(
            tensorflow_estimator,
            objective_metric_name,
            hyperparameter_ranges,
            objective_type=objective_type,
            max_jobs=10,  # Learning project constraint
            max_parallel_jobs=2,
            early_stopping_type='Auto'
        )
        
        return tuner
    
    def start_training(self, tuner, train_data_path: str, val_data_path: str):
        """
        Start training with data from S3
        """
        training_input = sagemaker.inputs.TrainingInput(
            s3_data=train_data_path,
            distribution='FullyReplicated',
            content_type='text/csv'
        )
        
        validation_input = sagemaker.inputs.TrainingInput(
            s3_data=val_data_path,
            distribution='FullyReplicated',
            content_type='text/csv'
        )
        
        tuner.fit({
            'training': training_input,
            'validation': validation_input
        })
        
        return tuner

# Training Script (two_tower_train.py)
"""
#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--hidden-units', type=str, default='512,256,128')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout-rate', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=50)
    
    # SageMaker directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    return parser.parse_args()

def load_data(data_dir: str):
    # Load training data from S3
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # Feature engineering and preprocessing
    user_features = preprocess_user_features(train_df)
    item_features = preprocess_item_features(train_df)
    labels = train_df['interaction'].values
    
    return user_features, item_features, labels

def preprocess_user_features(df):
    # Implement user feature preprocessing
    # This includes categorical encoding, normalization, etc.
    pass

def preprocess_item_features(df):
    # Implement item feature preprocessing
    pass

def main():
    args = parse_args()
    
    # Load and preprocess data
    user_train, item_train, y_train = load_data(args.training)
    user_val, item_val, y_val = load_data(args.validation)
    
    # Initialize model
    hidden_units = [int(x) for x in args.hidden_units.split(',')]
    
    model = TwoTowerRecommender(
        embedding_dim=args.embedding_dim,
        user_vocab_sizes=get_user_vocab_sizes(user_train),
        item_vocab_sizes=get_item_vocab_sizes(item_train),
        hidden_units=hidden_units
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=contrastive_loss,
        metrics=['AUC']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'best_model'),
            save_best_only=True
        )
    ]
    
    # Training
    history = model.fit(
        {'user_features': user_train, 'item_features': item_train},
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=({'user_features': user_val, 'item_features': item_val}, y_val),
        callbacks=callbacks
    )
    
    # Save model
    model.save(os.path.join(args.model_dir, 'two_tower_model'))
    
    # Print final metrics
    val_loss = min(history.history['val_loss'])
    val_auc = max(history.history['val_auc'])
    print(f'Validation Loss: {val_loss}')
    print(f'Validation AUC: {val_auc}')

if __name__ == '__main__':
    main()
"""
```

### Batch Inference Pipeline

**Purpose:** Generate embeddings for all users and items, update vector database.

```python
class BatchInferencePipeline:
    def __init__(self, sagemaker_session, model_name: str):
        self.sagemaker_session = sagemaker_session
        self.model_name = model_name
    
    def create_batch_transform_job(self, input_data_path: str, 
                                  output_data_path: str):
        """
        Create batch transform job for embedding generation
        """
        transformer = sagemaker.transformer.Transformer(
            model_name=self.model_name,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            output_path=output_data_path,
            sagemaker_session=self.sagemaker_session
        )
        
        transformer.transform(
            data=input_data_path,
            content_type='text/csv',
            split_type='Line',
            join_source='Input'
        )
        
        return transformer

# Lambda Function for OpenSearch Index Update
"""
import json
import boto3
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from aws_requests_auth.aws_auth import AWSRequestsAuth

def lambda_handler(event, context):
    
    # Triggered by S3 event when new embeddings are available
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Download embeddings from S3
    s3_client = boto3.client('s3')
    embeddings_data = s3_client.get_object(Bucket=bucket, Key=key)
    embeddings_df = pd.read_csv(embeddings_data['Body'])
    
    # Connect to OpenSearch
    opensearch_client = get_opensearch_client()
    
    # Batch update embeddings index
    actions = []
    for _, row in embeddings_df.iterrows():
        action = {
            "_op_type": "index",
            "_index": "fashion_items",
            "_id": row['article_id'],
            "_source": {
                "article_id": row['article_id'],
                "item_embedding": row['embedding'].tolist(),
                "category": row['category'],
                "price": row['price'],
                "availability": row['availability'],
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        actions.append(action)
    
    # Bulk index update
    from opensearchpy.helpers import bulk
    bulk(opensearch_client, actions)
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'Updated {len(actions)} embeddings')
    }

def get_opensearch_client():
    host = os.environ['OPENSEARCH_ENDPOINT']
    region = os.environ['AWS_REGION']
    service = 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWSRequestsAuth(credentials, region, service)
    
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    
    return client
"""
```

### Model Registry and Versioning

```python
# Model Registry Management
class ModelRegistry:
    def __init__(self, sagemaker_client):
        self.sagemaker_client = sagemaker_client
        
    def register_model(self, model_package_group_name: str, 
                      training_job_name: str, model_metrics: dict):
        """
        Register trained model in SageMaker Model Registry
        """
        model_package = self.sagemaker_client.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageDescription="Two-tower recommendation model",
            InferenceSpecification={
                'Containers': [{
                    'Image': '246618743249.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.8.0-cpu',
                    'ModelDataUrl': f's3://fashion-recommender-data/models/{training_job_name}/output/model.tar.gz'
                }],
                'SupportedContentTypes': ['text/csv'],
                'SupportedResponseMIMETypes': ['application/json']
            },
            ModelMetrics={
                'ModelQuality': {
                    'Statistics': {
                        'ContentType': 'application/json',
                        'S3Uri': f's3://fashion-recommender-data/models/{training_job_name}/evaluation/metrics.json'
                    }
                }
            },
            ModelApprovalStatus='PendingManualApproval'
        )
        
        return model_package['ModelPackageArn']
    
    def approve_model(self, model_package_arn: str):
        """
        Approve model for production deployment
        """
        self.sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus='Approved'
        )
```

## Vector Database Integration

### OpenSearch Configuration

**Architecture Strategy:** Use OpenSearch as vector database with ANN indices optimized for cosine similarity search.

```yaml
# OpenSearch Cluster Configuration
opensearch_cluster:
  cluster_name: "fashion-recommender-vectors"
  elasticsearch_version: "OpenSearch_1.3"
  
  cluster_config:
    instance_type: "t3.small.search"  # Learning project sizing
    instance_count: 1
    dedicated_master_enabled: false
    zone_awareness_enabled: false
    
  ebs_options:
    ebs_enabled: true
    volume_type: "gp2"
    volume_size: 20
    
  vpc_options:
    security_group_ids: ["sg-recommendation-opensearch"]
    subnet_ids: ["subnet-private-1a"]
    
  access_policies: |
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "AWS": "arn:aws:iam::ACCOUNT:role/RecommendationServiceRole"
          },
          "Action": "es:*",
          "Resource": "arn:aws:es:us-west-2:ACCOUNT:domain/fashion-recommender-vectors/*"
        }
      ]
    }

# Index Configuration
index_settings:
  settings:
    number_of_shards: 1
    number_of_replicas: 0
    index:
      knn: true
      knn.space_type: "cosinesimil"
      knn.algo_param.ef_search: 200
      knn.algo_param.ef_construction: 200
      knn.algo_param.m: 16
      
  mappings:
    properties:
      article_id:
        type: "keyword"
      item_embedding:
        type: "knn_vector"
        dimension: 256
        method:
          name: "hnsw"
          space_type: "cosinesimil"
          engine: "nmslib"
          parameters:
            ef_construction: 200
            m: 16
      category:
        type: "keyword"
      price:
        type: "float"
      availability:
        type: "boolean"
      popularity_score:
        type: "float"
      last_updated:
        type: "date"
```

### Similarity Search Optimization

```python
class OptimizedVectorSearch:
    def __init__(self, opensearch_client):
        self.client = opensearch_client
        self.embedding_cache = {}  # In-memory cache for hot embeddings
    
    async def search_similar_items(self, user_embedding: np.ndarray, 
                                  filters: dict = None, k: int = 100):
        """
        Optimized vector similarity search with filtering
        """
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [{
                        "knn": {
                            "item_embedding": {
                                "vector": user_embedding.tolist(),
                                "k": k,
                                "ef_search": min(k * 2, 400)  # Dynamic ef_search
                            }
                        }
                    }]
                }
            },
            "_source": {
                "excludes": ["item_embedding"]  # Don't return large embedding vectors
            }
        }
        
        # Add filters if specified
        if filters:
            filter_clauses = []
            if 'categories' in filters:
                filter_clauses.append({
                    "terms": {"category": filters['categories']}
                })
            if 'price_range' in filters:
                filter_clauses.append({
                    "range": {
                        "price": {
                            "gte": filters['price_range'][0],
                            "lte": filters['price_range'][1]
                        }
                    }
                })
            if 'availability' in filters:
                filter_clauses.append({
                    "term": {"availability": filters['availability']}
                })
            
            if filter_clauses:
                search_body["query"]["bool"]["filter"] = filter_clauses
        
        # Execute search with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.search(
                    index="fashion_items",
                    body=search_body
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        # Process results
        similar_items = []
        for hit in response["hits"]["hits"]:
            similar_items.append({
                "item_id": hit["_source"]["article_id"],
                "similarity_score": hit["_score"],
                "metadata": {
                    "category": hit["_source"]["category"],
                    "price": hit["_source"]["price"],
                    "availability": hit["_source"]["availability"]
                }
            })
        
        return similar_items

# Embedding Update Pipeline
class EmbeddingUpdater:
    def __init__(self, opensearch_client, s3_client):
        self.opensearch_client = opensearch_client
        self.s3_client = s3_client
        
    async def update_embeddings_from_s3(self, s3_path: str):
        """
        Batch update embeddings from S3 with optimized bulk operations
        """
        # Download embeddings in chunks
        embedding_chunks = self._download_embeddings_in_chunks(s3_path)
        
        for chunk in embedding_chunks:
            # Prepare bulk update actions
            actions = []
            for embedding_record in chunk:
                action = {
                    "_op_type": "index",
                    "_index": "fashion_items",
                    "_id": embedding_record["article_id"],
                    "_source": {
                        "item_embedding": embedding_record["embedding"],
                        "last_updated": datetime.utcnow().isoformat()
                    }
                }
                actions.append(action)
            
            # Bulk update with retry logic
            await self._bulk_update_with_retry(actions)
    
    async def _bulk_update_with_retry(self, actions: List[dict]):
        """
        Bulk update with exponential backoff retry
        """
        from opensearchpy.helpers import bulk
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success_count, failed_actions = bulk(
                    self.opensearch_client,
                    actions,
                    chunk_size=100,
                    timeout="60s"
                )
                
                if failed_actions:
                    print(f"Failed to update {len(failed_actions)} embeddings")
                    
                return success_count
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1 * (2 ** attempt))
```

## Feature Engineering

### User Feature Engineering

**Philosophy:** Create rich, time-aware user representations that capture both static demographics and dynamic behavioral patterns.

```python
class UserFeatureEngineering:
    def __init__(self, data_source):
        self.data_source = data_source
    
    def extract_user_features(self, transactions_df: pd.DataFrame, 
                             customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive user feature engineering pipeline
        """
        # Static demographic features
        demographic_features = self._extract_demographic_features(customers_df)
        
        # Behavioral features from transaction history
        behavioral_features = self._extract_behavioral_features(transactions_df)
        
        # Temporal pattern features
        temporal_features = self._extract_temporal_features(transactions_df)
        
        # Preference features
        preference_features = self._extract_preference_features(transactions_df)
        
        # Combine all features
        user_features = demographic_features.merge(
            behavioral_features, on='customer_id', how='left'
        ).merge(
            temporal_features, on='customer_id', how='left'
        ).merge(
            preference_features, on='customer_id', how='left'
        )
        
        return user_features
    
    def _extract_demographic_features(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process demographic features with proper encoding
        """
        features = customers_df.copy()
        
        # Age bucketing for better generalization
        features['age_bucket'] = pd.cut(
            features['age'], 
            bins=[0, 25, 35, 45, 55, 100], 
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        
        # Postal code to region mapping (privacy-safe)
        features['region'] = features['postal_code'].astype(str).str[:2]
        
        # Fashion Newsletter subscription indicator
        features['fashion_news_subscriber'] = (features['FN'] == 1).astype(int)
        
        # Active customer indicator
        features['is_active'] = (features['Active'] == 1).astype(int)
        
        # Club member features
        features['club_member_status'] = features['club_member_status'].fillna('None')
        
        return features[['customer_id', 'age_bucket', 'region', 'fashion_news_subscriber', 
                        'is_active', 'club_member_status']]
    
    def _extract_behavioral_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract behavioral patterns from transaction history
        """
        behavioral_agg = transactions_df.groupby('customer_id').agg({
            'price': ['mean', 'std', 'min', 'max', 'sum'],
            'article_id': ['count', 'nunique'],
            't_dat': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        behavioral_agg.columns = [
            'avg_price', 'price_std', 'min_price', 'max_price', 'total_spent',
            'total_purchases', 'unique_items', 'first_purchase', 'last_purchase'
        ]
        
        # Derived features
        behavioral_agg['purchase_frequency'] = (
            behavioral_agg['total_purchases'] / 
            ((pd.to_datetime(behavioral_agg['last_purchase']) - 
              pd.to_datetime(behavioral_agg['first_purchase'])).dt.days + 1)
        ).fillna(0)
        
        behavioral_agg['price_consistency'] = (
            1 - (behavioral_agg['price_std'] / behavioral_agg['avg_price'])
        ).fillna(1).clip(0, 1)
        
        behavioral_agg['customer_lifetime_days'] = (
            pd.to_datetime(behavioral_agg['last_purchase']) - 
            pd.to_datetime(behavioral_agg['first_purchase'])
        ).dt.days + 1
        
        return behavioral_agg.reset_index()
    
    def _extract_temporal_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal shopping patterns
        """
        transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
        transactions_df['day_of_week'] = transactions_df['t_dat'].dt.dayofweek
        transactions_df['month'] = transactions_df['t_dat'].dt.month
        transactions_df['season'] = transactions_df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Day of week preferences
        dow_preferences = transactions_df.groupby(['customer_id', 'day_of_week']).size().unstack(fill_value=0)
        dow_preferences = dow_preferences.div(dow_preferences.sum(axis=1), axis=0)
        dow_preferences.columns = [f'dow_pref_{i}' for i in range(7)]
        
        # Seasonal preferences
        seasonal_preferences = transactions_df.groupby(['customer_id', 'season']).size().unstack(fill_value=0)
        seasonal_preferences = seasonal_preferences.div(seasonal_preferences.sum(axis=1), axis=0)
        seasonal_preferences.columns = [f'season_pref_{col.lower()}' for col in seasonal_preferences.columns]
        
        # Recency features
        current_date = transactions_df['t_dat'].max()
        recency_features = transactions_df.groupby('customer_id')['t_dat'].max().reset_index()
        recency_features['days_since_last_purchase'] = (
            current_date - recency_features['t_dat']
        ).dt.days
        
        # Combine temporal features
        temporal_features = dow_preferences.merge(
            seasonal_preferences, left_index=True, right_index=True, how='outer'
        ).merge(
            recency_features[['customer_id', 'days_since_last_purchase']], 
            left_index=True, right_on='customer_id', how='outer'
        ).fillna(0)
        
        return temporal_features
    
    def _extract_preference_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract category and style preferences
        """
        # Merge with article metadata to get categories
        articles_df = self._load_article_metadata()
        transactions_with_articles = transactions_df.merge(
            articles_df[['article_id', 'product_group_name', 'garment_group_name', 
                        'index_name', 'colour_group_name']], 
            on='article_id', how='left'
        )
        
        # Category preferences (normalized)
        category_preferences = (
            transactions_with_articles.groupby(['customer_id', 'product_group_name'])
            .size().unstack(fill_value=0)
        )
        category_preferences = category_preferences.div(category_preferences.sum(axis=1), axis=0)
        
        # Color preferences
        color_preferences = (
            transactions_with_articles.groupby(['customer_id', 'colour_group_name'])
            .size().unstack(fill_value=0)
        )
        color_preferences = color_preferences.div(color_preferences.sum(axis=1), axis=0)
        
        # Style diversity (entropy-based)
        style_diversity = []
        for customer_id in transactions_with_articles['customer_id'].unique():
            customer_styles = transactions_with_articles[
                transactions_with_articles['customer_id'] == customer_id
            ]['garment_group_name'].value_counts(normalize=True)
            
            entropy = -sum(p * np.log2(p) for p in customer_styles if p > 0)
            style_diversity.append({'customer_id': customer_id, 'style_diversity': entropy})
        
        style_diversity_df = pd.DataFrame(style_diversity)
        
        # Combine preference features
        preference_features = category_preferences.merge(
            color_preferences, left_index=True, right_index=True, how='outer', suffixes=('_cat', '_color')
        ).merge(
            style_diversity_df, left_index=True, right_on='customer_id', how='outer'
        ).fillna(0)
        
        return preference_features
```

### Item Feature Engineering

```python
class ItemFeatureEngineering:
    def __init__(self, articles_df: pd.DataFrame):
        self.articles_df = articles_df
    
    def extract_item_features(self) -> pd.DataFrame:
        """
        Comprehensive item feature engineering pipeline
        """
        features = self.articles_df.copy()
        
        # Categorical feature encoding
        features = self._encode_categorical_features(features)
        
        # Price features
        features = self._extract_price_features(features)
        
        # Hierarchical category features
        features = self._extract_hierarchical_features(features)
        
        # Textual features from descriptions
        features = self._extract_textual_features(features)
        
        # Popularity and inventory features (requires transaction data)
        # This would be computed from transaction history
        
        return features
    
    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features with frequency and target encoding
        """
        # Label encoding for high-cardinality categoricals
        categorical_cols = [
            'product_group_name', 'garment_group_name', 'index_name',
            'index_group_name', 'section_name', 'colour_group_name'
        ]
        
        for col in categorical_cols:
            # Frequency encoding (popularity-based)
            freq_encoding = features[col].value_counts().to_dict()
            features[f'{col}_frequency'] = features[col].map(freq_encoding)
            
            # Label encoding for embedding layers
            features[f'{col}_encoded'] = pd.factorize(features[col])[0]
        
        return features
    
    def _extract_price_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Price-based feature engineering
        """
        # Price percentile within category
        for category in ['product_group_name', 'garment_group_name']:
            features[f'price_percentile_{category}'] = (
                features.groupby(category)['price']
                .transform(lambda x: x.rank(pct=True))
            )
        
        # Price tiers
        features['price_tier'] = pd.qcut(
            features['price'], 
            q=5, 
            labels=['Budget', 'Low', 'Mid', 'Premium', 'Luxury']
        )
        
        # Price relative to category average
        category_avg_price = features.groupby('product_group_name')['price'].transform('mean')
        features['price_vs_category_avg'] = features['price'] / category_avg_price
        
        return features
    
    def _extract_hierarchical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract hierarchical category relationships
        """
        # Create category hierarchy paths
        features['category_path'] = (
            features['index_group_name'].astype(str) + '/' +
            features['index_name'].astype(str) + '/' +
            features['section_name'].astype(str) + '/' +
            features['garment_group_name'].astype(str)
        )
        
        # Depth in hierarchy
        features['hierarchy_depth'] = features['category_path'].str.count('/') + 1
        
        # Category breadth (number of items in same category)
        for level in ['index_group_name', 'index_name', 'section_name', 'garment_group_name']:
            category_counts = features[level].value_counts()
            features[f'{level}_breadth'] = features[level].map(category_counts)
        
        return features
```

## Model Training Pipeline

### Complete Training Workflow

```python
# Step Functions Workflow for ML Pipeline
import boto3
import json
from datetime import datetime

class MLPipelineOrchestrator:
    def __init__(self):
        self.stepfunctions_client = boto3.client('stepfunctions')
        self.sagemaker_client = boto3.client('sagemaker')
        
    def create_training_workflow(self):
        """
        Define Step Functions state machine for ML pipeline
        """
        state_machine_definition = {
            "Comment": "Fashion Recommendation ML Training Pipeline",
            "StartAt": "DataValidation",
            "States": {
                "DataValidation": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::glue:startJobRun.sync",
                    "Parameters": {
                        "JobName": "validate-training-data",
                        "Arguments": {
                            "--input-path": "s3://fashion-recommender-data/processed/",
                            "--validation-rules": "s3://fashion-recommender-data/validation/rules.json"
                        }
                    },
                    "Next": "FeatureEngineering",
                    "Catch": [{
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "DataValidationFailed"
                    }]
                },
                
                "FeatureEngineering": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::glue:startJobRun.sync",
                    "Parameters": {
                        "JobName": "feature-engineering-pipeline",
                        "Arguments": {
                            "--input-path": "s3://fashion-recommender-data/processed/",
                            "--output-path": "s3://fashion-recommender-data/features/"
                        }
                    },
                    "Next": "ModelTraining"
                },
                
                "ModelTraining": {
                    "Type": "Parallel",
                    "Branches": [
                        {
                            "StartAt": "TwoTowerTraining",
                            "States": {
                                "TwoTowerTraining": {
                                    "Type": "Task",
                                    "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                                    "Parameters": {
                                        "TrainingJobName.$": "$.two_tower_job_name",
                                        "AlgorithmSpecification": {
                                            "TrainingImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.8.0-cpu-py39",
                                            "TrainingInputMode": "File"
                                        },
                                        "RoleArn": "arn:aws:iam::ACCOUNT:role/SageMakerRole",
                                        "InputDataConfig": [{
                                            "ChannelName": "training",
                                            "DataSource": {
                                                "S3DataSource": {
                                                    "S3DataType": "S3Prefix",
                                                    "S3Uri": "s3://fashion-recommender-data/features/training/",
                                                    "S3DataDistributionType": "FullyReplicated"
                                                }
                                            }
                                        }],
                                        "OutputDataConfig": {
                                            "S3OutputPath": "s3://fashion-recommender-data/models/two-tower/"
                                        },
                                        "ResourceConfig": {
                                            "InstanceType": "ml.m5.2xlarge",
                                            "InstanceCount": 1,
                                            "VolumeSizeInGB": 30
                                        },
                                        "StoppingCondition": {
                                            "MaxRuntimeInSeconds": 7200
                                        }
                                    },
                                    "End": True
                                }
                            }
                        },
                        {
                            "StartAt": "RankingModelTraining",
                            "States": {
                                "RankingModelTraining": {
                                    "Type": "Task",
                                    "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                                    "Parameters": {
                                        "TrainingJobName.$": "$.ranking_job_name",
                                        "AlgorithmSpecification": {
                                            "TrainingImage": "246618743249.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest",
                                            "TrainingInputMode": "File"
                                        },
                                        "RoleArn": "arn:aws:iam::ACCOUNT:role/SageMakerRole",
                                        "InputDataConfig": [{
                                            "ChannelName": "training",
                                            "DataSource": {
                                                "S3DataSource": {
                                                    "S3DataType": "S3Prefix",
                                                    "S3Uri": "s3://fashion-recommender-data/features/ranking/",
                                                    "S3DataDistributionType": "FullyReplicated"
                                                }
                                            }
                                        }],
                                        "OutputDataConfig": {
                                            "S3OutputPath": "s3://fashion-recommender-data/models/ranking/"
                                        },
                                        "ResourceConfig": {
                                            "InstanceType": "ml.m5.xlarge",
                                            "InstanceCount": 1,
                                            "VolumeSizeInGB": 20
                                        }
                                    },
                                    "End": True
                                }
                            }
                        }
                    ],
                    "Next": "ModelEvaluation"
                },
                
                "ModelEvaluation": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::lambda:invoke",
                    "Parameters": {
                        "FunctionName": "evaluate-models",
                        "Payload": {
                            "two_tower_model_path.$": "$[0].ModelArtifacts.S3ModelArtifacts",
                            "ranking_model_path.$": "$[1].ModelArtifacts.S3ModelArtifacts",
                            "validation_data_path": "s3://fashion-recommender-data/features/validation/"
                        }
                    },
                    "Next": "ModelApprovalDecision"
                },
                
                "ModelApprovalDecision": {
                    "Type": "Choice",
                    "Choices": [{
                        "Variable": "$.evaluation_results.approved",
                        "BooleanEquals": True,
                        "Next": "DeployModels"
                    }],
                    "Default": "ModelRejected"
                },
                
                "DeployModels": {
                    "Type": "Parallel",
                    "Branches": [
                        {
                            "StartAt": "DeployEmbeddingModel",
                            "States": {
                                "DeployEmbeddingModel": {
                                    "Type": "Task",
                                    "Resource": "arn:aws:states:::sagemaker:createEndpoint",
                                    "Parameters": {
                                        "EndpointName": "two-tower-embedding-endpoint",
                                        "EndpointConfigName": "two-tower-config-latest"
                                    },
                                    "End": True
                                }
                            }
                        },
                        {
                            "StartAt": "DeployRankingModel",
                            "States": {
                                "DeployRankingModel": {
                                    "Type": "Task",
                                    "Resource": "arn:aws:states:::sagemaker:createEndpoint",
                                    "Parameters": {
                                        "EndpointName": "xgboost-ranking-endpoint",
                                        "EndpointConfigName": "ranking-config-latest"
                                    },
                                    "End": True
                                }
                            }
                        }
                    ],
                    "Next": "UpdateVectorDatabase"
                },
                
                "UpdateVectorDatabase": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::batch:submitJob.sync",
                    "Parameters": {
                        "JobDefinition": "update-embeddings-job",
                        "JobName": "update-vector-database",
                        "JobQueue": "ml-inference-queue",
                        "Parameters": {
                            "modelEndpoint": "two-tower-embedding-endpoint",
                            "inputData": "s3://fashion-recommender-data/processed/articles/",
                            "outputPath": "s3://fashion-recommender-data/embeddings/"
                        }
                    },
                    "End": True
                },
                
                "DataValidationFailed": {
                    "Type": "Fail",
                    "Error": "DataValidationError",
                    "Cause": "Training data failed validation checks"
                },
                
                "ModelRejected": {
                    "Type": "Fail",
                    "Error": "ModelQualityError",  
                    "Cause": "Model failed quality evaluation"
                }
            }
        }
        
        return json.dumps(state_machine_definition)
```

### Model Evaluation Framework

```python
class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_two_tower_model(self, model_path: str, test_data_path: str):
        """
        Comprehensive evaluation of two-tower model
        """
        # Load model and test data
        model = tf.keras.models.load_model(model_path)
        test_data = self._load_test_data(test_data_path)
        
        # Embedding Quality Metrics
        user_embeddings = model.get_layer('user_embedding').predict(test_data['user_features'])
        item_embeddings = model.get_layer('item_embedding').predict(test_data['item_features'])
        
        # 1. Embedding Similarity Distribution
        similarity_scores = np.dot(user_embeddings, item_embeddings.T)
        
        self.metrics['embedding_stats'] = {
            'mean_similarity': float(np.mean(similarity_scores)),
            'std_similarity': float(np.std(similarity_scores)),
            'similarity_range': [float(np.min(similarity_scores)), float(np.max(similarity_scores))]
        }
        
        # 2. Recommendation Quality Metrics
        recommendations = self._generate_recommendations_for_evaluation(
            user_embeddings, item_embeddings, test_data['user_ids'], test_data['item_ids']
        )
        
        # Precision@K, Recall@K, NDCG@K
        precision_at_k = self._calculate_precision_at_k(recommendations, test_data['ground_truth'], k=10)
        recall_at_k = self._calculate_recall_at_k(recommendations, test_data['ground_truth'], k=10)
        ndcg_at_k = self._calculate_ndcg_at_k(recommendations, test_data['ground_truth'], k=10)
        
        self.metrics['recommendation_quality'] = {
            'precision_at_10': float(precision_at_k),
            'recall_at_10': float(recall_at_k),
            'ndcg_at_10': float(ndcg_at_k)
        }
        
        # 3. Diversity and Coverage Metrics
        catalog_coverage = self._calculate_catalog_coverage(recommendations, test_data['total_items'])
        diversity_score = self._calculate_intra_list_diversity(recommendations, item_embeddings)
        
        self.metrics['diversity_coverage'] = {
            'catalog_coverage': float(catalog_coverage),
            'avg_intra_list_diversity': float(diversity_score)
        }
        
        # 4. Cold Start Performance
        cold_start_performance = self._evaluate_cold_start_performance(
            model, test_data['cold_start_users']
        )
        
        self.metrics['cold_start'] = cold_start_performance
        
        return self.metrics
    
    def evaluate_ranking_model(self, model_endpoint: str, test_data_path: str):
        """
        Evaluate XGBoost ranking model performance
        """
        sagemaker_runtime = boto3.client('sagemaker-runtime')
        test_data = pd.read_csv(test_data_path)
        
        # Batch prediction
        predictions = []
        batch_size = 1000
        
        for i in range(0, len(test_data), batch_size):
            batch = test_data.iloc[i:i+batch_size]
            
            # Format for SageMaker endpoint
            payload = batch.drop(['user_id', 'item_id', 'label'], axis=1).to_csv(header=False, index=False)
            
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=model_endpoint,
                ContentType='text/csv',
                Body=payload
            )
            
            batch_predictions = json.loads(response['Body'].read().decode())
            predictions.extend(batch_predictions)
        
        # Calculate ranking metrics
        y_true = test_data['label'].values
        y_pred = np.array(predictions)
        
        # AUC-ROC
        from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
        
        auc_score = roc_auc_score(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        
        # Ranking-specific metrics
        ranking_metrics = self._calculate_ranking_metrics(test_data, predictions)
        
        return {
            'auc_roc': float(auc_score),
            'average_precision': float(avg_precision),
            'log_loss': float(logloss),
            **ranking_metrics
        }
    
    def _calculate_ranking_metrics(self, test_data: pd.DataFrame, predictions: List[float]):
        """
        Calculate ranking-specific metrics like MRR, MAP
        """
        # Group by user to calculate user-level ranking metrics
        test_data['prediction'] = predictions
        user_metrics = []
        
        for user_id in test_data['user_id'].unique():
            user_data = test_data[test_data['user_id'] == user_id].copy()
            user_data = user_data.sort_values('prediction', ascending=False)
            
            # Mean Reciprocal Rank
            positive_ranks = []
            for idx, (_, row) in enumerate(user_data.iterrows()):
                if row['label'] == 1:
                    positive_ranks.append(1.0 / (idx + 1))
            
            mrr = np.mean(positive_ranks) if positive_ranks else 0.0
            user_metrics.append(mrr)
        
        return {
            'mean_reciprocal_rank': float(np.mean(user_metrics)),
            'user_coverage': len(user_metrics) / test_data['user_id'].nunique()
        }
```

## Real-time Serving Architecture

### Lambda-based API Orchestration

**Philosophy:** Orchestrate the 4-stage pipeline through serverless functions with intelligent caching and error handling.

```python
# Main Recommendation API Lambda Function
import json
import boto3
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class RecommendationAPI:
    def __init__(self):
        self.opensearch_client = self._init_opensearch()
        self.dynamodb_client = boto3.client('dynamodb')
        self.elasticache_client = self._init_elasticache()
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        
        # Service clients
        self.candidate_generator = CandidateGenerator(
            self.opensearch_client, self.embedding_service
        )
        self.candidate_filter = CandidateFilter(
            self.dynamodb_client, self.elasticache_client
        )
        self.ranking_service = XGBoostRanker('xgboost-ranking-endpoint')
        self.ordering_service = RecommendationOrderer({})
        
        # Performance monitoring
        self.metrics = {}
    
    async def lambda_handler(self, event, context):
        """
        Main Lambda handler for recommendation API
        """
        start_time = time.time()
        
        try:
            # Parse request
            user_id = event['pathParameters']['user_id']
            limit = int(event.get('queryStringParameters', {}).get('limit', 20))
            context_data = json.loads(event.get('body', '{}'))
            
            # Check cache first (L1 Cache - API Gateway level)
            cache_key = f"recommendations:{user_id}:{limit}"
            cached_result = await self._check_cache(cache_key)
            
            if cached_result:
                self._record_metrics('cache_hit', time.time() - start_time)
                return self._format_response(200, cached_result)
            
            # Execute 4-stage pipeline
            recommendations = await self._execute_recommendation_pipeline(
                user_id, limit, context_data
            )
            
            # Cache results (TTL: 15 minutes)
            await self._cache_results(cache_key, recommendations, ttl=900)
            
            # Record performance metrics
            total_time = time.time() - start_time
            self._record_metrics('recommendation_generated', total_time)
            
            return self._format_response(200, {
                'user_id': user_id,
                'recommendations': recommendations,
                'generated_at': int(time.time()),
                'processing_time_ms': int(total_time * 1000)
            })
            
        except Exception as e:
            self._record_metrics('error', time.time() - start_time, str(e))
            return self._format_response(500, {
                'error': 'Internal server error',
                'request_id': context.aws_request_id
            })
    
    async def _execute_recommendation_pipeline(self, user_id: str, 
                                            limit: int, context: dict) -> List[dict]:
        """
        Execute full 4-stage recommendation pipeline
        """
        # Stage 1: Candidate Generation (Target: <50ms)
        stage1_start = time.time()
        candidates = await self.candidate_generator.generate_candidates(
            user_id, top_k=100
        )
        stage1_time = time.time() - stage1_start
        
        if not candidates:
            # Fallback to popular items
            candidates = await self._get_popular_items_fallback(limit)
        
        # Stage 2: Filtering (Target: <30ms)
        stage2_start = time.time()
        filtered_candidates = await self.candidate_filter.filter_candidates(
            candidates, user_id
        )
        stage2_time = time.time() - stage2_start
        
        if not filtered_candidates:
            # Return popular items if no candidates pass filtering
            return await self._format_popular_items_response(limit)
        
        # Stage 3: Ranking (Target: <100ms)
        stage3_start = time.time()
        ranked_candidates = await self.ranking_service.rank_candidates(
            filtered_candidates, user_id, context
        )
        stage3_time = time.time() - stage3_start
        
        # Stage 4: Ordering (Target: <20ms)
        stage4_start = time.time()
        final_recommendations = await self.ordering_service.order_recommendations(
            ranked_candidates, user_id, limit
        )
        stage4_time = time.time() - stage4_start
        
        # Record stage-level metrics
        self._record_stage_metrics({
            'stage1_candidate_generation': stage1_time,
            'stage2_filtering': stage2_time,
            'stage3_ranking': stage3_time,
            'stage4_ordering': stage4_time,
            'total_candidates_generated': len(candidates),
            'candidates_after_filtering': len(filtered_candidates),
            'final_recommendations': len(final_recommendations)
        })
        
        # Format final response
        recommendations = await self._format_recommendations(final_recommendations)
        
        return recommendations
    
    async def _format_recommendations(self, item_ids: List[str]) -> List[dict]:
        """
        Format recommendations with metadata for client consumption
        """
        # Batch fetch item metadata
        item_metadata = await self._batch_fetch_item_metadata(item_ids)
        
        formatted_recommendations = []
        for idx, item_id in enumerate(item_ids):
            metadata = item_metadata.get(item_id, {})
            
            recommendation = {
                'item_id': item_id,
                'rank': idx + 1,
                'title': metadata.get('prod_name', 'Unknown Product'),
                'category': metadata.get('product_group_name', 'Unknown Category'),
                'price': metadata.get('price', 0.0),
                'image_url': f"https://assets.fashion-recommender.com/images/{item_id}.jpg",
                'description': metadata.get('detail_desc', ''),
                'availability': metadata.get('availability', True),
                'recommendation_reason': self._generate_recommendation_reason(
                    item_id, metadata, idx
                )
            }
            
            formatted_recommendations.append(recommendation)
        
        return formatted_recommendations
    
    def _generate_recommendation_reason(self, item_id: str, 
                                      metadata: dict, rank: int) -> str:
        """
        Generate human-readable recommendation reasons
        """
        reasons = []
        
        if rank < 3:
            reasons.append("Top pick for you")
        
        category = metadata.get('product_group_name', '')
        if category:
            reasons.append(f"Popular in {category}")
        
        price = metadata.get('price', 0)
        if price > 0:
            if price < 20:
                reasons.append("Great value")
            elif price > 100:
                reasons.append("Premium quality")
        
        if metadata.get('is_new_arrival', False):
            reasons.append("New arrival")
        
        return " • ".join(reasons[:2]) if reasons else "Recommended for you"
    
    def _record_metrics(self, metric_name: str, duration: float, error: str = None):
        """
        Record performance metrics for monitoring
        """
        # Send to CloudWatch
        cloudwatch = boto3.client('cloudwatch')
        
        metrics = [{
            'MetricName': f'RecommendationAPI_{metric_name}_Duration',
            'Value': duration * 1000,  # Convert to milliseconds
            'Unit': 'Milliseconds'
        }]
        
        if error:
            metrics.append({
                'MetricName': f'RecommendationAPI_{metric_name}_Error',
                'Value': 1,
                'Unit': 'Count'
            })
        
        cloudwatch.put_metric_data(
            Namespace='Fashion/RecommendationSystem',
            MetricData=metrics
        )

def lambda_handler(event, context):
    """
    AWS Lambda entry point
    """
    api = RecommendationAPI()
    
    # Run async function in Lambda
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(api.lambda_handler(event, context))
    finally:
        loop.close()
```

### Caching Strategy Implementation

```python
class IntelligentCaching:
    def __init__(self):
        self.elasticache_client = self._init_elasticache()
        self.cache_stats = {}
    
    async def get_cached_recommendations(self, user_id: str, 
                                       context_hash: str) -> Optional[dict]:
        """
        Multi-level caching strategy for recommendations
        """
        # L1: User-specific cache (15 min TTL)
        l1_key = f"rec:user:{user_id}:{context_hash}"
        l1_result = await self.elasticache_client.get(l1_key)
        
        if l1_result:
            self._record_cache_hit('L1')
            return json.loads(l1_result)
        
        # L2: Segment-based cache (30 min TTL)
        user_segment = await self._get_user_segment(user_id)
        l2_key = f"rec:segment:{user_segment}:{context_hash}"
        l2_result = await self.elasticache_client.get(l2_key)
        
        if l2_result:
            self._record_cache_hit('L2')
            # Promote to L1 cache
            await self.elasticache_client.setex(l1_key, 900, l2_result)
            return json.loads(l2_result)
        
        self._record_cache_miss()
        return None
    
    async def cache_recommendations(self, user_id: str, context_hash: str, 
                                  recommendations: dict):
        """
        Store recommendations in appropriate cache levels
        """
        recommendations_json = json.dumps(recommendations)
        
        # L1: User-specific cache
        l1_key = f"rec:user:{user_id}:{context_hash}"
        await self.elasticache_client.setex(l1_key, 900, recommendations_json)  # 15 min
        
        # L2: Segment-based cache (if user is in major segment)
        user_segment = await self._get_user_segment(user_id)
        if await self._is_major_segment(user_segment):
            l2_key = f"rec:segment:{user_segment}:{context_hash}"
            await self.elasticache_client.setex(l2_key, 1800, recommendations_json)  # 30 min
    
    async def _get_user_segment(self, user_id: str) -> str:
        """
        Determine user segment for caching strategy
        """
        # Simple segmentation based on user behavior
        user_profile = await self._get_user_profile(user_id)
        
        age_bucket = user_profile.get('age_bucket', 'unknown')
        purchase_frequency = user_profile.get('purchase_frequency', 0)
        
        if purchase_frequency > 5:  # High-frequency shopper
            return f"high_freq_{age_bucket}"
        elif purchase_frequency > 1:  # Regular shopper
            return f"regular_{age_bucket}"
        else:  # Occasional shopper
            return f"occasional_{age_bucket}"
```

## Alternative Approaches Analysis

### When to Use Different Recommendation Algorithms

**Decision Matrix:** Choose recommendation approach based on business constraints and data characteristics.

| Approach | Data Requirements | Latency | Personalization | Cold Start | When to Use |
|----------|-------------------|---------|-----------------|------------|-------------|
| **Two-Tower + 4-Stage (Our Choice)** | Rich user/item features + interactions | <200ms | High | Good | Fashion e-commerce, rich catalog, personalization critical |
| **Matrix Factorization** | Interaction data only | <50ms | Medium | Poor | Simple catalogs, limited features, batch OK |
| **Content-Based Filtering** | Rich item features | <100ms | Low | Excellent | New catalogs, few interactions, content-heavy |
| **Deep Learning End-to-End** | Large datasets + compute | 200-500ms | Very High | Good | High-value transactions, GPU budget available |
| **Contextual Bandits** | Real-time feedback | <100ms | Adaptive | Excellent | Heavy exploration needed, rapid adaptation |
| **Session-Based (RNNs)** | Sequential interaction data | 100-300ms | High | Poor | Strong session patterns, sequential importance |

### Detailed Alternative Implementation Patterns

```python
# Alternative 1: Pure Collaborative Filtering (Matrix Factorization)
class MatrixFactorizationRecommender:
    """
    Use when: Simple catalogs, limited item features, batch processing acceptable
    Pros: Fast inference, simple implementation, no feature engineering
    Cons: Cold start problems, no content understanding
    """
    def __init__(self, n_factors=100, learning_rate=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
    
    def train(self, interaction_matrix):
        """Train matrix factorization model using SGD"""
        n_users, n_items = interaction_matrix.shape
        
        # Initialize user and item factor matrices
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # SGD training loop
        for epoch in range(self.n_epochs):
            for user_idx in range(n_users):
                for item_idx in range(n_items):
                    if interaction_matrix[user_idx, item_idx] > 0:
                        # Calculate prediction error
                        prediction = np.dot(self.user_factors[user_idx], 
                                          self.item_factors[item_idx])
                        error = interaction_matrix[user_idx, item_idx] - prediction
                        
                        # Update factors
                        user_factor = self.user_factors[user_idx]
                        self.user_factors[user_idx] += (
                            self.learning_rate * error * self.item_factors[item_idx]
                        )
                        self.item_factors[item_idx] += (
                            self.learning_rate * error * user_factor
                        )
    
    def recommend(self, user_idx, n_recommendations=20):
        """Generate recommendations for user"""
        user_scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        top_items = np.argsort(user_scores)[::-1][:n_recommendations]
        return top_items

# Alternative 2: Content-Based Filtering
class ContentBasedRecommender:
    """
    Use when: New catalogs, rich item features, few user interactions
    Pros: No cold start, interpretable, works with new items
    Cons: Limited personalization, feature engineering intensive
    """
    def __init__(self):
        self.item_profiles = {}
        self.user_profiles = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    
    def build_item_profiles(self, items_df):
        """Build item profiles from features"""
        # Combine textual features
        items_df['combined_features'] = (
            items_df['product_group_name'] + ' ' +
            items_df['garment_group_name'] + ' ' +
            items_df['colour_group_name'] + ' ' +
            items_df['detail_desc'].fillna('')
        )
        
        # TF-IDF vectorization
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(items_df['combined_features'])
        
        # Store item profiles
        for idx, item_id in enumerate(items_df['article_id']):
            self.item_profiles[item_id] = tfidf_matrix[idx].toarray()[0]
    
    def build_user_profile(self, user_id, user_interactions):
        """Build user profile from interaction history"""
        interacted_items = user_interactions['article_id'].tolist()
        
        # Weighted average of interacted item profiles
        user_vector = np.zeros(len(self.item_profiles[interacted_items[0]]))
        total_weight = 0
        
        for item_id in interacted_items:
            if item_id in self.item_profiles:
                # Weight by recency (more recent = higher weight)
                weight = 1.0  # Simplification - could be based on recency/rating
                user_vector += weight * self.item_profiles[item_id]
                total_weight += weight
        
        self.user_profiles[user_id] = user_vector / total_weight if total_weight > 0 else user_vector
    
    def recommend(self, user_id, n_recommendations=20):
        """Generate content-based recommendations"""
        user_profile = self.user_profiles.get(user_id)
        if user_profile is None:
            return []  # No profile available
        
        # Calculate similarity with all items
        similarities = {}
        for item_id, item_profile in self.item_profiles.items():
            similarity = cosine_similarity([user_profile], [item_profile])[0][0]
            similarities[item_id] = similarity
        
        # Return top similar items
        top_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in top_items[:n_recommendations]]

# Alternative 3: Session-Based Recommendations (Simplified RNN)
class SessionBasedRecommender:
    """
    Use when: Strong sequential patterns, session-based interaction important
    Pros: Captures sequential patterns, good for session-based recommendations
    Cons: Requires sequential data, complex to implement and tune
    """
    def __init__(self, embedding_dim=50, hidden_dim=100):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """Build RNN model for session-based recommendations"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=50000, output_dim=self.embedding_dim),
            tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(self.hidden_dim),
            tf.keras.layers.Dense(50000, activation='softmax')  # Vocabulary size
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_session_data(self, sessions_df):
        """Prepare sequential session data for training"""
        sequences = []
        targets = []
        
        for session_id in sessions_df['session_id'].unique():
            session_items = sessions_df[sessions_df['session_id'] == session_id]['article_id'].tolist()
            
            if len(session_items) > 2:  # Minimum sequence length
                for i in range(1, len(session_items)):
                    sequences.append(session_items[:i])
                    targets.append(session_items[i])
        
        return sequences, targets
    
    def recommend_for_session(self, current_session, n_recommendations=20):
        """Generate recommendations based on current session"""
        if not current_session:
            return []
        
        # Pad sequence to fixed length
        padded_session = tf.keras.preprocessing.sequence.pad_sequences(
            [current_session], maxlen=10, padding='pre'
        )
        
        # Get predictions
        predictions = self.model.predict(padded_session)[0]
        
        # Return top predicted items
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        return top_indices.tolist()
```

## Learning vs Production Considerations

### Model Complexity Trade-offs

**Learning Project Optimizations:**

```python
# Simplified Two-Tower for Learning (vs Production)
class SimplifiedTwoTower:
    """
    Learning project simplifications:
    - Smaller embedding dimensions (128 vs 256-512)
    - Fewer hidden layers (2 vs 4-6)
    - CPU-based training (vs GPU clusters)
    - Simpler loss function (vs complex multi-task learning)
    """
    def __init__(self):
        self.embedding_dim = 128  # Production: 256-512
        self.hidden_units = [256, 128]  # Production: [512, 256, 128, 64]
        self.dropout_rate = 0.3  # Production: Layer-specific dropout
        
    def build_simplified_architecture(self):
        """Simplified for learning while maintaining core concepts"""
        return {
            'user_tower': [
                'embedding_layer',  # Combine all categoricals
                'dense_256_relu',
                'dropout_0.3',
                'dense_128_relu',
                'l2_normalize'
            ],
            'item_tower': [
                'embedding_layer',
                'dense_256_relu', 
                'dropout_0.3',
                'dense_128_relu',
                'l2_normalize'
            ],
            'training': {
                'loss': 'binary_crossentropy',  # Production: contrastive loss
                'optimizer': 'adam',  # Production: custom learning schedules
                'metrics': ['auc']  # Production: custom metrics
            }
        }

# Production vs Learning Infrastructure Scaling
class InfrastructureScaling:
    """
    Document scaling considerations between learning and production
    """
    
    @staticmethod
    def get_scaling_comparison():
        return {
            'data_volume': {
                'learning': '100K interactions, 1K users, 500 items',
                'production': '31M interactions, 1.3M users, 105K items',
                'scaling_factor': '300x data volume'
            },
            'compute_requirements': {
                'learning': {
                    'training': 'ml.m5.xlarge (4 vCPU, 16GB RAM)',
                    'inference': 't3.small.search (1 vCPU, 2GB RAM)',
                    'cost': '$50-100/month'
                },
                'production': {
                    'training': 'ml.p3.8xlarge cluster (32 vCPU, 244GB RAM, 4 GPU)',
                    'inference': 'c5.2xlarge cluster (8 vCPU, 16GB RAM)',
                    'cost': '$2000-5000/month'
                }
            },
            'latency_requirements': {
                'learning': '<500ms acceptable',
                'production': '<100ms required',
                'optimization_needed': '5x latency improvement'
            },
            'availability_requirements': {
                'learning': '95% uptime acceptable',
                'production': '99.9% uptime required',
                'infrastructure_complexity': '10x monitoring complexity'
            }
        }

# What Changes When Scaling to Production
class ProductionUpgradePattern:
    """
    Document what needs to change when moving from learning to production
    """
    
    def get_upgrade_checklist(self):
        return {
            'model_architecture': {
                'learning': 'Single two-tower model',
                'production': 'Multi-task learning (CTR + CVR + retention)',
                'changes_needed': [
                    'Add multiple prediction heads',
                    'Implement multi-task loss balancing',
                    'Add model ensemble strategies'
                ]
            },
            'feature_engineering': {
                'learning': 'Basic user/item features',
                'production': 'Real-time feature store integration',
                'changes_needed': [
                    'Implement feature versioning',
                    'Add real-time feature computation',
                    'Build feature quality monitoring'
                ]
            },
            'training_pipeline': {
                'learning': 'Batch training every few days',
                'production': 'Incremental learning + online updates',
                'changes_needed': [
                    'Implement incremental training',
                    'Add model drift detection',
                    'Build automated retraining triggers'
                ]
            },
            'serving_infrastructure': {
                'learning': 'Single Lambda function',
                'production': 'Multi-region deployment with failover',
                'changes_needed': [
                    'Implement circuit breakers',
                    'Add multi-region replication',
                    'Build sophisticated caching layers'
                ]
            },
            'monitoring': {
                'learning': 'Basic CloudWatch metrics',
                'production': 'ML-specific monitoring + business metrics',
                'changes_needed': [
                    'Add model drift detection',
                    'Implement business metric tracking',
                    'Build alerting for ML-specific issues'
                ]
            }
        }
```

### Cost Optimization Strategies

```python
class CostOptimizationStrategies:
    """
    Document cost optimization techniques for different scales
    """
    
    def get_cost_optimization_guide(self):
        return {
            'compute_optimization': {
                'training': {
                    'learning': 'Use Spot Instances (70% savings)',
                    'production': 'Reserved Instances + Spot Fleet',
                    'techniques': [
                        'Schedule training during off-peak hours',
                        'Use smaller instance types with longer training time',
                        'Implement early stopping to avoid overtraining'
                    ]
                },
                'inference': {
                    'learning': 'Serverless Lambda (pay per request)',
                    'production': 'Auto-scaling EC2 + Lambda hybrid',
                    'techniques': [
                        'Right-size inference instances',
                        'Use Lambda for bursty traffic',
                        'Implement intelligent request batching'
                    ]
                }
            },
            'storage_optimization': {
                'data_lake': {
                    'learning': 'S3 Standard → S3 IA after 30 days',
                    'production': 'S3 Intelligent Tiering',
                    'techniques': [
                        'Compress data with Parquet + Snappy',
                        'Partition by date for query efficiency',
                        'Delete intermediate processing files'
                    ]
                },
                'vector_database': {
                    'learning': 'Single-node OpenSearch t3.small',
                    'production': 'Multi-node cluster with appropriate sizing',
                    'techniques': [
                        'Use index compression',
                        'Implement index lifecycle management',
                        'Optimize embedding dimensions'
                    ]
                }
            },
            'monitoring_costs': {
                'learning': '$10-20/month for basic monitoring',
                'production': '$200-500/month for comprehensive monitoring',
                'optimization_tips': [
                    'Use custom metrics sparingly',
                    'Implement log aggregation to reduce costs',
                    'Set up billing alerts early'
                ]
            }
        }
```

---

This comprehensive ML layer documentation provides deep technical detail on implementing an embedding-based hybrid recommendation system using AWS services. The documentation balances learning project simplifications with production-grade patterns, making it valuable both for understanding the concepts and implementing a scalable system.

The key innovations covered include:

1. **Two-tower neural architecture** optimized for fashion e-commerce
2. **4-stage recommendation pipeline** with detailed reasoning for each stage
3. **AWS SageMaker integration** with complete training and inference workflows  
4. **OpenSearch vector database** configuration for efficient similarity search
5. **Comprehensive feature engineering** patterns for recommendation systems
6. **Alternative approaches analysis** to understand when to use different techniques
7. **Production vs learning trade-offs** to understand scaling considerations

The documentation serves as both a technical specification and implementation guide for building production-scale recommendation systems using AWS services.