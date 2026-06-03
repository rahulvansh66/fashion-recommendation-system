---
⚠️ **REFERENCE PROJECT DISCLAIMER** ⚠️

**THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION**

- **DO NOT USE** unless explicitly asked to reference old code
- **CURRENT IMPLEMENTATION** is in `system-design/` directory
- This file is for **REFERENCE ONLY** to understand legacy approaches
- All new development should follow current system design specifications

---

# Inference Pipeline Analysis

## Overview

This analysis documents the real-time inference pipeline implementation for the fashion recommendation system. The implementation leverages Hopsworks for model serving, vector similarity search, and real-time feature engineering, integrated with both traditional ML models (CatBoost) and LLM-based ranking systems.

## Architecture Overview

The inference pipeline consists of two main components:
1. **Query Model Pipeline**: Handles user query processing and candidate retrieval via vector similarity search
2. **Ranking Model Pipeline**: Scores and ranks candidates using either CatBoost or LLM-based models

Both pipelines are deployed as real-time services on Hopsworks with REST API endpoints.

## Vector Database Integration

### Embedding Computation
- **Model**: `all-MiniLM-L6-v2` sentence transformer model for feature embeddings
- **Implementation**: Located in `4_ip_computing_item_embeddings.ipynb`
- **Process**:
  ```python
  # Preprocess items for embedding
  item_df = features.embeddings.preprocess(train_df, candidate_features)
  
  # Compute embeddings using candidate model
  embeddings_df = features.embeddings.embed(df=item_df, candidate_model=candidate_model)
  ```

### Vector Similarity Search
- **Database**: Hopsworks Feature Store with built-in vector indexing
- **Implementation**: Located in `ranking_transformer.py`
- **Search Logic**:
  ```python
  # Search for candidate items using query embedding
  neighbors = self.candidate_index.find_neighbors(
      inputs["query_emb"],
      k=100,  # Retrieve top 100 candidates
  )
  ```

### Vector Index Configuration
- **Feature Group**: `candidate_embeddings` with online enabled
- **Index**: Embedding index on 16-dimensional vectors
- **Storage**: 11,820 item embeddings stored in feature group
- **Feature View**: `candidate_embeddings` for real-time serving

## Deployment Pipeline Architecture

### Model Deployment Components

#### 1. Ranking Model Deployment
```python
# Deploy CatBoost ranking model
ranking_deployment = hopsworks_integration.ranking_serving.HopsworksRankingModel.deploy(
    project=project
)
```

**Features:**
- CatBoost model with secure serialization
- Custom `Predict` class for Hopsworks integration
- Probability prediction for positive class
- Article ID tracking and score mapping

#### 2. Query Model Deployment
```python
# Deploy query processing model
query_model_deployment = hopsworks_integration.two_tower_serving.HopsworksQueryModel.deploy(
    ranking_model_type="ranking"  # or "llmranking"
)
```

**Features:**
- Two-tower architecture for query and item embeddings
- Support for both traditional and LLM ranking backends
- Real-time feature engineering integration

#### 3. LLM Ranking Deployment
```python
# Deploy LLM-based ranking model
ranking_deployment = hopsworks_integration.llm_ranking_serving.HopsworksLLMRankingModel.deploy()
```

**Features:**
- OpenAI GPT-4o-mini integration via LangChain
- Natural language reasoning for ranking decisions
- Limited to 20 candidates for performance reasons

## API Design

### Request/Response Formats

#### Query Model API
**Request:**
```json
[{
  "customer_id": "d327d0ad9e30085a436933dfbb7f77cf42e38447993a078ed35d93e3fd350ecf",
  "transaction_date": "2022-11-15T12:16:25.330916"
}]
```

**Response:**
```json
{
  "predictions": {
    "ranking": [
      [0.8542, "670079001"],
      [0.7234, "299768002"], 
      [0.6891, "324946001"]
    ]
  }
}
```

#### Ranking Model API
**Request:**
```json
[{
  "customer_id": "customer_hash",
  "month_sin": 1.2246467991473532e-16,
  "query_emb": [0.214135289, 0.571055949, ...],
  "month_cos": -1.0
}]
```

**Response:**
```json
{
  "predictions": {
    "ranking": [
      [0.9234, "592846001"],
      [0.8765, "536139006"],
      [0.8123, "408554004"]
    ]
  }
}
```

### Error Handling
- **Connection Errors**: Automatic retry with exponential backoff
- **Timeout Handling**: 60-second timeout for LLM predictions
- **Fallback Scores**: Default score of 0 for failed LLM predictions
- **Logging**: Comprehensive logging at INFO level for debugging

## Real-time Feature Engineering

### Customer Feature Integration
```python
# Retrieve customer features in real-time
customer_features = self.customer_fv.get_feature_vector(
    {"customer_id": customer_id},
    return_type="pandas",
)
```

### Temporal Feature Processing
```python
# Calculate temporal features using on-demand transformation
feature_vector = self.ranking_fv._batch_scoring_server.compute_on_demand_features(
    feature_vectors=pd.DataFrame([inputs]), 
    request_parameters={"month": month_of_purchase}
)
```

### Article Feature Enrichment
```python
# Batch retrieve article features for candidates
articles_data = [
    self.articles_fv.get_feature_vector({"article_id": item_id})
    for item_id in item_id_list
]
```

## Streamlit Web Interface Implementation

### Application Architecture
- **Entry Point**: `streamlit_app.py` with customer selection and page routing
- **Components**: Modular UI components in `recsys/ui/` directory
- **State Management**: Streamlit session state for user interactions

### Key Features

#### 1. Customer Recommendations Page
```python
def customer_recommendations(articles_fv, ranking_deployment, query_model_deployment, customer_id):
    # Process customer request through full pipeline
    # Display ranked recommendations with interaction tracking
```

#### 2. LLM Recommendations Page  
```python
def llm_recommendations(articles_fv, openai_api_key, customer_id):
    # Natural language query interface
    # LLM-powered ranking and explanation
```

#### 3. Interaction Tracking
```python
class InteractionTracker:
    def track(self, customer_id, item_id, interaction_type):
        # Track clicks, views, purchases
        # Update feature groups in real-time
```

### Performance Optimizations

#### 1. Deployment Management
```python
# Initialize deployments with caching
with st.spinner("🚀 Starting Deployments..."):
    articles_fv, ranking_deployment, query_model_deployment = get_deployments()

# Stop deployments on demand
if st.button("⏹️ Stop Deployments"):
    ranking_deployment.stop()
    query_model_deployment.stop()
```

#### 2. Image Processing Pipeline
```python
def fetch_and_process_image(image_url):
    # Fetch product images with caching
    # Resize and optimize for display
```

#### 3. Feature Caching
- **Customer Features**: Cached in Hopsworks feature views
- **Article Metadata**: Batch retrieval and caching
- **Embeddings**: Pre-computed and indexed for fast retrieval

## Performance Optimization Techniques

### 1. Candidate Filtering
```python
# Filter out already purchased items
already_bought_items_ids = (
    self.transactions_fg.select("article_id")
    .filter(self.transactions_fg.customer_id==customer_id)
    .read(dataframe_type="pandas")
    .values.reshape(-1).tolist()
)

item_id_list = [
    str(item_id)
    for item_id in neighbors
    if str(item_id) not in already_bought_items_ids
]
```

### 2. Batch Processing
- **Feature Retrieval**: Batch article feature lookups
- **Embedding Search**: Single vector search for multiple candidates
- **Model Inference**: Batch prediction for ranking model

### 3. LLM Optimization
```python
# Limit candidates for LLM ranking due to latency constraints
features = inputs[0].pop("ranking_features")[:20]  # Max 20 candidates
article_ids = inputs[0].pop("article_ids")[:20]
```

### 4. Caching Strategy
- **Model Caching**: Models cached in Hopsworks deployments
- **Feature Caching**: Feature views with serving optimization
- **Session Caching**: Streamlit session state for UI responsiveness

## Infrastructure and Deployment

### Hopsworks Integration
- **Feature Store**: Centralized feature management and serving
- **Model Registry**: Versioned model artifacts and metadata
- **Deployments**: Serverless inference endpoints
- **Secrets Management**: Secure API key storage (OpenAI, etc.)

### Scaling Configuration
- **Auto-scaling**: Hopsworks automatic resource scaling
- **Load Balancing**: Built-in load balancing for deployments  
- **Monitoring**: Deployment logs and metrics in Hopsworks UI

### API Performance Metrics
- **Query Model**: ~2-3 seconds end-to-end latency
- **Ranking Model**: ~1-2 seconds for 100 candidates
- **LLM Ranking**: ~15-30 seconds for 20 candidates (due to OpenAI API)

## Error Handling and Monitoring

### Logging Strategy
```python
logging.info(f"✅ Inputs: {inputs}")
logging.info(f"🦅 Predicting with OpenAI model for {len(features)} instances")
logging.info(f"LLM Scores: {scores}")
```

### Error Recovery
- **Model Failures**: Graceful degradation with default scores
- **API Timeouts**: Retry logic with exponential backoff
- **Feature Retrieval Errors**: Fallback to default values

### Monitoring Points
- **Deployment Status**: Real-time deployment health checks
- **API Response Times**: Latency monitoring per endpoint
- **Model Performance**: Prediction score distributions
- **User Interactions**: Click-through rates and engagement metrics

## Integration Points

### External Services
- **OpenAI API**: LLM ranking with GPT-4o-mini
- **Hopsworks Cloud**: Feature store and model serving
- **Image CDN**: Product image serving and caching

### Data Dependencies
- **Feature Views**: Real-time feature serving
- **Vector Index**: Similarity search infrastructure  
- **Transaction History**: Customer interaction filtering
- **Article Catalog**: Product metadata and features

This inference pipeline demonstrates a production-ready recommendation system with hybrid ML/LLM approaches, real-time feature engineering, and scalable serving infrastructure.
---
⚠️ **END OF REFERENCE PROJECT FILE** ⚠️

Remember: This is archived code. Use `system-design/` for current implementation.

---
