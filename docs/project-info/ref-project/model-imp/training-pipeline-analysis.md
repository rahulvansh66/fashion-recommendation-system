# Training Pipeline Analysis - Fashion Recommendation System

## Overview

This document provides a comprehensive analysis of the training pipeline implementation for the fashion recommendation system based on the H&M dataset. The system implements a two-stage approach: a two-tower model for candidate retrieval and a CatBoost model for ranking optimization.

## Two-Tower Model Architecture

### System Design
The two-tower architecture consists of separate neural networks for encoding user queries and item candidates into a shared embedding space. The model is designed for efficient retrieval by computing similarity between query and candidate embeddings.

### Customer/Query Encoder Architecture

**Input Features:**
- `customer_id`: Categorical user identifier  
- `age`: Continuous numerical feature (normalized)
- `month_sin`: Temporal feature (sine encoding of purchase month)
- `month_cos`: Temporal feature (cosine encoding of purchase month)

**Network Architecture:**
```python
# User embedding layer
StringLookup(vocabulary=user_ids) → Embedding(vocab_size + 1, embedding_dim)

# Age normalization
Normalization(axis=None)

# Feature concatenation
[user_embedding, normalized_age, month_sin, month_cos] → concat

# Feed-forward network
Dense(embedding_dim, activation="relu") → Dense(embedding_dim)
```

**Architecture Specifications:**
- Embedding dimension: 16 (configurable via `TWO_TOWER_MODEL_EMBEDDING_SIZE`)
- User vocabulary size: Dynamic based on training data (~966 unique customers)
- Final output: 16-dimensional user embedding vector
- Activation: ReLU for hidden layer, linear for output

### Item/Candidate Encoder Architecture

**Input Features:**
- `article_id`: Categorical item identifier
- `garment_group_name`: Categorical feature (e.g., "Jersey Fancy", "Knitwear")
- `index_group_name`: Categorical feature (e.g., "Ladieswear", "Menswear")

**Network Architecture:**
```python
# Item embedding layer
StringLookup(vocabulary=item_ids) → Embedding(vocab_size + 1, embedding_dim)

# Categorical feature encoding (one-hot)
StringLookup(garment_groups) → one_hot_encoding
StringLookup(index_groups) → one_hot_encoding

# Feature concatenation  
[item_embedding, garment_group_onehot, index_group_onehot] → concat

# Feed-forward network
Dense(embedding_dim, activation="relu") → Dense(embedding_dim)
```

**Architecture Specifications:**
- Embedding dimension: 16 (same as query tower)
- Item vocabulary size: Dynamic (~11,820 unique items)
- Garment groups: One-hot encoded categorical features
- Index groups: One-hot encoded categorical features
- Final output: 16-dimensional item embedding vector

## Training Configuration

### Two-Tower Model Hyperparameters
- **Learning rate**: 0.01
- **Batch size**: 2048
- **Number of epochs**: 10
- **Weight decay**: 0.001
- **Optimizer**: AdamW
- **Embedding dimension**: 16
- **Validation split**: 0.1
- **Test split**: 0.1

### Loss Function and Training Objective
- **Task**: Retrieval using `tfrs.tasks.Retrieval`
- **Loss computation**: Factorized top-k loss with candidate sampling
- **Metrics**: `FactorizedTopK` with top-1, top-5, top-10, top-50, top-100 accuracy
- **Regularization**: L2 weight decay applied to all trainable variables

### Training Process
1. **Data preprocessing**: Feature normalization and vocabulary extraction
2. **Model initialization**: Initialize age normalization layer with training data statistics
3. **Gradient computation**: Custom train_step with gradient tape
4. **Evaluation**: Custom test_step computing retrieval metrics
5. **Early stopping**: Based on validation loss monitoring

## CatBoost Ranking Model

### Model Configuration
```python
CatBoostClassifier(
    learning_rate=0.2,
    iterations=100, 
    depth=10,
    scale_pos_weight=10,
    early_stopping_rounds=5,
    use_best_model=True
)
```

### Training Features
**Input Features Used:**
- `age`: Customer age at time of purchase
- `product_type_name`: Specific product type classification
- `product_group_name`: Higher-level product grouping
- `graphical_appearance_name`: Visual pattern/design features
- `colour_group_name`: Color classification
- `perceived_colour_value_name`: Perceived color intensity
- `perceived_colour_master_name`: Primary color classification
- `department_name`: Retail department classification
- `index_name`: Internal product classification
- `index_group_name`: Gender/category grouping
- `section_name`: Store section classification  
- `garment_group_name`: Garment type classification
- `month_sin`, `month_cos`: Temporal purchase patterns

### Hyperparameters Analysis
- **Learning rate**: 0.2 (relatively high for gradient boosting)
- **Max iterations**: 100 with early stopping at 5 rounds
- **Tree depth**: 10 (deep trees for complex pattern capture)
- **Scale pos weight**: 10 (handles class imbalance in purchase data)
- **Early stopping**: Prevents overfitting with 5-round patience

## Evaluation Methodology

### Two-Tower Model Evaluation
**Metrics Used:**
- Top-K categorical accuracy (K = 1, 5, 10, 50, 100)
- Retrieval loss (factorized top-k loss)
- Regularization loss tracking
- Total loss (retrieval + regularization)

**Evaluation Process:**
1. Generate query embeddings for validation users
2. Retrieve top-K most similar items from candidate pool  
3. Measure hit rate: fraction of actual purchases in top-K retrievals
4. Monitor training/validation loss progression

**Performance Results:**
The training logs show validation metrics consistently at 0.0000 across all top-K accuracies, indicating potential issues with the model configuration or evaluation setup in this specific run.

### CatBoost Ranking Model Evaluation

**Metrics Used:**
- Precision, Recall, F1-score (binary classification)
- Classification accuracy
- Feature importance analysis

**Performance Results:**
```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00     38778
           1       0.96      1.00      0.98      1942

    accuracy                           1.00     40720
   macro avg       0.98      1.00      0.99     40720
weighted avg       1.00      1.00      1.00     40720
```

**Feature Importance Analysis:**
Most important features identified by CatBoost:
1. `month_cos`: 58.9% (seasonal purchase patterns)
2. `month_sin`: 33.6% (temporal purchase behavior)  
3. `product_type_name`: 1.5% (specific product classification)
4. `age`: 1.5% (customer demographics)
5. `perceived_colour_value_name`: 0.9% (color preferences)

## Training Pipeline Integration

### Data Flow Architecture
1. **Feature Views**: Hopsworks feature store integration for data access
2. **Two-Tower Training**: Retrieval model training with embedding generation
3. **Model Registry**: Hopsworks model registry for versioning and deployment
4. **Ranking Training**: CatBoost model training on enriched feature set
5. **Performance Evaluation**: Comprehensive metrics collection and analysis

### Dataset Specifications
**Two-Tower Dataset:**
- Training samples: 16,300
- Validation samples: 2,037  
- Unique users: 966
- Unique items: 11,820

**Ranking Dataset:**
- Validation split: 0.1 (10% of data)
- Feature count: 14 categorical and numerical features
- Class balance: Handled via `scale_pos_weight=10`

### Model Deployment Pipeline
1. **Model Serialization**: TensorFlow SavedModel format for two-tower components
2. **Registry Upload**: Automated model versioning in Hopsworks
3. **Feature Pipeline**: Integration with feature store for real-time serving
4. **Inference Optimization**: Separated query and candidate encoders for efficient serving

## Technical Implementation Details

### Memory and Performance Considerations
- **Batch processing**: Large batch sizes (2048) for efficient GPU utilization
- **Data caching**: `.cache()` operations on TensorFlow datasets
- **Shuffle buffering**: 10x batch size shuffle buffer for training randomization
- **Feature preprocessing**: Efficient string tokenization and one-hot encoding

### Model Architecture Decisions
- **Embedding dimensions**: 16D chosen for memory efficiency vs. expressiveness trade-off
- **Two-stage design**: Retrieval → Ranking for scalable recommendation serving
- **Categorical handling**: String lookup + embedding for IDs, one-hot for categories
- **Temporal encoding**: Sin/cos transformation preserves cyclical time patterns

## Performance Analysis and Optimization Opportunities

### Current Limitations
1. **Two-tower metrics**: All validation accuracies at 0.0% suggest training issues
2. **Feature coverage**: Limited feature set may constrain model expressiveness
3. **Embedding dimension**: 16D may be insufficient for complex user-item relationships
4. **Evaluation dataset**: Small validation set (2,037 samples) may not be representative

### Recommended Improvements
1. **Feature engineering**: Add image embeddings, price features, seasonal indicators
2. **Architecture scaling**: Experiment with larger embedding dimensions (32, 64, 128)
3. **Advanced loss functions**: Implement listwise ranking losses for better optimization
4. **Ensemble methods**: Combine multiple retrieval models for improved coverage
5. **Hyperparameter tuning**: Systematic search for optimal learning rates and regularization

## Conclusion

The training pipeline implements a modern two-stage recommendation architecture with clear separation between retrieval and ranking components. The CatBoost ranking model shows strong performance with 98% F1-score, while the two-tower retrieval model requires further investigation and optimization. The temporal features dominate the ranking model, indicating strong seasonal patterns in fashion purchases that the system successfully captures.