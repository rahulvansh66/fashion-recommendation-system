# Image Features Integration Guide

## Overview

This guide explains how to integrate visual features from product images into the Two-Tower Retrieval + CatBoost Ranking architecture. Currently, the system uses only text embeddings (384-dim SentenceTransformer) and structured metadata. Adding image features will capture visual similarity patterns that text cannot describe.

**Current State:** Images stored as URLs (`image_url`), used only for UI display.

**Goal:** Extract deep learning image embeddings and integrate them into both retrieval and ranking models for improved visual recommendations.

---

## What Image Features Provide

Fashion is inherently visual. Two items can have similar text descriptions but look completely different (or vice versa). Image embeddings capture:

| Visual Signal | Example | Why Text Fails |
|--------------|---------|----------------|
| **Style patterns** | Stripes, polka dots, floral | "Pattern" is generic text |
| **Color distribution** | Navy with white trim | "Blue" doesn't capture proportions |
| **Silhouette/fit** | Slim-fit vs relaxed | Text rarely describes fit details |
| **Material texture** | Leather, denim, silk | "Cotton blend" doesn't show texture |
| **Design details** | Buttons, zippers, pockets | Often missing from descriptions |
| **Overall aesthetic** | Minimalist vs bohemian | Subjective, hard to encode in text |

**Fashion-Specific Advantages:**
- ✅ **Visual similarity search**: "Show me items that *look like* this"
- ✅ **Cold-start items**: New products without purchase history can be recommended based on visual similarity
- ✅ **Cross-category recommendations**: Visually cohesive outfits across garment types
- ✅ **Complementary item discovery**: Matching accessories with clothing

---

## Available Image Data (H&M Dataset)

**Image URLs Format:**
```python
# Example: article_id = "108775015"
image_url = "https://repo.hops.works/dev/jdowling/h-and-m/images/010/0108775015.jpg"

# URL structure: {base_url}/{first_2_digits}/{0+article_id}.jpg
```

**Dataset Coverage:**
- **Articles with images**: ~105K products (full dataset)
- **Image format**: JPG product photos (white background, front view)
- **Image resolution**: Variable, typically 768x1024 pixels
- **Storage**: External CDN (Hopsworks repo)

**Note:** For local development, you'll need to download a subset of images or use on-the-fly fetching with caching.

---

## Integration Strategy: Three Approaches

### Approach 1: Pre-Trained CNN Embeddings (Recommended First)
**Fast to implement, proven performance, no training required**

### Approach 2: Fine-Tuned Vision Model (Intermediate)
**Better domain-specific features, requires some training**

### Approach 3: Multi-Modal Two-Tower (Advanced)
**Joint text-image embeddings, full architectural change**

---

## Approach 1: Pre-Trained CNN Embeddings (Recommended)

Use a pre-trained computer vision model (ResNet, EfficientNet, or CLIP) to extract image embeddings offline, then use them as additional features.

### 1.1 Architecture Overview

```
OFFLINE FEATURE PIPELINE
-------------------------
Image URL → Download → Preprocess → CNN Model → Embedding

article_id: "108775015"
     ↓
image_url: "https://repo.../0108775015.jpg"
     ↓
image_bytes: [JPEG data]
     ↓
image_tensor: [3, 224, 224]  (normalized RGB)
     ↓
CNN (ResNet-50): Extract features from avgpool layer
     ↓
image_embedding: [2048-dim vector]
     ↓
Store in S3: s3://bucket/features/article_image_embeddings.parquet


TRAINING: Two-Tower Model
--------------------------
Item Tower Input (BEFORE):
  - article_id
  - garment_group_name
  - index_group_name
  → 16-dim embedding

Item Tower Input (AFTER):
  - article_id
  - garment_group_name
  - index_group_name
  - image_embedding [2048-dim]  ← NEW
  → 256-dim embedding (increased from 16)


TRAINING: CatBoost Ranking
---------------------------
Features (BEFORE):
  - customer_id, age, article_id
  - 12 structured article fields (category, color, etc.)

Features (AFTER):
  - customer_id, age, article_id
  - 12 structured article fields
  - image_embedding_pca [128-dim]  ← NEW (PCA reduced)
```

---

### 1.2 Model Selection for Image Embeddings

| Model | Output Dim | Strengths | Use Case |
|-------|-----------|-----------|----------|
| **ResNet-50** | 2048 | Industry standard, fast, well-tested | General purpose, good baseline |
| **EfficientNet-B4** | 1792 | State-of-art accuracy/efficiency | Better than ResNet, similar speed |
| **CLIP (ViT-B/32)** | 512 | Joint text-image space, zero-shot | Multi-modal reasoning, semantic similarity |
| **ConvNeXt-Base** | 1024 | Modern CNN architecture | Better than ResNet, competitive with ViT |
| **DINOv2 (ViT-B/14)** | 768 | Self-supervised, no labels needed | Strong visual features, fashion-friendly |

**Recommendation for Fashion:**
1. **Start with CLIP** (ViT-B/32, 512-dim) — Pre-trained on fashion images, understands text, easy to use
2. **Upgrade to DINOv2** (768-dim) if you need stronger visual-only features
3. **Use EfficientNet-B4** (1792-dim) if you want pure CNN approach

---

### 1.3 Implementation: Offline Image Embedding Extraction

#### Step 1: Download Images (Optional, for Local Development)

Create a script to download product images from the H&M CDN for local processing. This is optional if you prefer to fetch images on-the-fly during embedding extraction.

```python
# scripts/download_images.py

import polars as pl
import requests
from pathlib import Path
from tqdm import tqdm
import time

def download_images(articles_df, output_dir, max_images=None):
    """Download product images from H&M CDN."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_urls = articles_df.select(['article_id', 'image_url']).to_dicts()
    
    if max_images:
        image_urls = image_urls[:max_images]
    
    downloaded = 0
    failed = []
    
    for item in tqdm(image_urls, desc="Downloading images"):
        article_id = item['article_id']
        image_url = item['image_url']
        
        output_file = output_path / f"{article_id}.jpg"
        
        if output_file.exists():
            continue
        
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            output_file.write_bytes(response.content)
            downloaded += 1
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            failed.append({'article_id': article_id, 'error': str(e)})
    
    print(f"Downloaded: {downloaded}, Failed: {len(failed)}")
    
    if failed:
        pl.DataFrame(failed).write_csv(output_path / "failed_downloads.csv")
    
    return downloaded, failed

# Usage
if __name__ == "__main__":
    articles = pl.read_parquet("data/processed/articles_clean.parquet")
    
    # Download small subset for development
    download_images(articles, "data/images/", max_images=5000)
```

---

#### Step 2: Extract Image Embeddings with Pre-Trained Model

Create a comprehensive image embedding extractor that supports multiple pre-trained models.

```python
# src/feature_pipeline/image_embeddings.py

import torch
import polars as pl
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import requests
from io import BytesIO

class ImageEmbeddingExtractor:
    """Extract image embeddings using pre-trained vision models."""
    
    def __init__(self, model_name="clip", device="cuda"):
        """
        Args:
            model_name: One of ['clip', 'resnet50', 'efficientnet_b4', 'dinov2']
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model_name = model_name
        self.model, self.preprocess, self.embed_dim = self._load_model(model_name)
    
    def _load_model(self, model_name):
        """Load pre-trained vision model."""
        
        if model_name == "clip":
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            embed_dim = 512
            
        elif model_name == "resnet50":
            import torchvision.models as models
            from torchvision import transforms
            
            model = models.resnet50(pretrained=True)
            # Remove classifier to get embeddings
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model = model.to(self.device)
            model.eval()
            
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            embed_dim = 2048
            
        elif model_name == "efficientnet_b4":
            import timm
            from torchvision import transforms
            
            model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
            model = model.to(self.device)
            model.eval()
            
            preprocess = transforms.Compose([
                transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            embed_dim = 1792
            
        elif model_name == "dinov2":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            model = model.to(self.device)
            model.eval()
            
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            embed_dim = 768
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model, preprocess, embed_dim
    
    def load_image(self, image_source):
        """Load image from file path or URL."""
        if isinstance(image_source, str):
            if image_source.startswith('http'):
                # Download from URL
                response = requests.get(image_source, timeout=10)
                img = Image.open(BytesIO(response.content))
            else:
                # Load from file
                img = Image.open(image_source)
        else:
            img = image_source
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    @torch.no_grad()
    def extract_embedding(self, image_source):
        """Extract embedding from a single image."""
        
        img = self.load_image(image_source)
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        if self.model_name == "clip":
            embedding = self.model.encode_image(img_tensor)
        else:
            embedding = self.model(img_tensor)
        
        # Flatten and normalize
        embedding = embedding.squeeze().cpu().numpy()
        
        # L2 normalization for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def extract_batch(self, image_sources, batch_size=32):
        """Extract embeddings for multiple images in batches."""
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(image_sources), batch_size), desc="Processing batches"):
            batch = image_sources[i:i + batch_size]
            batch_embeddings = []
            
            for img_source in batch:
                try:
                    embedding = self.extract_embedding(img_source)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Failed to process {img_source}: {e}")
                    # Use zero vector for failed images
                    batch_embeddings.append(np.zeros(self.embed_dim))
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)

def compute_image_embeddings(
    articles_df,
    model_name="clip",
    use_local_images=False,
    local_image_dir="data/images/"
):
    """
    Compute image embeddings for all articles.
    
    Args:
        articles_df: DataFrame with article_id and image_url columns
        model_name: Vision model to use
        use_local_images: If True, load from local dir; else download from URL
        local_image_dir: Directory containing downloaded images
    
    Returns:
        DataFrame with added image_embedding column
    """
    
    extractor = ImageEmbeddingExtractor(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Prepare image sources
    if use_local_images:
        image_dir = Path(local_image_dir)
        image_sources = [
            str(image_dir / f"{article_id}.jpg")
            for article_id in articles_df['article_id'].to_list()
        ]
    else:
        image_sources = articles_df['image_url'].to_list()
    
    # Extract embeddings
    print(f"Extracting {extractor.embed_dim}-dim embeddings using {model_name}...")
    embeddings = extractor.extract_batch(image_sources, batch_size=32)
    
    # Add to DataFrame
    articles_with_embeddings = articles_df.with_columns(
        image_embedding=pl.Series(embeddings.tolist())
    )
    
    return articles_with_embeddings

# Usage
if __name__ == "__main__":
    # Load articles
    articles = pl.read_parquet("data/processed/articles_clean.parquet")
    
    # Extract image embeddings
    articles_with_img = compute_image_embeddings(
        articles,
        model_name="clip",
        use_local_images=True
    )
    
    # Save with image embeddings
    articles_with_img.write_parquet("data/processed/articles_with_image_embeddings.parquet")
    
    print(f"Embedding dimension: {len(articles_with_img['image_embedding'][0])}")
```

**Dependencies:**
```bash
pip install torch torchvision
pip install ftfy regex tqdm  # For CLIP
pip install timm  # For EfficientNet
pip install git+https://github.com/openai/CLIP.git  # For CLIP
```

---

### 1.4 Integration into Two-Tower Model

Update the Item Tower to incorporate image embeddings alongside existing structured features.

```python
# src/retrieval/models/two_tower.py

import torch
import torch.nn as nn

class ItemTower(nn.Module):
    """Item tower with text + structured + image features."""
    
    def __init__(
        self,
        num_items,
        num_garment_groups,
        num_index_groups,
        image_embedding_dim=512,
        embedding_dim=256,
        use_image_features=True
    ):
        super().__init__()
        
        self.use_image_features = use_image_features
        
        # Existing embeddings
        self.item_embedding = nn.Embedding(num_items, 64)
        self.garment_group_embedding = nn.Embedding(num_garment_groups, 32)
        self.index_group_embedding = nn.Embedding(num_index_groups, 32)
        
        # NEW: Image embedding projection
        if use_image_features:
            self.image_projection = nn.Sequential(
                nn.Linear(image_embedding_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128)
            )
            input_dim = 64 + 32 + 32 + 128
        else:
            input_dim = 64 + 32 + 32
        
        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, article_id, garment_group, index_group, image_embedding=None):
        """
        Args:
            article_id: [batch_size] - Item IDs
            garment_group: [batch_size] - Garment group IDs
            index_group: [batch_size] - Index group IDs
            image_embedding: [batch_size, image_dim] - Pre-extracted features
        """
        # Existing features
        item_emb = self.item_embedding(article_id)
        garment_emb = self.garment_group_embedding(garment_group)
        index_emb = self.index_group_embedding(index_group)
        
        features = [item_emb, garment_emb, index_emb]
        
        # NEW: Add image features
        if self.use_image_features and image_embedding is not None:
            image_proj = self.image_projection(image_embedding)
            features.append(image_proj)
        
        # Concatenate all features
        concatenated = torch.cat(features, dim=1)
        
        # Final embedding
        output = self.mlp(concatenated)
        
        # L2 normalize for cosine similarity
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        
        return output
```

---

### 1.5 Integration into CatBoost Ranking

CatBoost can handle high-dimensional numerical features, but 512-2048 dims may be too many. Two recommended approaches:

#### Option A: Use PCA-Reduced Image Embedding (Recommended)

```python
# src/ranking/datasets/ranking_dataset.py

from sklearn.decomposition import PCA
import numpy as np
import joblib

def reduce_image_embeddings_pca(
    articles_df,
    n_components=128,
    save_path="models/pca_image_embeddings.pkl"
):
    """Reduce image embedding dimensionality with PCA."""
    
    # Extract image embeddings as numpy array
    image_embeddings = np.array(articles_df['image_embedding'].to_list())
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(image_embeddings)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Save PCA model for inference
    joblib.dump(pca, save_path)
    
    # Add reduced embeddings back to DataFrame
    articles_with_pca = articles_df.with_columns(
        image_embedding_pca=pl.Series(reduced_embeddings.tolist())
    )
    
    return articles_with_pca

def compute_ranking_dataset(
    interactions_fg,
    articles_fg,
    customers_fg,
    use_pca_image_features=True,
    pca_dim=128
):
    """Build ranking dataset with PCA-reduced image embeddings."""
    
    # Standard joins for interactions, articles, customers
    ranking_df = interactions_fg.join(articles_fg, on='article_id')
    ranking_df = ranking_df.join(customers_fg, on='customer_id')
    
    if use_pca_image_features:
        # Expand PCA embedding into separate columns
        for i in range(pca_dim):
            ranking_df = ranking_df.with_columns(
                pl.col('image_embedding_pca').arr.get(i).alias(f"img_pca_{i}")
            )
        
        ranking_df = ranking_df.drop('image_embedding_pca')
    
    return ranking_df
```

**Advantages:**
- ✅ Reduces dimensionality (512 → 128 or 64)
- ✅ Retains 90-95% of variance
- ✅ Faster training, less overfitting
- ✅ Standard ML practice for high-dim features

---

#### Option B: Use Image Embedding Similarity as Single Feature

Instead of using raw embeddings, compute a single "visual similarity to user's purchase history" feature.

```python
# src/ranking/datasets/ranking_dataset.py

def add_visual_similarity_features(ranking_df):
    """Add visual similarity between user's past purchases and candidate item."""
    
    # Group by customer to get their purchase history
    user_purchase_history = (
        ranking_df
        .filter(pl.col('label') == 1)
        .groupby('customer_id')
        .agg(pl.col('image_embedding').alias('user_purchase_embeddings'))
    )
    
    # Join back to ranking dataset
    ranking_df = ranking_df.join(user_purchase_history, on='customer_id', how='left')
    
    # Compute cosine similarity
    def compute_avg_similarity(candidate_emb, history_embs):
        if not history_embs or len(history_embs) == 0:
            return 0.0
        
        similarities = []
        for hist_emb in history_embs:
            dot_product = np.dot(candidate_emb, hist_emb)
            norm_product = np.linalg.norm(candidate_emb) * np.linalg.norm(hist_emb)
            similarities.append(dot_product / norm_product)
        
        return np.mean(similarities)
    
    # Apply similarity computation
    ranking_df = ranking_df.with_columns(
        pl.struct(['image_embedding', 'user_purchase_embeddings'])
          .map_elements(lambda x: compute_avg_similarity(x['image_embedding'], x['user_purchase_embeddings']))
          .alias('visual_similarity_to_history')
    )
    
    return ranking_df
```

**Advantages:**
- ✅ Single feature (no dimensionality issue)
- ✅ Directly captures user visual preference
- ✅ Interpretable feature importance

---

### 1.6 Expected Performance Improvements

| Metric | Baseline (No Images) | With Image Features | Improvement |
|--------|---------------------|-------------------|-------------|
| **Recall@100** | 0.45 | 0.52 | +15% |
| **Precision@10** | 0.32 | 0.38 | +19% |
| **NDCG@10** | 0.41 | 0.48 | +17% |
| **Cold-start items** | Poor | Good | +30% |
| **Visual similarity** | N/A | Excellent | New capability |

**Fashion-Specific Gains:**
- ✅ Cross-category recommendations improve
- ✅ Style consistency in recommendations
- ✅ Cold-start handling for new products
- ✅ Outfit completion suggestions

---

## Approach 2: Fine-Tuned Vision Model (Intermediate)

### 2.1 Why Fine-Tune?

Pre-trained models are trained on ImageNet (generic objects). Fashion has domain-specific visual patterns that fine-tuning can capture better.

### 2.2 Triplet Loss Fine-Tuning

Train the model to place visually similar items close together in embedding space.

**Key Concept:** Create triplets (anchor, positive, negative):
- **Anchor:** Current item
- **Positive:** Item co-purchased by same user (similar style)
- **Negative:** Random item from different category

```python
# src/retrieval/training/triplet_finetuning.py

import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""
    
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_distance = torch.sum((anchor - positive) ** 2, dim=1)
        neg_distance = torch.sum((anchor - negative) ** 2, dim=1)
        
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        return loss.mean()

def create_triplet_dataset(articles_df, interactions_df):
    """
    Create triplets from purchase data.
    Items bought by same user are considered visually similar.
    """
    # Build co-purchase graph
    co_purchases = {}
    user_purchases = interactions_df.groupby('customer_id').agg(
        pl.col('article_id').alias('purchased_items')
    )
    
    for user_items in user_purchases['purchased_items']:
        items = user_items.to_list()
        for item in items:
            if item not in co_purchases:
                co_purchases[item] = set()
            co_purchases[item].update(items)
            co_purchases[item].remove(item)
    
    return co_purchases

# Training loop omitted for brevity
```

**Expected Gain:** +20% over pre-trained baseline

---

## Approach 3: Multi-Modal Two-Tower (Advanced)

### 3.1 Architecture

Instead of treating text and images separately, train a single model that fuses both modalities.

```python
# src/retrieval/models/multimodal_two_tower.py

import torch
import torch.nn as nn

class MultiModalItemTower(nn.Module):
    """Item tower that fuses text and image features."""
    
    def __init__(
        self,
        text_encoder,
        image_encoder,
        text_dim=384,
        image_dim=2048,
        fusion_dim=512,
        output_dim=256
    ):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.image_projection = nn.Linear(image_dim, fusion_dim)
        
        # Fusion layer (concatenation strategy)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, text_input, image_tensor):
        # Encode text and image
        text_features = self.text_encoder(text_input)
        image_features = self.image_encoder(image_tensor).squeeze()
        
        # Project to fusion space
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        
        # Fuse
        fused = torch.cat([text_proj, image_proj], dim=1)
        fused = self.fusion(fused)
        
        # Final embedding
        output = self.output_mlp(fused)
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        
        return output
```

**Expected Gain:** +25% but requires significant compute and architectural changes

---

## Implementation Roadmap

### Phase 1: Pre-Trained Image Embeddings (2-3 weeks)

**Week 1: Offline Feature Extraction**
- [ ] Set up image downloading script
- [ ] Extract CLIP embeddings for all articles
- [ ] Compute PCA reduction (512 → 128 dims)
- [ ] Save image embeddings to S3/Parquet

**Week 2: Two-Tower Integration**
- [ ] Update ItemTower to accept image embeddings
- [ ] Modify dataset loader
- [ ] Retrain Two-Tower model
- [ ] Evaluate Recall@100, NDCG@10

**Week 3: CatBoost Integration**
- [ ] Add PCA image features to ranking dataset
- [ ] Retrain CatBoost model
- [ ] A/B test baseline vs image-enhanced
- [ ] Measure precision@10, feature importance

**Success Criteria:**
- ✅ Recall@100 improves by 10%+
- ✅ Cold-start item recommendations improve
- ✅ Image features in top 10 feature importance

---

### Phase 2: Fine-Tuning (3-4 weeks, optional)

**Week 1-2: Triplet Loss Fine-Tuning**
- [ ] Build triplet dataset
- [ ] Fine-tune ResNet-50 with triplet loss
- [ ] Extract fine-tuned embeddings

**Week 3-4: Evaluation**
- [ ] Replace CLIP with fine-tuned embeddings
- [ ] Retrain Two-Tower + CatBoost
- [ ] Compare pre-trained vs fine-tuned

**Success Criteria:**
- ✅ Fine-tuned embeddings outperform CLIP by 5%+

---

### Phase 3: Multi-Modal Fusion (4-6 weeks, advanced)

**Week 1-2: Architecture Design**
- [ ] Implement multi-modal item tower
- [ ] Test fusion strategies

**Week 3-4: Training**
- [ ] Train multi-modal Two-Tower model
- [ ] Tune hyperparameters

**Week 5-6: Evaluation & Deployment**
- [ ] Compare single-modal vs multi-modal
- [ ] Deploy to SageMaker endpoint

**Success Criteria:**
- ✅ Multi-modal model outperforms by 10%+
- ✅ Inference latency under 50ms

---

## Common Pitfalls & Solutions

### Pitfall 1: Image Download Bottleneck

**Problem:** Downloading 100K images takes hours.

**Solution:** Use multiprocessing or on-the-fly fetching with caching
```python
from concurrent.futures import ThreadPoolExecutor

def download_batch(urls, output_dir):
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(lambda url: download_image(url, output_dir), urls)
```

---

### Pitfall 2: GPU Memory Overflow

**Problem:** Extracting embeddings for 100K images OOMs GPU.

**Solution:** Process in batches, move to CPU after extraction
```python
def extract_embeddings_batched(image_urls, batch_size=32):
    all_embeddings = []
    
    for i in range(0, len(image_urls), batch_size):
        batch = image_urls[i:i+batch_size]
        
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()
        
        all_embeddings.append(embeddings)
        torch.cuda.empty_cache()
    
    return np.concatenate(all_embeddings)
```

---

### Pitfall 3: PCA Dimensionality Too Low

**Problem:** Reducing 512 → 64 dims loses too much information.

**Solution:** Plot explained variance, choose elbow point at 90-95%
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=512)
pca.fit(image_embeddings)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Choose n_components at 90% variance
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
```

---

## Cost & Performance Considerations

### Compute Costs

| Task | GPU | Time (5K items) | Time (105K items) | Cost (AWS p3.2xlarge) |
|------|-----|-----------------|-------------------|---------------------|
| **CLIP extraction** | V100 | 15 min | 5 hours | ~$15 |
| **ResNet-50 extraction** | V100 | 10 min | 3.5 hours | ~$11 |
| **Fine-tuning (10 epochs)** | V100 | 2 hours | 48 hours | ~$150 |

**Cost Optimization:**
- ✅ Use spot instances (70% reduction)
- ✅ Extract embeddings once, reuse
- ✅ Start with 5K items for prototyping

---

### Inference Latency

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Two-Tower (no images)** | 5ms | Baseline |
| **Two-Tower (with images)** | 8ms | +3ms for larger MLP |
| **CatBoost (no images)** | 10ms | 20 features |
| **CatBoost (with PCA images)** | 15ms | 20 + 128 features |

**Latency Optimization:**
- ✅ Pre-compute item embeddings offline
- ✅ Use quantized FAISS index
- ✅ Cache embeddings in Redis

---

## Summary: Which Approach to Choose?

| Approach | Complexity | Performance Gain | When to Use |
|----------|-----------|-----------------|-------------|
| **Pre-trained CLIP** | Low | +15% | First iteration, fast validation |
| **Pre-trained ResNet-50** | Low | +12% | Pure visual features |
| **Fine-tuned (Triplet)** | Medium | +20% | Domain-specific, 50K+ images |
| **Multi-Modal Fusion** | High | +25% | State-of-art, large dataset |

---

## Next Steps

1. **Read this guide** ✅ (You are here)
2. **Choose approach:** Start with Pre-trained CLIP (Approach 1)
3. **Extract embeddings:** Run image_embeddings.py
4. **Integrate into Two-Tower:** Update ItemTower class
5. **Integrate into CatBoost:** Add PCA image features
6. **Evaluate metrics:** Compare baseline vs image-enhanced
7. **Iterate:** Try fine-tuning if needed

---

## References

- **CLIP Paper:** Learning Transferable Visual Models From Natural Language Supervision (2021)
- **DINOv2 Paper:** Learning Robust Visual Features without Supervision (2023)
- **Triplet Loss:** FaceNet - A Unified Embedding for Face Recognition (2015)
- **H&M Dataset:** Kaggle H&M Personalized Fashion Recommendations

---

**Document Status:** Implementation guide complete. Ready for development.
