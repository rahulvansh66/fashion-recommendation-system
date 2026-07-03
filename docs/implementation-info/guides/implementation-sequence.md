# Implementation Sequence Guide

**Purpose:** This document outlines the sequence of notebooks (for experimentation and prototyping) and pipelines (for productionization) required to build the Fashion Recommendation System. It aligns with the guides on EDA, pre-processing, and feature engineering.

---

## 1. Notebook Sequence (Experimentation & Prototyping)

Notebooks are used to interactively explore data, define logic, and prototype models before writing production pipeline code. 

**Important Note on Data Scale:** Notebook 0 and Notebook 1 operate on the **full dataset** to create and validate a representative sample. Notebooks 2 through 6 operate exclusively on the **sampled data** to allow for fast, iterative prototyping.

They should be executed in the following order:

### Notebook 0: `00_stratified_user_sampling.ipynb`
* **Purpose:** Create a reproducible, proportion-preserving dev sample (and a tiny dummy subset) from the full H&M dataset.
* **Actions:**
  * Read the full raw CSVs (`articles`, `customers`, `transactions_train`).
  * Apply a temporal split (`cutoff = 2020-03-31`) to label users by purchase tier and recency using only pre-cutoff history.
  * Allocate quotas and draw a stratified sample of users (e.g., ~1,000 or ~2,000 users).
  * Filter the transactions and articles to match the sampled users.
  * Write the sampled data as Parquet files (Hive layout) to be used by all subsequent notebooks.
* **Reference:** [`stratified_user_sampling.ipynb`](../../../notebooks/stratified_user_sampling.ipynb)

### Notebook 1: `01_raw_data_eda_and_cleaning.ipynb`
* **Purpose:** Unsupervised EDA on the full dataset, data quality checks, and sample validation.
* **Actions:**
  * **Full Dataset EDA:** Look at global distributions, long-tail item popularity, overall seasonal trends, and missing data patterns across the entire population. This informs temporal framing and negative sampling strategies.
  * **Sample Validation:** Compare the distributions (e.g., age, price, category popularity) of the sampled data against the full dataset to ensure the sample is truly representative and hasn't introduced bias.
  * Define strategies for missing values, outliers, and category grouping based on these observations.
* **Reference:** [`eda-preprocessing-guide.md`](./eda-preprocessing-guide.md)
* **Note:** As per the generic EDA guide, **do include** checks for dataset shape (Step 2A), data quality/missingness (Step 2B), individual feature distributions/outliers (Step 2C), and unsupervised feature relationships (Step 2E). **Do NOT include** Feature vs. Target analysis (Step 2D) here, because the target label (future purchases) has not been constructed yet.

### Notebook 2: `02_temporal_framing_and_labels.ipynb`
* **Purpose:** Implement problem framing, core data cleaning, and label generation logic on the sampled data.
* **Actions:**
  * Ingest sampled Parquet files, apply type-casting, and filter invalid rows.
  * Enrich transactions with article and customer dimension data.
  * Establish the temporal framework (`snap_date` cutoffs and 7-day forward label windows).
  * Generate positive labels (purchases in the 7-day forward window) and sample window-aware negatives. *This creates the target label (1 or 0).*
* **Pre-processing Notes (per `pre-processing-guide.md`):**
  * **Schema Normalization & Null Handling:** Parse dates, cap ages (16-100), impute missing ages, standardize strings, and drop null join keys.
  * **Outliers & Validity:** Drop negative prices, winsorize extreme prices (e.g., 99th percentile), and deduplicate transactions.
  * **Orphan Records:** Inner-join transactions to articles/customers and drop orphans.
  * **Temporal Framing & Leakage Prevention:** Establish strict `snap_date` cutoffs and generate window-aware negative samples (1:5 ratio, exclude seen items).
* **Reference:** [`pre-processing-guide.md`](./pre-processing-guide.md)

### Notebook 3: `03_feature_engineering.ipynb`
* **Purpose:** Prototype the generation of raw predictive signals.
* **Actions:**
  * Calculate **item features** (popularity, recency, price stats).
  * Calculate **user features** (category/color preferences, purchase frequency).
  * Calculate **user-item cross features** (repurchase counts, price vs budget z-scores).
  * Apply temporal decay weights (180-day half-life).
  * Perform post-FE null imputation.
  * *Crucially, all features are computed strictly using pre-cutoff data.*
* **Pre-processing Notes (per `pre-processing-guide.md`):**
  * **Sparsity & Active-Item Filters (Post-FE):** Drop ranking pair rows where `item_pop_30d` equals zero (dead SKUs filter).
  * **Post-FE Imputation:** Fill count features and cross-feature nulls with 0 default imputation.
* **Reference:** [`features-eng.md`](./features-eng.md)

### Notebook 4: `04_supervised_eda_and_feature_selection.ipynb`
* **Purpose:** Feature vs. Target analysis and dimensionality reduction.
* **Actions:**
  * Join the features from Notebook 3 with the labels from Notebook 2.
  * **Supervised EDA:** Analyze Feature vs. Target (e.g., *Does a higher `user_item_repurchase_30d` correlate with a label of 1?*).
  * **Feature Selection:** Run a correlation matrix to find redundant/collinear features.
  * Train a quick baseline Random Forest or XGBoost to check Feature Importance and SHAP values.
  * Drop features with near-zero variance or no predictive power to finalize the model input schema.
* **Pre-processing Notes (per `pre-processing-guide.md`):**
  * **Feature Formatting:** Pass native categoricals to XGBoost (no target encoding in v1).
* **Reference:** [`eda-preprocessing-guide.md`](./eda-preprocessing-guide.md)
* **Note 2 (Baseline vs. Final XGBoost):** If training an XGBoost model in both Notebook 4 and Notebook 6 seems redundant, note that their **goals, setup, and scale** are completely different. Notebook 4 is a fast baseline for feature selection, while Notebook 6 is for metric optimization.

### Notebook 5: `05_two_tower_retrieval_experiments.ipynb`
* **Purpose:** Prototype the Two-Tower retrieval model architecture.
* **Actions:**
  * Format preprocessed data for retrieval (build vocabularies, normalize age).
  * Build and train the PyTorch Two-Tower model using in-batch negative sampling.
  * Generate embeddings and test FAISS indexing.
* **Pre-processing Notes (per `pre-processing-guide.md`):**
  * **Feature Formatting:** Build string-to-index vocabularies on training rows only. Z-score normalize the `age` column using train-set mean and std.

### Notebook 6: `06_xgboost_ranking_experiments.ipynb`
* **Purpose:** Prototype the XGBoost ranker.
* **Actions:**
  * Join finalized engineered features to the label dataset (positives + sampled negatives).
  * Train XGBoost model using proper `scale_pos_weight`.
  * Evaluate ranking metrics (e.g., MAP@10, NDCG) across validation and test temporal splits.
* **Pre-processing Notes (per `pre-processing-guide.md`):**
  * **Feature Formatting:** Pass native categoricals to XGBoost. Use `scale_pos_weight = 5` for the 1:5 negative ratio setup.

---

## 2. Pipeline Sequence (Productionization)

Once notebook logic is validated, it translates into automated offline pipelines (e.g., AWS Glue, Step Functions, SageMaker Pipelines). They execute in the following dependency order on the **full dataset**:

### Pipeline 1: Data Ingestion & Cleaning Pipeline
* **Purpose:** Convert raw data into a reliable, queryable `clean/` layer.
* **Actions:**
  * Read raw transactions, articles, and customers from S3.
  * Apply schema normalization, type-casting, and strict null handling (parse dates, cap ages 16-100, impute missing ages).
  * Handle outliers & validity (drop negative prices, winsorize extreme prices at 99th percentile, deduplicate transactions keeping sequence ID).
  * Perform dimension enrichment (inner-join articles and customers to transactions, dropping orphans).
  * Write validated Parquet files to the `clean/` bucket prefix.

### Pipeline 2: Feature Engineering Pipeline
* **Purpose:** Generate and store model-ready feature sets.
* **Actions:**
  * Group historical data using the defined `snap_date` schedule.
  * Execute Spark/Glue jobs to compute item, user, user-item, and transaction features adhering to strict temporal cutoffs (preventing leakage).
  * Apply post-FE imputation (fill count features and cross-feature nulls with 0 default imputation).
  * Apply sparsity filters (e.g., dropping dead SKUs where `item_pop_30d` equals zero).
  * Export materialized feature tables to the `features/` bucket prefix.

### Pipeline 3: Retrieval (Two-Tower) Training Pipeline
* **Purpose:** Train the candidate generation model and build the vector index.
* **Actions:**
  * **Pre-process:** Build string-to-index vocabularies on training rows only. Z-score normalize the `age` column using train-set mean and std.
  * **Train:** Launch SageMaker training job for the PyTorch Two-Tower model.
  * **Export:** Run batch inference to generate item embeddings and build the FAISS index artifact.

### Pipeline 4: Ranking (XGBoost) Training Pipeline
* **Purpose:** Train the final ranking model.
* **Actions:**
  * **Pre-process:** Construct the final pair-row dataset by joining labels, window-aware negatives (1:5 ratio, excluding seen items), and precomputed features. Format features by passing native categoricals (no target encoding).
  * **Train:** Launch SageMaker training job for XGBoost with `scale_pos_weight = 5`.
  * **Export:** Save the trained XGBoost model artifact.

### Pipeline 5: Online Serving & Cache Pre-warming Pipeline
* **Purpose:** Deploy artifacts to the serving infrastructure.
* **Actions:**
  * Load the latest FAISS index into the serving environment.
  * Push the most recent user and item feature tables into Redis for low-latency online retrieval.
  * Update SageMaker Serverless endpoints with the newly trained XGBoost model.