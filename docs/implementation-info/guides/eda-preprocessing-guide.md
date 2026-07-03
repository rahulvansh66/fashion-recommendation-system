## ML Model Cheatsheet: EDA → Preprocessing Decisions

### 1) Start with the problem

Before touching the data, clarify:

* **Target**: what are you predicting?
* **Task type**: classification, regression, ranking, clustering, time series
* **Success metric**: accuracy, F1, RMSE, AUC, etc.
* **Unit of prediction**: user, transaction, session, day, product
* **Constraints**: latency, interpretability, memory, fairness

This tells you what kind of EDA matters most.

---

## 2) EDA approach: what to check and why

### A. Understand the dataset shape

Check:

* number of rows and columns
* column types: numeric, categorical, datetime, text, ID
* target distribution
* duplicate rows
* train/test split leakage risks

**Use this to decide:**

* drop duplicate rows if they are unintended
* exclude pure identifiers from modeling
* choose split strategy early if there is time/user/product grouping

---

### B. Check data quality

Look for:

* missing values
* impossible values
* inconsistent categories
* wrong data types
* formatting issues
* constant or near-constant columns

**Use this to decide:**

* impute, drop, or create missing flags
* fix type conversion
* standardize category labels
* remove useless columns
* cap or correct invalid values

---

### C. Study each feature one by one

For numeric features:

* distribution
* skewness
* outliers
* range
* zero inflation
* multimodality

For categorical features:

* number of unique values
* rare categories
* dominant categories
* category imbalance

For datetime features:

* trend over time
* seasonality
* gaps
* leakage from future timestamps

For text features:

* length distribution
* empty/short texts
* language consistency
* special characters

**Use this to decide:**

* log transform skewed numeric features
* winsorize or cap extreme outliers
* group rare categories into “Other”
* extract date parts
* use text cleaning or vectorization

---

### D. Analyze feature vs target

This is the most important EDA step.

Check:

* how each feature relates to the target
* class separation
* monotonic patterns
* thresholds
* interaction effects
* leakage features that almost directly reveal target

**Methods to measure relationships (Feature Selection):**

*(Always check required assumptions before applying the respective hypothesis test or correlation metric!)*

* **Linear/Monotonic vs. Target:** 
  * Use **Pearson Correlation** for strict linear relationships (assumes normality, linearity, homoscedasticity).
  * Use **Spearman’s Rank Correlation** for monotonic relationships (highly recommended for tree-based models like XGBoost, as it is robust to outliers and nonlinear scaling).
* **Numeric vs. Target (Hypothesis Testing):** Helps to check if feature means differ significantly across target classes. First, check whether the assumptions are satisfied, and then use t-test/ANOVA or choose the appropriate test accordingly.Refer : `docs\implementation-info\guides\hypothesis-testing-guide.md`
* **Categorical vs. Target:** First, check whether the assumptions are satisfied, and then use Chi-Square Test of Independence or choose the appropriate test accordingly. refer : `docs\implementation-info\guides\hypothesis-testing-guide.md`
* **Complex/Non-linear relationships:** Use Mutual Information (MI) scores, which capture complex patterns that correlation misses.

**Use this to decide:**

* keep features with signal (high Correlation, high MI score, or low p-value)
* remove leakage-prone columns
* bin continuous variables if target relation is non-linear
* create interaction features
* use encoding or scaling based on model choice

---

### E. Check relationships among features

Look at:

* correlation matrix for numeric columns
* redundant features
* multicollinearity
* high-cardinality categorical overlap
* feature clusters
* feature importance proxies

**Use this to decide:**

* remove one of highly correlated features for linear models
* keep correlated features for tree models if useful
* reduce dimensionality
* avoid duplicate information
* use regularization

---

### F. Check sample and split behavior

Investigate:

* train/test distribution shift
* subgroup differences
* time-based drift
* class imbalance
* group leakage

**Use this to decide:**

* stratified split for imbalance
* group split for repeated entities
* time split for temporal data
* reweighting or resampling
* robust validation strategy

---

## 3) EDA → preprocessing decision map

| EDA finding                               | Preprocessing action                                        |
| ----------------------------------------- | ----------------------------------------------------------- |
| Missing values are rare and random        | Impute with median/mode or simple constant                  |
| Missingness is informative                | Add missing indicator flag                                  |
| Feature is highly skewed                  | Log / Box-Cox / Yeo-Johnson transform                       |
| Extreme outliers are genuine but harmful  | Cap / winsorize / robust scale                              |
| Outliers are data errors                  | Correct or remove                                           |
| Categorical has many rare levels          | Group rare levels into “Other”                              |
| High-cardinality categorical              | Target encoding, hashing, frequency encoding, or embeddings |
| Numeric features on very different scales | Standardize or normalize                                    |
| Two features are highly correlated        | Drop one, combine, or regularize                            |
| Target classes are imbalanced             | Class weights, oversampling, undersampling                  |
| Feature has near-zero variance            | Remove it                                                   |
| Date column contains future info          | Remove or only use past-derived features                    |
| Strong seasonality/trend                  | Add time features, lag features, rolling stats              |
| Text is noisy/variable length             | Clean, normalize, tokenize, vectorize                       |
| Data differs by subgroup                  | Fairness checks, subgroup-wise preprocessing                |
| Train/test distributions differ           | Use robust transforms, domain-aware split, drift handling   |

---

## 4) Preprocessing approach: practical sequence

### Step 1: Clean types and structure

* convert dates to datetime
* convert numeric strings to numeric
* standardize category spelling
* remove duplicate rows if needed
* fix impossible values

### Step 2: Handle missing values

Choose based on EDA:

* numeric: median, mean, model-based, or sentinel
* categorical: mode or “Unknown”
* time series: forward fill, interpolation, seasonal fill
* add missing flags when missingness itself carries signal

### Step 3: Treat outliers

Choose based on context:

* cap extreme values
* transform skewed features
* use robust scalers
* remove only when clearly erroneous

### Step 4: Encode categorical variables

Choose based on cardinality and model:

* one-hot for low-cardinality
* ordinal for true ordered categories
* target encoding for high-cardinality supervised problems
* embeddings for deep models
* frequency hashing when categories are huge

### Step 5: Scale numeric features

Choose based on model:

* standardization for linear models, SVM, neural nets
* robust scaling if outliers are heavy
* usually not needed for tree models

### Step 6: Feature engineering

Driven by EDA:

* binning
* interactions
* ratios
* log transforms
* datetime parts
* lag/rolling features
* text length, counts, and embeddings

### Step 7: Reduce noise

* remove constant and duplicate features
* remove leakage
* remove redundant features if needed
* use feature selection or regularization

---

## 5) How EDA should guide preprocessing for different model types

### Linear / logistic regression

Focus on:

* scaling
* outliers
* multicollinearity
* linearity with target
* encoding quality

### Tree-based models

Focus on:

* missingness
* category handling
* leakage
* overfitting from high-cardinality categories
* extreme noise

### Neural networks

Focus on:

* scaling
* clean numeric ranges
* embeddings for categorical/text
* stable train/test distribution

### Time series models

Focus on:

* chronological split
* missing intervals
* seasonality
* lag features
* leakage from future data

---

## 6) A simple EDA checklist for every project

### Data integrity

* [ ] Correct column types
* [ ] Duplicate rows checked
* [ ] Target defined clearly
* [ ] Leakage columns identified

### Missingness

* [ ] Missing % per column
* [ ] Missingness pattern understood
* [ ] Imputation strategy chosen

### Distribution

* [ ] Numeric skew checked
* [ ] Outliers checked
* [ ] Categorical rarity checked

### Target relationship

* [ ] Important feature-target relationships explored visually
* [ ] Statistical significance or Mutual Information checked for categorical features (e.g., Chi-Square)
* [ ] Correlation checked for numeric features (Pearson/Spearman, after validating assumptions)
* [ ] Non-linear patterns noticed
* [ ] Class imbalance checked

### Feature relationships

* [ ] Correlation/redundancy checked
* [ ] High-cardinality categorical reviewed
* [ ] Feature interactions considered

### Split strategy

* [ ] Random / stratified / group / time split chosen
* [ ] Train-test distribution compared

### Preprocessing decision

* [ ] Scaling needed or not
* [ ] Encoding method chosen
* [ ] Transformations decided
* [ ] Feature selection plan made

---

## 7) A good rule of thumb

Do not preprocess blindly.
Let EDA answer these questions first:

* Is the data clean enough?
* Which features are useful?
* What is noisy, redundant, or leaking?
* Which transforms make the relationship easier for the model?
* What split strategy avoids false confidence?

---

## 8) One-line mental model

**EDA tells you what the data is doing. Preprocessing turns that understanding into model-friendly inputs.**

---

## 9) Experiment Tracking & Hyperparameter Tuning

Once EDA and preprocessing are complete, you will move to the modeling phase.

* **Experiment Tracking**: Use **MLflow** to track your preprocessing decisions, hyperparameters, and resulting metrics.
* **Hyperparameter Tuning**: Use **Optuna** to find the optimal hyperparameters for your models.
* **Setup**: In our architecture, we use AWS Managed MLflow and Optuna backed by a persistent SQLite database on EBS (which allows for easy export/import to a local MLflow server).

If you want, I can turn this into a **one-page interview-ready version** or a **project workflow template** you can reuse on any dataset.
