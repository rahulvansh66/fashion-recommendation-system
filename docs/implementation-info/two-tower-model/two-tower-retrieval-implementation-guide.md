# Two-Tower Retrieval — Implementation Guide

**Purpose:** Document the **implemented** Stage-1 retrieval training stack in this repository — code layout, data flow, pipelines, SageMaker jobs, MLflow/Optuna integration, and how to run experiments.

**Related (read in this order):**

| Document | Role |
|----------|------|
| [`two-tower-retrieval-training-guide.md`](./two-tower-retrieval-training-guide.md) | Model semantics — architecture, loss, hyperparameters, evaluation (reference / textbook) |
| [`mlflow-optuna-experiment-guide.md`](../guides/mlflow-optuna-experiment-guide.md) | AWS Managed MLflow + Optuna lifecycle and anti-patterns |
| [`features-eng.md`](../guides/features-eng.md) | Upstream transaction feature definitions |
| [`../../superpowers/specs/2026-06-11-two-tower-retrieval-experiments-design.md`](../../superpowers/specs/2026-06-11-two-tower-retrieval-experiments-design.md) | Approved design spec |

**Contract:** [`v1-requirements.md`](../../system-design/v1/v1-requirements.md) FR-BATCH-02 (temporal split), FR-BATCH-04 (future production pipeline).

---

## 1. What was built

Stage-1 **two-tower retrieval** training with:

- PyTorch dual-encoder model (`nn.Module` towers)
- **Log-q popularity correction** for in-batch negative debiasing
- **FR-BATCH-02** temporal train / val / test splits (not random 10/10)
- **AWS Managed MLflow** experiment tracking (metrics logged from pipeline scripts)
- **Optuna** HPO (3 trials; trial 0 enqueued with guide defaults)
- **SageMaker Training Job per trial** (`ml.m5.large` spot)
- Optional **SageMaker Processing** job to run the Optuna orchestrator
- Terraform modules for MLflow tracking server + Optuna RDS

```text
Feature Parquet (transactions/)
        │
        ▼
Temporal split + stage to S3  ──►  train.parquet / val.parquet / test.parquet
        │
        ├─► src/.../two_tower/train.py  (single job, MLflow logging)
        │
        └─► pipelines/hpo/run_two_tower_study.py
                 │  (Optuna on RDS; 1 SageMaker Training Job per trial)
                 ▼
            AWS Managed MLflow  +  query/candidate PyTorch checkpoints
```

---

## 2. Repository layout

| Path | Responsibility |
|------|----------------|
| `src/fashion_recommendation_system/models/retrieval/two_tower/` | Model code — **source of truth** (see module map below) |
| `src/.../two_tower/train.py` | CLI / SageMaker entrypoint; MLflow logging; artifact export |
| `pipelines/hpo/run_two_tower_study.py` | Optuna study; launches child Training Jobs |
| `pipelines/sagemaker/launch_training_job.py` | PyTorch Estimator wrapper |
| `pipelines/sagemaker/hpo_processing_job.py` | Launches Processing job running the HPO script |
| `configs/models/two_tower.yaml` | Frozen defaults + temporal split + SageMaker/Optuna settings |
| `configs/hpo/two_tower_search_space.yaml` | Optuna search space |
| `notebooks/two_tower_retrieval_experiments.ipynb` | Optional experiment driver (orchestration + MLflow UI queries) |
| `notebooks/utils/two_tower_training_helpers.py` | Split/stage/schema helpers (no `src/` import) |
| `notebooks/utils/ml_config.py` | Infra env vars for notebooks |
| `infra/` | Terraform: MLflow tracking server + Optuna RDS |
| `requirements-training.txt` | Python deps for training/HPO/SageMaker jobs |
| `tests/unit/test_two_tower_splits.py` | Temporal split unit tests |

**Config rule:** Hyperparameters live in `configs/**/*.yaml`. Infrastructure endpoints live in `src/fashion_recommendation_system/config.py` (via `.env.local`).

---

## 3. Notebook vs pipeline

Both exist; **only the pipeline is required** for experiments.

| Layer | Required? | Trains model? | Logs to MLflow? |
|-------|-----------|---------------|-----------------|
| `src/.../two_tower/train.py` | Yes (core) | Yes | Yes |
| `pipelines/hpo/run_two_tower_study.py` | For HPO | Orchestrates jobs | Yes (parent + nested trials) |
| `notebooks/two_tower_retrieval_experiments.ipynb` | No | No (calls pipelines) | Reads runs only |

**Typical workflow:**

1. **Prototype / validate** — run `train.py` locally (or one SageMaker job) until metrics look sane.
2. **HPO** — run `run_two_tower_study.py` or the Processing launcher.
3. **Notebook** — optional convenience for env checks, staging splits, launching jobs, browsing MLflow.

MLflow tracks runs whenever `train.py` or `run_two_tower_study.py` runs with `MLFLOW_TRACKING_URI` set — **not** because the notebook was used.

---

## 4. Input data

### 4.1 Source path

```
s3://{S3_BUCKET}/dataset/sample_2000_users/features/
```

Local dev mirror (Hive-partitioned by `snap_date`):

```
s3/dataset/sample_2000_users/features/
```

Produced by `notebooks/03_feature_engineering.ipynb` (or `pipelines/run_feature_pipeline.py` when wired).

### 4.2 Required columns (canonical names)

No renaming at load time. These are the columns the model consumes:

**Query tower**

| Column | Type | Notes |
|--------|------|-------|
| `customer_id` | string | Embedding lookup |
| `age` | float | Z-score normalized (train-only stats) |
| `txn_month_sin` | float | Seasonality, already ∈ [-1, 1] |
| `txn_month_cos` | float | Seasonality |

**Candidate tower**

| Column | Type | Notes |
|--------|------|-------|
| `article_id` | string | Embedding lookup |
| `item_category` | string | One-hot + Dense (garment-group equivalent) |
| `index_group_name` | string | One-hot + Dense |

Also required for splitting: `snap_date` (and `label` on anchor rows — only `label == 1` positives are kept after split).

Validation is enforced in `split.load_transactions()` and `notebooks/utils/two_tower_training_helpers.verify_schema()`.

### 4.3 Temporal split (FR-BATCH-02)

Implemented in `src/.../two_tower/split.py`. The **snap-date + 7-day forward label window** scheme:

| Role | Snap dates | Rows selected |
|------|------------|---------------|
| **Train** | `2020-03-31`, `2020-04-07` | Rows with matching `snap_date`; `label == 1` only; stacked |
| **Val** | `2020-04-14`, `2020-04-28` | Same rule for val snaps; stacked |
| **Test** | `2020-05-15` | Same rule for test snap |

Snap dates and label windows are configured in `configs/models/two_tower.yaml` under `temporal_split`.
### 4.4 Staged splits for SageMaker

Before Training Jobs, write split Parquet to:

```
s3://{bucket}/experiments/two_tower/{run_id}/train.parquet
s3://{bucket}/experiments/two_tower/{run_id}/val.parquet
s3://{bucket}/experiments/two_tower/{run_id}/test.parquet
```

Helpers: `stage_splits_s3()` / `stage_splits_local()` in `notebooks/utils/two_tower_training_helpers.py`.

Vocabularies and the popularity table are built from **train split only**.

---

## 5. Model implementation

Conceptual architecture matches [`two-tower-retrieval-training-guide.md`](./two-tower-retrieval-training-guide.md). Implementation uses updated column names (`txn_month_sin/cos`, `item_category`).

### 5.1 Module map

| File | Contents |
|------|----------|
| `model.py` | `QueryTower`, `ItemTower` (`nn.Module`) |
| `loss.py` | `build_article_prob_map()`, `popularity_corrected_loss()` |
| `preprocess.py` | Vocab maps, age z-score, batch encoding |
| `dataset.py` | PyTorch `Dataset` / `DataLoader` builders |
| `evaluate.py` | Recall@K over factorized candidate corpus |
| `export.py` | Checkpoint save/load (`.pt` + JSON) |
| `train.py` | SageMaker CLI entrypoint |
| `inference.py` | SageMaker handler stub (`model_fn`, `predict_fn`) |
| `split.py` | Pandas load + temporal split (no torch import) |

### 5.2 Query tower (`model.py`)

```
customer_id → Vocabulary index → nn.Embedding(emb_dim)
age         → z-score (train mean/std)
txn_month_sin, txn_month_cos → pass-through
→ concat → Linear(relu) → Linear → 16-d vector
```

### 5.3 Candidate tower (`model.py`)

```
article_id      → Vocabulary index → nn.Embedding(emb_dim)
item_category   → Vocabulary index → F.one_hot
index_group_name → Vocabulary index → F.one_hot
→ concat → Linear(relu) → Linear → 16-d vector
```

### 5.4 Loss and sampling

- **Positives:** every row in the train Parquet is one implicit purchase pair.
- **Negatives:** in-batch only — other items in the same batch (default batch size 2048 → 2047 negatives per row in the softmax).
- **No explicit negative rows** and no `label` column.

See training guide §3 for the conceptual explanation.

### 5.5 Popularity correction (training only)

From `tmp/recsys-v2/two-tower-cg/custom_cross_entropy_loss.py`, implemented in `loss.py`:

1. Precompute `P(article_id) = count / N_train` → dict keyed by embedding index.
2. In training step, compute logits `L = U @ Vᵀ` (batch × batch).
3. Subtract `log P(item_j)` from column `j` (item `j` in the batch).
4. Apply in-batch softmax CE with diagonal labels `0..batch_size-1`.

**Eval / test do not apply log-q correction** — standard dot-product scoring in `evaluate.py`.

### 5.6 Evaluation

- Corpus: deduplicated train articles through the candidate tower once per epoch.
- Headline metric: **`val_recall_at_100`** from `evaluate.recall_at_100()`.
- Optional: when `--test-uri` is passed in train mode, `train.py` also logs `test_recall_at_100` after training (informational; not used for Optuna).

### 5.7 Default hyperparameters

From `configs/models/two_tower.yaml` (same as training guide):

| Parameter | Default |
|-----------|---------|
| `embedding_dim` | 16 |
| `batch_size` | 2048 |
| `epochs` | 10 |
| `learning_rate` | 0.01 |
| `weight_decay` | 0.001 |

Optimizer: `torch.optim.AdamW`.

---

## 6. Training pipeline (`src/.../two_tower/train.py`)

### 6.1 CLI flags

| Flag | Purpose |
|------|---------|
| `--train-uri` | Train split Parquet (required) |
| `--val-uri` | Val split Parquet (required) |
| `--test-uri` | Test split (optional; logged after train if set) |
| `--mode` | `train` (default) or `eval` |
| `--embedding-dim`, `--batch-size`, `--epochs`, `--learning-rate`, `--weight-decay` | Hyperparameters |
| `--trial-number` | Optuna trial index (for run naming) |
| `--mlflow-run-id` | Parent run for nested MLflow logging |

### 6.2 MLflow logging

When `MLFLOW_TRACKING_URI` is set, each invocation:

- Starts an MLflow run (`trial_{n}` or `two_tower_train`)
- Tags: `git_sha`, `feature_snapshot`, `feature_cutoff`, `model=two_tower`, `data_env`
- Logs params, per-epoch loss, `val_recall_at_100`, top-K metrics
- Artifacts: `query_tower.pt`, `candidate_tower.pt`, `preprocess_state.json`, `metrics.json`

### 6.3 Local smoke run

```bash
# From repo root; install deps first:
# pip install -r requirements-training.txt

python src/fashion_recommendation_system/models/retrieval/two_tower/train.py \
  --train-uri s3/experiments/two_tower/{run_id}/train.parquet \
  --val-uri s3/experiments/two_tower/{run_id}/val.parquet \
  --embedding-dim 16 --batch-size 512 --epochs 2
```

Set `MLFLOW_TRACKING_URI` in `.env.local` to log to AWS Managed MLflow.

### 6.4 SageMaker Training Job

Launched via `pipelines/sagemaker/launch_training_job.py`:

- Framework: PyTorch 2.3, Python 3.11
- Instance: `ml.m5.large` (2 vCPU, 8 GiB) spot by default
- `source_dir`: `src/fashion_recommendation_system/models/retrieval/two_tower/`
- `dependencies`: `src/`, `configs/`

Environment passed to the job: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `GIT_SHA`, `FEATURE_SNAPSHOT`, `DATA_ENV`.

---

## 7. Hyperparameter optimization

### 7.1 Search space

`configs/hpo/two_tower_search_space.yaml`:

| Parameter | Range |
|-----------|-------|
| `learning_rate` | 1e-4 – 1e-2 (log) |
| `embedding_dim` | 16, 32, 64 |
| `batch_size` | 512, 1024, 2048 |
| `weight_decay` | 1e-4 – 1e-2 (log) |
| `epochs` | 5 – 15 |

### 7.2 Optuna study

Script: `pipelines/hpo/run_two_tower_study.py`

| Setting | Value |
|---------|-------|
| Study name | `two_tower_hpo_v1` (configurable) |
| Storage | `OPTUNA_STORAGE_URI` (RDS PostgreSQL) |
| Trials | **3** (`n_trials` in config) |
| Trial 0 | Enqueued with guide defaults |
| Objective | Maximize `val_recall_at_100` |
| Tune on | **Val only** — never test |

Each trial launches a SageMaker Training Job, waits for completion, reads `result.json` from the job output channel.

MLflow: parent study run + nested trial runs via `MLflowCallback`.

### 7.3 Launch HPO

**Direct (e.g. on Processing instance or dev box with SageMaker access):**

```bash
export MLFLOW_TRACKING_URI=...   # SageMaker tracking server ARN
export OPTUNA_STORAGE_URI=...    # postgresql+psycopg2://...
export SAGEMAKER_ROLE_ARN=...

python pipelines/hpo/run_two_tower_study.py \
  --train-uri s3://.../train.parquet \
  --val-uri s3://.../val.parquet
```

**Via SageMaker Processing:**

```bash
python pipelines/sagemaker/hpo_processing_job.py \
  --train-uri s3://.../train.parquet \
  --val-uri s3://.../val.parquet
```

### 7.4 After HPO

Export best params to `configs/models/two_tower.yaml` (notebook §7.2 helper or manually). Future weekly SageMaker Pipeline (FR-BATCH-04) trains with frozen YAML — **no Optuna in production**.

---

## 8. Experiment notebook

**File:** `notebooks/two_tower_retrieval_experiments.ipynb`

| Section | Purpose |
|---------|---------|
| 1 Setup | Load `.env.local`, configs, verify schema |
| 2 Infrastructure | MLflow server status, Optuna RDS connectivity |
| 3 Data prep | Temporal split, stage to S3 |
| 4 Baseline smoke | Optional single Training Job |
| 5 HPO | Launch Processing orchestrator; monitor MLflow |
| 6 Final eval | Best trial + optional test job |
| 7 Handoff | Export best params to YAML |

**Project rule:** notebooks do **not** import from `src/`. Helpers live in `notebooks/utils/`.

SageMaker launch cells are **commented out** by default — uncomment after `.env.local` is configured.

---

## 9. Infrastructure

### 9.1 Environment variables

Copy `.env.example` → `.env.local` (gitignored):

| Variable | Purpose |
|----------|---------|
| `MLFLOW_TRACKING_URI` | SageMaker tracking server ARN |
| `MLFLOW_EXPERIMENT` | e.g. `fashion-reco-dev` |
| `OPTUNA_STORAGE_URI` | PostgreSQL URI for Optuna |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role |
| `S3_BUCKET` | Artifact and experiment bucket |
| `AWS_REGION` | e.g. `us-east-1` |

AWS credentials: standard SDK chain (`AWS_ACCESS_KEY_ID`, etc.) in `.env.local`.

### 9.2 Terraform (`infra/`)

| Module | Resources |
|--------|-----------|
| `modules/mlflow_tracking_server/` | SageMaker MLflow tracking server (Small) |
| `modules/optuna_rds/` | `db.t4g.micro` PostgreSQL for Optuna |

```bash
cd infra
terraform init
terraform apply -var-file=environments/dev/terraform.tfvars
```

Outputs → `.env.local`: `mlflow_tracking_server_arn`, `optuna_storage_uri`.

**Session lifecycle:** start MLflow server → run experiments → **stop** server (do not delete). See [`mlflow-optuna-experiment-guide.md`](../guides/mlflow-optuna-experiment-guide.md).

Parameter Store is **not** required for v1 dev.

---

## 10. Dependencies

| File | Use |
|------|-----|
| `requirements-training.txt` | Training, HPO, SageMaker, MLflow, Optuna |
| `src/.../two_tower/requirements.txt` | Subset for SageMaker `source_dir` |
| `pyproject.toml` `[project.optional-dependencies.training]` | Editable install |

Install:

```bash
pip install -r requirements-training.txt
pip install -e .
```

Notebooks additionally need `requirements-notebooks.txt` (includes `python-dotenv`).

---

## 11. Tests

| Test | File | Needs TensorFlow? |
|------|------|-------------------|
| Temporal split boundaries | `tests/unit/test_two_tower_splits.py` | No |
| Popularity table probabilities | `tests/unit/test_two_tower_popularity.py` | No |

```bash
pytest tests/unit/test_two_tower_splits.py -q
```

---

## 12. Artifacts and handoff

| Output | Location |
|--------|----------|
| Query tower checkpoint | MLflow artifact `query_tower.pt` |
| Candidate tower checkpoint | MLflow artifact `candidate_tower.pt` |
| Metrics JSON | `metrics.json` in run artifacts |
| Best hyperparams | `configs/models/two_tower.yaml` (after HPO) |

Downstream (not yet implemented in this deliverable):

- SageMaker Model Registry promotion (FR-BATCH-04)
- FAISS index build from candidate embeddings
- Online retrieval via SageMaker + Lambda

See [`item-embeddings-and-inference-pipeline-guide.md`](../guides/item-embeddings-and-inference-pipeline-guide.md) for the inference path.

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Missing required columns: age` | FE not extended yet | Add `age`, `index_group_name` in feature engineering |
| MLflow run missing | `MLFLOW_TRACKING_URI` unset in job env | Pass env to Estimator / export before launch |
| OOM on `ml.m5.large` | Batch 2048 too large for instance | Reduce `batch_size` or use `ml.m5.xlarge` |
| Optuna study empty | Wrong `OPTUNA_STORAGE_URI` | Verify RDS connectivity and URI format |
| Import error for `torch` | Training venv not installed | `pip install -r requirements-training.txt` |

---

## 14. Source references

| Topic | Location |
|-------|----------|
| Design spec | `docs/superpowers/specs/2026-06-11-two-tower-retrieval-experiments-design.md` |
| Model semantics (reference) | `docs/implementation-info/two-tower-model/two-tower-retrieval-training-guide.md` |
| Implementation guide | `docs/implementation-info/two-tower-model/two-tower-retrieval-implementation-guide.md` |
| MLflow + Optuna | `docs/implementation-info/guides/mlflow-optuna-experiment-guide.md` |
| Reference two-tower (tmp) | `tmp/recsys/training/two_tower.py` |
| Popularity correction (tmp) | `tmp/recsys-v2/two-tower-cg/custom_cross_entropy_loss.py` |
| Temporal split contract | `docs/system-design/v1/v1-requirements.md` FR-BATCH-02 |
