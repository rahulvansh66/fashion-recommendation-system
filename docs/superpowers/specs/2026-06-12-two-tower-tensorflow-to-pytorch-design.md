---
title: Two-Tower TensorFlow to PyTorch Migration Design
date: 2026-06-12
author: rahul.vansh
project: Fashion Recommendation System
status: approved
supersedes_framework: TensorFlow 2.15 + tensorflow-recommenders
related_docs:
  - docs/system-design/v1/v1-hld.md
  - docs/system-design/v1/v1-requirements.md
  - docs/system-design/project-structure.md
  - docs/superpowers/specs/2026-06-11-two-tower-retrieval-experiments-design.md
  - docs/implementation-info/two-tower-model/two-tower-retrieval-implementation-guide.md
  - docs/implementation-info/two-tower-model/two-tower-retrieval-training-guide.md
---

# Two-Tower TensorFlow to PyTorch Migration Design

## Overview

Replace the existing **TensorFlow + TFRS** two-tower retrieval stack with **plain PyTorch** (`torch` only — no TorchRec), and **restructure modules** to match [`project-structure.md`](../../system-design/project-structure.md). Training semantics, data contracts, MLflow/Optuna orchestration, and the headline metric (**`val_recall_at_100`**) remain unchanged.

**Motivation:** v1 HLD specifies PyTorch for training and SageMaker inference. The current TF implementation was an experiment-phase shortcut; migration aligns code with architecture docs and downstream FAISS / endpoint paths.

**Migration strategy:** Full cutover — remove TensorFlow dependencies after PyTorch implementation is validated. No TF weight conversion; retrain from scratch on the same staged splits.

**Out of scope:** FAISS index build, full SageMaker inference endpoint deployment, Model Registry promotion, FR-BATCH-04 weekly pipeline, feature-engineering changes.

---

## Goals

| Goal | Detail |
|------|--------|
| Framework | Plain PyTorch (`nn.Module`, `DataLoader`, `torch.optim.AdamW`) |
| Layout | HLD module split under `src/.../two_tower/` (see §3) |
| Model semantics | Same towers, in-batch negatives, log-q popularity correction (train only) |
| Temporal integrity | FR-BATCH-02 splits unchanged (`split.py`) |
| Experiment tracking | MLflow params/metrics/artifacts via `mlflow.pytorch.log_model` |
| HPO | Optuna + SageMaker unchanged; launcher switches to `PyTorch` Estimator |
| Headline metric | `val_recall_at_100` (Recall@100 over full train-item corpus) |
| Dependency removal | Drop `tensorflow`, `tensorflow-recommenders` from all requirement files |

---

## Non-Goals

| Item | Reason |
|------|--------|
| TorchRec | Overkill for ~105k items, single-node training, custom log-q loss |
| TF metric parity gate | Retrain validates correctness; no requirement to match old MLflow run numbers |
| Dual TF/PyTorch runtime | Single framework after migration |
| Moving split logic to `data/` | `split.py` stays in `two_tower/` (framework-agnostic, already correct) |

---

## Current vs Target Layout

### Current (TensorFlow)

```text
src/fashion_recommendation_system/models/retrieval/two_tower/
  model.py          # TwoTowerModel (Keras) + custom train_step
  towers.py         # QueryTower, ItemTower
  dataset.py        # tf.data builders
  popularity.py     # StaticHashTable for log-q
  trainer.py        # build_model, train_model, model.fit()
  split.py          # pandas temporal split (unchanged)

pipelines/training/two_tower/
  train.py          # SageMaker entrypoint
  requirements.txt

pipelines/sagemaker/launch_training_job.py   # TensorFlow Estimator
```

### Target (PyTorch per HLD)

```text
src/fashion_recommendation_system/models/retrieval/two_tower/
  model.py          # QueryTower, ItemTower (nn.Module)
  loss.py           # In-batch softmax CE + log-q correction
  dataset.py        # PyTorch Dataset + DataLoader builders
  preprocess.py     # Vocab maps, age normalization stats, index encoding
  evaluate.py       # Recall@K over factorized candidate corpus
  export.py         # Checkpoint + vocab JSON + SageMaker model.tar.gz layout
  train.py          # SageMaker CLI entrypoint (MLflow, argparse)
  inference.py      # Stub model_fn / predict_fn for future endpoint
  split.py          # Unchanged (pandas only)

pipelines/sagemaker/launch_training_job.py   # PyTorch Estimator → src/.../two_tower/
```

**Removed:** `towers.py`, `trainer.py`, `popularity.py`, `pipelines/training/two_tower/` (entire directory).

---

## TensorFlow → PyTorch Mapping

| TensorFlow / TFRS | PyTorch replacement | Module |
|-------------------|---------------------|--------|
| `StringLookup` + `Embedding` | `dict[str,int]` vocab + `nn.Embedding`; index `0` = unknown | `preprocess.py`, `model.py` |
| `Normalization` (age) | Train mean/std in `AgeNormalizer`; z-score in forward | `preprocess.py`, `model.py` |
| `tf.one_hot` (categories) | `F.one_hot` from vocab indices | `preprocess.py`, `model.py` |
| `tf.data.Dataset` | `TwoTowerDataset` + `DataLoader` | `dataset.py` |
| `tfrs.tasks.Retrieval` loss | `logits = U @ V.T`; `F.cross_entropy(labels=range(B))` | `loss.py` |
| Log-q column correction | Subtract `log P(article_j)` per batch column | `loss.py` |
| `tfrs.metrics.FactorizedTopK` | Pre-embed corpus; batched `U @ V_all.T`; top-k hit rate | `evaluate.py` |
| `model.fit()` | Manual epoch loop in `train.py` | `train.py` |
| Keras `SavedModel` | `state_dict` + JSON metadata; `mlflow.pytorch.log_model` | `export.py` |
| `tf.keras.optimizers.AdamW` | `torch.optim.AdamW` | `train.py` |
| SageMaker `TensorFlow` Estimator | SageMaker `PyTorch` Estimator (2.x, py311) | `launch_training_job.py` |

---

## Module Responsibilities

### `model.py`

- `QueryTower(nn.Module)`: customer embedding + normalized age + month sin/cos → MLP → 16-d vector.
- `ItemTower(nn.Module)`: article embedding + one-hot categories → MLP → 16-d vector.
- Accepts **integer indices** (from `preprocess.py`), not raw strings, in `forward()`.

### `preprocess.py`

- `Vocabulary`: bidirectional string↔index maps; unknown → index `0`.
- `AgeNormalizer`: compute mean/std from train `age`; apply z-score.
- `PreprocessState`: serializable bundle (vocabs + age stats) saved with checkpoints.
- `encode_batch(raw_dict) -> dict[str, Tensor]`: batch encoding for training/eval.

### `loss.py`

- `build_article_prob_map(train_df, item_vocab) -> dict[int, float]`: P(article) from train counts keyed by embedding index.
- `popularity_corrected_loss(user_emb, item_emb, article_indices, prob_map)`: in-batch softmax CE with log-q debiasing.
- **Training only** — eval uses uncorrected dot-product scores.

### `dataset.py`

- `TwoTowerDataset`: one row = one purchase pair; returns raw feature dict per sample.
- `build_dataloaders(train_df, val_df, batch_size)`: train shuffled + cached in memory; val batched.
- `get_unique_items_df(train_df)`: deduplicated candidate rows for eval corpus.

Re-export `QUERY_FEATURES`, `CANDIDATE_FEATURES`, `ALL_FEATURES`, `build_vocabularies` from `split.py` for backward-compatible imports.

### `evaluate.py`

- `embed_candidate_corpus(item_tower, items_df, preprocess_state, device) -> Tensor`: shape `(N_items, emb_dim)`.
- `recall_at_k(query_tower, val_loader, corpus_embeddings, article_index_map, k=100) -> float`.
- `evaluate(model, val_loader, corpus, preprocess_state, device) -> dict[str, float]`: returns `val_recall_at_100` and optional top-1/5/10/50.

### `export.py`

- `save_checkpoint(query_tower, item_tower, preprocess_state, out_dir)`.
- `load_checkpoint(out_dir, ...) -> tuple`.
- `build_model_tar(out_dir)`: layout for SageMaker model artifact.

Artifact layout:

```text
model/
├── query_tower.pt
├── candidate_tower.pt
├── preprocess_state.json
└── metrics.json
```

### `train.py` (SageMaker entrypoint)

- Same CLI flags as current `pipelines/training/two_tower/train.py`.
- Modes: `train` (fit + val eval) and `eval` (load checkpoint + test).
- MLflow: params, per-epoch loss, `val_recall_at_100`, artifacts via `export.py` + `mlflow.log_artifacts`.
- Writes `result.json` to `SM_OUTPUT_DATA_DIR` for Optuna HPO.

### `inference.py` (stub)

- `model_fn(model_dir)`: load towers + preprocess state.
- `predict_fn(input_data, model)`: encode query features → query embedding.
- Full endpoint wiring deferred; stub satisfies HLD file layout and future endpoint work.

### `split.py`

- **No changes** to temporal split logic or `build_vocabularies()`.

---

## Data & Training Semantics (Unchanged)

### Input columns

| Column | Tower | PyTorch preprocessing |
|--------|-------|----------------------|
| `customer_id` | Query | Vocab index → `nn.Embedding` |
| `age` | Query | Z-score (train stats) |
| `txn_month_sin`, `txn_month_cos` | Query | Pass-through float |
| `article_id` | Candidate | Vocab index → `nn.Embedding` |
| `item_category` | Candidate | Vocab index → `F.one_hot` |
| `index_group_name` | Candidate | Vocab index → `F.one_hot` |

### Temporal split (FR-BATCH-02)

Configured in `configs/models/two_tower.yaml`; applied via `split.apply_temporal_split()`.

### Hyperparameters (defaults unchanged)

| Parameter | Default |
|-----------|---------|
| `embedding_dim` | 16 |
| `batch_size` | 2048 |
| `epochs` | 10 |
| `learning_rate` | 0.01 |
| `weight_decay` | 0.001 |

Optimizer: `torch.optim.AdamW`.

### Evaluation procedure

1. Deduplicate train articles → candidate corpus.
2. Embed corpus once per eval via `ItemTower`.
3. For each val batch: embed queries, compute `U @ V_all.T`, check if true article in top-100.
4. Log `val_recall_at_100` to MLflow.

---

## Pipeline & Infrastructure Changes

### SageMaker Training Job

| Setting | Before | After |
|---------|--------|-------|
| Estimator | `sagemaker.tensorflow.TensorFlow` | `sagemaker.pytorch.PyTorch` |
| `entry_point` | `train.py` | `train.py` |
| `source_dir` | `pipelines/training/two_tower/` | `src/fashion_recommendation_system/models/retrieval/two_tower/` |
| `dependencies` | `src/`, `configs/` | `configs/` + repo root on `PYTHONPATH` via env or parent package install |
| `framework_version` | `2.15` | `2.3` (or latest SM-supported 2.x) |
| `py_version` | `py311` | `py311` |

**Note:** SageMaker PyTorch jobs need `src` importable. Pass `SM_SOURCE_DIR` or install editable package in a `requirements.txt` colocated with `train.py`.

### HPO / Notebook

- `pipelines/hpo/run_two_tower_study.py`: no logic change; reads `result.json` as today.
- `pipelines/sagemaker/hpo_processing_job.py`: verify env vars unchanged.
- `notebooks/two_tower_retrieval_experiments.ipynb`: update docstrings/cell comments referencing TF paths only (no `src/` imports).

### Dependencies

**`requirements-training.txt`** and **`pyproject.toml`** `[project.optional-dependencies.training]`:

```text
# Remove
tensorflow>=2.15,<2.17
tensorflow-recommenders>=0.7.3

# Add
torch>=2.2,<2.5
```

Add `requirements.txt` beside `src/.../two_tower/train.py` for SageMaker container pip install.

---

## Testing Strategy

| Test | File | Notes |
|------|------|-------|
| Temporal splits | `tests/unit/test_two_tower_splits.py` | Unchanged |
| Article prob map | `tests/unit/test_two_tower_popularity.py` | Rewrite for `build_article_prob_map()` in `loss.py` |
| Loss smoke | `tests/unit/test_two_tower_loss.py` | New: diagonal label gets lowest loss vs shuffled |
| Eval smoke | `tests/unit/test_two_tower_evaluate.py` | New: perfect embeddings → recall@100 = 1.0 |

Integration smoke (manual):

```bash
python src/fashion_recommendation_system/models/retrieval/two_tower/train.py \
  --train-uri <staged_train.parquet> \
  --val-uri <staged_val.parquet> \
  --batch-size 512 --epochs 2
```

---

## Documentation Updates

| Document | Change |
|----------|--------|
| `docs/implementation-info/two-tower-model/two-tower-retrieval-implementation-guide.md` | PyTorch modules, new paths, PyTorch Estimator |
| `docs/implementation-info/two-tower-model/two-tower-retrieval-training-guide.md` | Framework references TF → PyTorch |
| `docs/implementation-info/two-tower-model/README.md` | Link to this migration spec |
| `docs/superpowers/specs/2026-06-11-two-tower-retrieval-experiments-design.md` | Add note: framework migrated to PyTorch per this spec |

---

## Rollout & Validation

1. Implement PyTorch modules and unit tests locally.
2. Run local smoke train (2 epochs, batch 512) — loss decreases, metrics logged.
3. Run `pytest tests/unit/test_two_tower*.py`.
4. Remove TF files and dependencies.
5. Optional: one SageMaker PyTorch Training Job to confirm container wiring.

**Rollback:** Revert git commit; TF code removed only after step 3 passes.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| SageMaker `source_dir` cannot import `fashion_recommendation_system` | Colocated `requirements.txt` with `-e .` or set `PYTHONPATH` in Estimator env |
| OOM on `ml.m5.large` at batch 2048 | Same as today — reduce batch or upsize instance in config |
| Eval slower without TFRS FactorizedTopK | Corpus embed once per epoch; numpy/torch matmul is sufficient at ~105k items |
| MLflow artifact format change | Document new `.pt` + JSON layout; old SavedModels not auto-migrated |

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | Plain PyTorch | HLD alignment; full control over log-q loss; no TorchRec ops burden |
| Layout | Full HLD restructure | User requirement; separates loss/eval/export from monolithic trainer |
| Weight migration | None | Different graph execution; retrain is simpler and correct |
| Entrypoint location | `src/.../two_tower/train.py` | Matches `project-structure.md` |
| `inference.py` | Stub only | HLD file present; endpoint work is separate deliverable |
