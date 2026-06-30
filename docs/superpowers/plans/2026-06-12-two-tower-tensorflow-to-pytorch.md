# Two-Tower TensorFlow → PyTorch Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TensorFlow/TFRS two-tower retrieval with plain PyTorch, restructured per HLD (`model.py`, `loss.py`, `dataset.py`, `preprocess.py`, `evaluate.py`, `export.py`, `train.py`, `inference.py`).

**Architecture:** `nn.Module` dual towers; in-batch softmax CE with log-q correction in `loss.py`; Recall@100 via factorized corpus embeddings in `evaluate.py`; SageMaker entrypoint moves to `src/.../two_tower/train.py`; PyTorch Estimator in launcher.

**Tech Stack:** PyTorch 2.x, pandas, pyarrow, MLflow, Optuna, SageMaker PyTorch Estimator, pytest.

**Design spec:** [`docs/superpowers/specs/2026-06-12-two-tower-tensorflow-to-pytorch-design.md`](../specs/2026-06-12-two-tower-tensorflow-to-pytorch-design.md)

---

## File Map

| Action | Path |
|--------|------|
| Create | `src/fashion_recommendation_system/models/retrieval/two_tower/preprocess.py` |
| Create | `src/fashion_recommendation_system/models/retrieval/two_tower/loss.py` |
| Create | `src/fashion_recommendation_system/models/retrieval/two_tower/evaluate.py` |
| Create | `src/fashion_recommendation_system/models/retrieval/two_tower/export.py` |
| Create | `src/fashion_recommendation_system/models/retrieval/two_tower/inference.py` |
| Rewrite | `src/fashion_recommendation_system/models/retrieval/two_tower/model.py` |
| Rewrite | `src/fashion_recommendation_system/models/retrieval/two_tower/dataset.py` |
| Create | `src/fashion_recommendation_system/models/retrieval/two_tower/train.py` |
| Create | `src/fashion_recommendation_system/models/retrieval/two_tower/requirements.txt` |
| Keep | `src/fashion_recommendation_system/models/retrieval/two_tower/split.py` |
| Delete | `src/fashion_recommendation_system/models/retrieval/two_tower/towers.py` |
| Delete | `src/fashion_recommendation_system/models/retrieval/two_tower/trainer.py` |
| Delete | `src/fashion_recommendation_system/models/retrieval/two_tower/popularity.py` |
| Delete | `pipelines/training/two_tower/` (entire directory) |
| Modify | `pipelines/sagemaker/launch_training_job.py` |
| Modify | `requirements-training.txt`, `pyproject.toml` |
| Modify | `tests/unit/test_two_tower_popularity.py` |
| Create | `tests/unit/test_two_tower_loss.py` |
| Create | `tests/unit/test_two_tower_evaluate.py` |
| Modify | `docs/implementation-info/two-tower-model/*.md` |

---

### Task 1: Swap training dependencies

**Files:**
- Modify: `requirements-training.txt`
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace TensorFlow deps with PyTorch**

In `requirements-training.txt`, remove:
```text
tensorflow>=2.15,<2.17
tensorflow-recommenders>=0.7.3
```
Add:
```text
torch>=2.2,<2.5
```

In `pyproject.toml` `[project.optional-dependencies]` training section, make the same swap.

- [ ] **Step 2: Install and verify**

Run: `pip install -r requirements-training.txt && python -c "import torch; print(torch.__version__)"`
Expected: prints a 2.x version, no tensorflow import.

- [ ] **Step 3: Commit**

```bash
git add requirements-training.txt pyproject.toml
git commit -m "chore: replace tensorflow with torch in training deps"
```

---

### Task 2: Implement preprocess.py

**Files:**
- Create: `src/fashion_recommendation_system/models/retrieval/two_tower/preprocess.py`
- Test: `tests/unit/test_two_tower_preprocess.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_two_tower_preprocess.py`:

```python
"""Unit tests for two-tower preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from fashion_recommendation_system.models.retrieval.two_tower.preprocess import (
    AgeNormalizer,
    PreprocessState,
    Vocabulary,
    build_preprocess_state,
)


def test_vocabulary_unknown_maps_to_zero() -> None:
    vocab = Vocabulary(["a1", "a2"])
    assert vocab.encode("a1") == 1
    assert vocab.encode("missing") == 0


def test_age_normalizer_zscore() -> None:
    norm = AgeNormalizer.from_series(pd.Series([10.0, 20.0, 30.0]))
    assert norm.normalize(20.0) == pytest.approx(0.0)


def test_build_preprocess_state_from_train() -> None:
    train_df = pd.DataFrame(
        {
            "customer_id": ["c1"],
            "age": [25.0],
            "article_id": ["a1"],
            "item_category": ["cat1"],
            "index_group_name": ["g1"],
        }
    )
    vocabs = {
        "user_ids": ["c1"],
        "item_ids": ["a1"],
        "item_categories": ["cat1"],
        "index_groups": ["g1"],
    }
    state = build_preprocess_state(train_df, vocabs)
    assert isinstance(state, PreprocessState)
    assert state.user_vocab.encode("c1") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_two_tower_preprocess.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement preprocess.py**

Key types:
- `Vocabulary`: reserve index `0` for unknown; `encode(str)->int`, `size` property.
- `AgeNormalizer`: `from_series()`, `normalize(float)`, `to_dict()` / `from_dict()`.
- `PreprocessState`: holds all vocabs + age normalizer; JSON serializable.
- `build_preprocess_state(train_df, vocabs) -> PreprocessState`.
- `encode_row(state, row_dict) -> dict` returning int/float tensors-ready values.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_two_tower_preprocess.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fashion_recommendation_system/models/retrieval/two_tower/preprocess.py tests/unit/test_two_tower_preprocess.py
git commit -m "feat(two-tower): add PyTorch preprocessing module"
```

---

### Task 3: Implement loss.py

**Files:**
- Create: `src/fashion_recommendation_system/models/retrieval/two_tower/loss.py`
- Modify: `tests/unit/test_two_tower_popularity.py` → rename logic to test `build_article_prob_map`
- Create: `tests/unit/test_two_tower_loss.py`

- [ ] **Step 1: Rewrite popularity test for PyTorch-free prob map**

Update `tests/unit/test_two_tower_popularity.py`:

```python
"""Unit tests for log-q article probability map."""

from __future__ import annotations

import pandas as pd
import pytest

from fashion_recommendation_system.models.retrieval.two_tower.loss import build_article_prob_map
from fashion_recommendation_system.models.retrieval.two_tower.preprocess import Vocabulary


def test_article_prob_map_sums_to_one() -> None:
    train_df = pd.DataFrame({"article_id": ["a1", "a1", "a2"]})
    vocab = Vocabulary(["a1", "a2"])
    prob_map = build_article_prob_map(train_df, vocab)
    assert prob_map[vocab.encode("a1")] == pytest.approx(2 / 3)
    assert prob_map[vocab.encode("a2")] == pytest.approx(1 / 3)
    assert sum(prob_map.values()) == pytest.approx(1.0)
```

(Fix import path typo: use `.` not `/` in import.)

- [ ] **Step 2: Write loss smoke test**

Create `tests/unit/test_two_tower_loss.py`:

```python
"""Unit tests for popularity-corrected in-batch loss."""

from __future__ import annotations

import torch

from fashion_recommendation_system.models.retrieval.two_tower.loss import popularity_corrected_loss


def test_loss_is_low_when_positives_aligned() -> None:
    batch_size, dim = 4, 8
    emb = torch.eye(batch_size, dim)
    article_indices = torch.arange(1, batch_size + 1)
    prob_map = {i: 1.0 / batch_size for i in range(1, batch_size + 1)}
    loss = popularity_corrected_loss(emb, emb, article_indices, prob_map)
    assert loss.item() < 0.5
```

- [ ] **Step 3: Implement loss.py**

```python
def build_article_prob_map(train_df, item_vocab: Vocabulary) -> dict[int, float]:
    counts = train_df.groupby("article_id").size()
    total = len(train_df)
    return {item_vocab.encode(str(aid)): count / total for aid, count in counts.items()}

def popularity_corrected_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    article_indices: torch.Tensor,
    prob_map: dict[int, float],
    default_prob: float = 1e-8,
) -> torch.Tensor:
    logits = user_emb @ item_emb.T
    col_probs = torch.tensor(
        [prob_map.get(int(i), default_prob) for i in article_indices],
        device=logits.device,
        dtype=logits.dtype,
    )
    corrected = logits - torch.log(col_probs).unsqueeze(0)
    labels = torch.arange(logits.size(0), device=logits.device)
    return torch.nn.functional.cross_entropy(corrected, labels)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_two_tower_popularity.py tests/unit/test_two_tower_loss.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fashion_recommendation_system/models/retrieval/two_tower/loss.py tests/unit/test_two_tower_*.py
git commit -m "feat(two-tower): add log-q loss and article prob map"
```

---

### Task 4: Implement model.py (PyTorch towers)

**Files:**
- Rewrite: `src/fashion_recommendation_system/models/retrieval/two_tower/model.py`
- Delete (later in Task 9): `towers.py`

- [ ] **Step 1: Replace model.py with nn.Module towers**

`QueryTower` and `ItemTower` accept encoded tensors/indices from `PreprocessState`:
- Query: `customer_idx (B,)`, `age (B,)`, `txn_month_sin (B,)`, `txn_month_cos (B,)`
- Item: `article_idx (B,)`, `category_idx (B,)`, `index_group_idx (B,)`

Architecture mirrors TF version:
- Embedding → concat → Linear(emb_dim, relu) → Linear(emb_dim) → output

- [ ] **Step 2: Smoke test in Python REPL**

Run:
```bash
python -c "
import torch
from fashion_recommendation_system.models.retrieval.two_tower.model import QueryTower, ItemTower
q = QueryTower(num_users=10, emb_dim=16)
i = ItemTower(num_items=10, num_categories=5, num_index_groups=3, emb_dim=16)
b = 4
print(q({'customer_idx': torch.zeros(b,dtype=torch.long), 'age': torch.randn(b), 'txn_month_sin': torch.randn(b), 'txn_month_cos': torch.randn(b)}).shape)
"
```
Expected: `torch.Size([4, 16])`

- [ ] **Step 3: Commit**

```bash
git add src/fashion_recommendation_system/models/retrieval/two_tower/model.py
git commit -m "feat(two-tower): rewrite towers as PyTorch nn.Module"
```

---

### Task 5: Implement dataset.py

**Files:**
- Rewrite: `src/fashion_recommendation_system/models/retrieval/two_tower/dataset.py`

- [ ] **Step 1: Implement TwoTowerDataset and DataLoaders**

- `TwoTowerDataset(df)`: stores raw string/float columns; `__getitem__` returns dict per row.
- `collate_raw_batch(list)`: stack into batch dict of lists (strings stay strings until preprocess in train loop).
- `build_dataloaders(train_df, val_df, batch_size, num_workers=0)`.
- `get_unique_items_df(train_df)`: dedupe on `article_id`.
- Re-export constants from `split.py`.

- [ ] **Step 2: Smoke test**

Run quick script loading a tiny DataFrame; assert batch keys match `ALL_FEATURES`.

- [ ] **Step 3: Commit**

```bash
git add src/fashion_recommendation_system/models/retrieval/two_tower/dataset.py
git commit -m "feat(two-tower): add PyTorch Dataset and DataLoader builders"
```

---

### Task 6: Implement evaluate.py

**Files:**
- Create: `src/fashion_recommendation_system/models/retrieval/two_tower/evaluate.py`
- Create: `tests/unit/test_two_tower_evaluate.py`

- [ ] **Step 1: Write failing eval test**

```python
"""Unit tests for recall@K evaluation."""

from __future__ import annotations

import torch

from fashion_recommendation_system.models.retrieval.two_tower.evaluate import recall_at_k_from_scores


def test_recall_at_k_perfect_diagonal() -> None:
    scores = torch.eye(5)
    labels = torch.arange(5)
    assert recall_at_k_from_scores(scores, labels, k=1) == 1.0
```

- [ ] **Step 2: Implement evaluate.py**

Functions:
- `recall_at_k_from_scores(scores, true_indices, k) -> float`
- `embed_candidate_corpus(item_tower, items_df, state, device) -> Tensor`
- `evaluate_epoch(query_tower, item_tower, val_loader, corpus_emb, article_indices, state, device, k=100) -> dict`

Map each val row's `article_id` to corpus row index for label lookup.

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_two_tower_evaluate.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fashion_recommendation_system/models/retrieval/two_tower/evaluate.py tests/unit/test_two_tower_evaluate.py
git commit -m "feat(two-tower): add Recall@K evaluation module"
```

---

### Task 7: Implement export.py

**Files:**
- Create: `src/fashion_recommendation_system/models/retrieval/two_tower/export.py`

- [ ] **Step 1: Implement save/load helpers**

```python
def save_artifacts(
    query_tower: QueryTower,
    item_tower: ItemTower,
    state: PreprocessState,
    out_dir: Path,
    metrics: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(query_tower.state_dict(), out_dir / "query_tower.pt")
    torch.save(item_tower.state_dict(), out_dir / "candidate_tower.pt")
    (out_dir / "preprocess_state.json").write_text(state.to_json())
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
```

Implement symmetric `load_artifacts()` rebuilding modules from saved vocab sizes.

- [ ] **Step 2: Commit**

```bash
git add src/fashion_recommendation_system/models/retrieval/two_tower/export.py
git commit -m "feat(two-tower): add checkpoint export and load helpers"
```

---

### Task 8: Implement train.py (SageMaker entrypoint)

**Files:**
- Create: `src/fashion_recommendation_system/models/retrieval/two_tower/train.py`
- Create: `src/fashion_recommendation_system/models/retrieval/two_tower/requirements.txt`

- [ ] **Step 1: Port CLI from pipelines/training/two_tower/train.py**

Same argparse flags. Training loop:

```python
for epoch in range(epochs):
    query_tower.train(); item_tower.train()
    for batch in train_loader:
        encoded = encode_batch(state, batch)
        user_emb = query_tower(encoded)
        item_emb = item_tower(encoded)
        loss = popularity_corrected_loss(...)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val_metrics = evaluate_epoch(...)
    mlflow.log_metric("val_recall_at_100", val_metrics["recall_at_100"], step=epoch)
```

Use `torch.optim.AdamW(lr=..., weight_decay=...)`.

- [ ] **Step 2: Add SageMaker requirements.txt**

```text
torch>=2.2,<2.5
pandas>=2.0
pyarrow>=14.0
PyYAML>=6.0
mlflow>=2.10
```

For full package imports, add `-e ../../../..` or copy pattern from existing pipeline (install `fashion_recommendation_system` from repo root in container via Estimator `dependencies`).

- [ ] **Step 3: Local smoke run**

Run with staged parquet splits, `--batch-size 512 --epochs 2`.
Expected: loss logged, `val_recall_at_100` in output JSON.

- [ ] **Step 4: Commit**

```bash
git add src/fashion_recommendation_system/models/retrieval/two_tower/train.py \
        src/fashion_recommendation_system/models/retrieval/two_tower/requirements.txt
git commit -m "feat(two-tower): add PyTorch SageMaker training entrypoint"
```

---

### Task 9: Add inference.py stub

**Files:**
- Create: `src/fashion_recommendation_system/models/retrieval/two_tower/inference.py`

- [ ] **Step 1: Implement stub handlers**

```python
def model_fn(model_dir: str):
    """Load towers and preprocess state for SageMaker inference."""
    return load_artifacts(Path(model_dir))

def predict_fn(input_data, model):
    """Return query embedding for a single encoded request."""
    query_tower, _, state = model
    # encode and forward — minimal stub returning numpy array
    ...
```

- [ ] **Step 2: Commit**

```bash
git add src/fashion_recommendation_system/models/retrieval/two_tower/inference.py
git commit -m "feat(two-tower): add inference handler stub"
```

---

### Task 10: Update SageMaker launcher

**Files:**
- Modify: `pipelines/sagemaker/launch_training_job.py`

- [ ] **Step 1: Switch to PyTorch Estimator**

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir=str(root / "src" / "fashion_recommendation_system" / "models" / "retrieval" / "two_tower"),
    dependencies=[str(root / "src"), str(root / "configs")],
    framework_version="2.3",
    py_version="py311",
    ...
)
```

Update module docstring: TensorFlow → PyTorch.

- [ ] **Step 2: Commit**

```bash
git add pipelines/sagemaker/launch_training_job.py
git commit -m "feat(sagemaker): switch two-tower training to PyTorch estimator"
```

---

### Task 11: Remove TensorFlow code

**Files:**
- Delete: `towers.py`, `trainer.py`, `popularity.py`
- Delete: `pipelines/training/two_tower/` directory

- [ ] **Step 1: Delete obsolete files**

```bash
rm src/fashion_recommendation_system/models/retrieval/two_tower/towers.py
rm src/fashion_recommendation_system/models/retrieval/two_tower/trainer.py
rm src/fashion_recommendation_system/models/retrieval/two_tower/popularity.py
rm -rf pipelines/training/two_tower/
```

- [ ] **Step 2: Grep for stale imports**

Run: `rg "tensorflow|tensorflow_recommenders|build_tf_datasets|trainer import|from.*towers" --glob '!docs/**'`
Expected: no hits outside docs/historical specs.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore(two-tower): remove TensorFlow implementation files"
```

---

### Task 12: Update documentation

**Files:**
- Modify: `docs/implementation-info/two-tower-model/two-tower-retrieval-implementation-guide.md`
- Modify: `docs/implementation-info/two-tower-model/two-tower-retrieval-training-guide.md`
- Modify: `docs/implementation-info/two-tower-model/README.md`

- [ ] **Step 1: Update implementation guide**

Replace TF module map, SavedModel artifacts, TensorFlow Estimator sections with PyTorch equivalents from design spec §3–§6.

- [ ] **Step 2: Update training guide framework references**

Change "TensorFlow + TFRS" → "PyTorch"; update component table (StringLookup → Vocabulary + nn.Embedding, etc.).

- [ ] **Step 3: Add migration spec link to README**

- [ ] **Step 4: Commit**

```bash
git add docs/implementation-info/two-tower-model/
git commit -m "docs: update two-tower guides for PyTorch migration"
```

---

### Task 13: Final verification

- [ ] **Step 1: Run all two-tower unit tests**

Run: `pytest tests/unit/test_two_tower_splits.py tests/unit/test_two_tower_preprocess.py tests/unit/test_two_tower_popularity.py tests/unit/test_two_tower_loss.py tests/unit/test_two_tower_evaluate.py -v`
Expected: all PASS

- [ ] **Step 2: Confirm no tensorflow in requirements**

Run: `rg "tensorflow" requirements-training.txt pyproject.toml`
Expected: no matches

- [ ] **Step 3: Local training smoke (if parquet splits available)**

Run train.py with 2 epochs; confirm `result.json` contains `val_recall_at_100`.

---

## Plan Self-Review

| Spec requirement | Task |
|------------------|------|
| Plain PyTorch, no TorchRec | Task 1, all modules |
| HLD file layout | Tasks 2–9 |
| Log-q correction train-only | Task 3, 8 |
| Recall@100 eval | Task 6 |
| SageMaker PyTorch Estimator | Task 10 |
| Remove TF deps/files | Tasks 1, 11 |
| MLflow artifacts | Tasks 7, 8 |
| split.py unchanged | Explicitly kept |
| inference.py stub | Task 9 |
| Docs updated | Task 12 |

No TBD placeholders remain.
