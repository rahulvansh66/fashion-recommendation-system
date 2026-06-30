# Two-Tower Retrieval Model — Documentation

Stage-1 retrieval (dual-encoder) training and experiment runbook.

| Document | Description |
|----------|-------------|
| [`two-tower-retrieval-training-guide.md`](./two-tower-retrieval-training-guide.md) | Model semantics — features, architecture, loss, hyperparameters, evaluation |
| [`two-tower-retrieval-implementation-guide.md`](./two-tower-retrieval-implementation-guide.md) | Repo implementation — code layout, pipelines, SageMaker, MLflow/Optuna, runbook |
| [`../../superpowers/specs/2026-06-12-two-tower-tensorflow-to-pytorch-design.md`](../../superpowers/specs/2026-06-12-two-tower-tensorflow-to-pytorch-design.md) | Approved migration spec — TensorFlow → PyTorch + HLD restructure |
| [`../../superpowers/plans/2026-06-12-two-tower-tensorflow-to-pytorch.md`](../../superpowers/plans/2026-06-12-two-tower-tensorflow-to-pytorch.md) | Step-by-step implementation plan |

**Related (other folders):**

- [`../guides/mlflow-optuna-experiment-guide.md`](../guides/mlflow-optuna-experiment-guide.md) — experiment tracking and HPO lifecycle
- [`../guides/features-eng.md`](../guides/features-eng.md) — upstream feature engineering
- [`../../superpowers/specs/2026-06-11-two-tower-retrieval-experiments-design.md`](../../superpowers/specs/2026-06-11-two-tower-retrieval-experiments-design.md) — approved design spec
