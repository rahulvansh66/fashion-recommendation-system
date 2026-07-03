# Multi-Stage H&M Fashion Recommendation System

<!-- Building a two-stage recommendation system that predicts purchase likelihood over the next 7 days, deployed on AWS. -->

<!-- Personalized top-15 fashion recommendations: -->
Building a personalized two-stage recommendation system that predicts which products a user may buy in the next 7 days using Two-Tower candidate retrieval and XGBoost ranking, deployed on AWS.

The system learns users' shopping behavior from purchase history and engineered time-series features to generate personalized recommendations.

<!-- Recommendations are generated based on user historical purchase sequences and time-series
patterns. -->

<!-- Built on AWS — FastAPI on ECS Fargate, SageMaker inference, FAISS vector search, Redis caching, and S3 data lake — with infrastructure managed by Terraform.  -->

<!-- (ECS Fargate, SageMaker, Redis, S3) -->

 <!-- Cache → Retrieve → Filter → Rank → Order pipeline on ECS Fargate, SageMaker, FAISS (Lambda), ElastiCache Redis, and S3, deployed with Terraform.  -->

<!-- See full plan [`docs/system-design/v1/v1-hld.md`](docs/system-design/v1/v1-hld.md) for HLD architecture. -->

## Prerequisites

- **Python 3.11+** (3.11 recommended)
- **Java 8+** for local PySpark notebooks (Java 17+ if you upgrade to PySpark 3.5+)

See **[Java + PySpark local setup guide](docs/implementation-info/guides/java-pyspark-local-setup.md)** for Mac/Windows install, `JAVA_HOME`, and troubleshooting.

Check Java:

```bash
java -version
```

## Python environments

This repo uses **two separate virtual environments** — common when the notebook stack (PySpark, Jupyter) is heavy and should stay out of the application/script environment.

| Environment | Path | Requirements | Used for |
|-------------|------|--------------|----------|
| **Scripts / app** | `.venv` | `requirements.txt` | `src/`, tests, lint, future FastAPI / pipelines |
| **Notebooks** | `.venv-notebooks` | `requirements-notebooks.txt` | `notebooks/` (PySpark, EDA) |

Future dependency splits (`requirements-serving.txt`, `requirements-training.txt`) are documented in [`docs/system-design/project-structure.md`](docs/system-design/project-structure.md).

### One-time setup

From the repo root:

```bash
# 1. Script / app environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

# 2. Notebook environment + Jupyter kernel
python -m venv .venv-notebooks
source .venv-notebooks/bin/activate
pip install --upgrade pip
pip install -r requirements-notebooks.txt
# Project-local kernel (best for Cursor/VS Code) + user-level fallback
python -m ipykernel install --prefix="$(pwd)/.venv-notebooks" --name fashion-reco-notebooks --display-name "Fashion Reco (notebooks)"
python -m ipykernel install --user --name fashion-reco-notebooks --display-name "Fashion Reco (notebooks)"
deactivate
```

### Daily use

**Scripts / tests** (activate script env):

```bash
source .venv/bin/activate
# run tests, lint, future CLI / app code
deactivate
```

**Notebooks** (Cursor / VS Code):

1. Open a notebook under `notebooks/`
2. Click **Select Kernel** (top-right)
3. Pick one of:
   - **Fashion Reco (notebooks)** under *Jupyter Kernel…* (scroll past Recommended)
   - **Python 3.11.x (`.venv-notebooks`)** under *Python Environments*
   - **Enter interpreter path…** → `.venv-notebooks/bin/python`
4. If the kernel list looks stale: **Developer: Reload Window** (`Cmd+Shift+P`)

Workspace default interpreter is set in `.vscode/settings.json` to `.venv-notebooks`.

Or from terminal:

```bash
source .venv-notebooks/bin/activate
jupyter notebook notebooks/
```

### Stratified sampling notebook

Builds `dataset/sample/` from `dataset/full/` (~1K stratified users):

```bash
source .venv-notebooks/bin/activate
jupyter notebook notebooks/stratified_user_sampling.ipynb
```

Run all cells. Output: `dataset/sample/{customers,articles,transactions_train}.csv` and `sampling_manifest.json`.

Spec: [`docs/superpowers/specs/2026-06-04-stratified-user-sampling-design.md`](docs/superpowers/specs/2026-06-04-stratified-user-sampling-design.md)

## Dataset layout

| Path | Description |
|------|-------------|
| `dataset/full/` | Full H&M CSVs (not in git) |
| `dataset/sample/` | Stratified dev sample (not in git) |

Download the full H&M dataset locally before running the sampling notebook.

## Documentation

| Topic | Document |
|-------|----------|
| Requirements | [`docs/system-design/v1/v1-requirements.md`](docs/system-design/v1/v1-requirements.md) |
| Architecture | [`docs/system-design/v1/v1-hld.md`](docs/system-design/v1/v1-hld.md) |
| Schema | [`docs/system-design/schema-info.md`](docs/system-design/schema-info.md) |
| Repo layout | [`docs/system-design/project-structure.md`](docs/system-design/project-structure.md) |
