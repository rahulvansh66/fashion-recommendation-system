---
title: Stratified User Sampling Notebook Design
date: 2026-06-04
author: rahul.vansh
project: Fashion Recommendation System
status: approved
related_docs:
  - docs/system-design/v1/v1-requirements.md
  - docs/system-design/v1/v1-hld.md
  - docs/system-design/schema-info.md
  - docs/system-design/v1/v1-infrastructure-layer.md
---

# Stratified User Sampling Notebook Design

## Overview

Create a PySpark notebook at `notebooks/stratified_user_sampling.ipynb` that builds a reproducible, proportion-preserving dev sample from the full H&M dataset. The notebook reads from `dataset/full/`, writes filtered CSVs to `dataset/sample/`, and is structured so the same logic can migrate to an AWS Glue PySpark job with minimal changes (config-driven paths only).

**Out of scope for this notebook:** `dataset/full/images/` (skipped entirely).

**Future path:** `dataset/sample/` → upload/ingest → `s3://fashion-reco-{env}/raw/`.

---

## Goals

| Goal | Detail |
|------|--------|
| Representative sample | ~1,000 users with purchase-tier and recency proportions matching the full dataset |
| Reproducibility | Fixed seed + deterministic ordering; re-runs produce identical `sampled_customer_ids` |
| Glue-ready | Pure PySpark, no Pandas; paths and params via config dict + env vars |
| Clean output | Filter transactions, customers, and articles to the sampled subset only |

---

## Segment Definitions

All labels are computed using transaction history **as of cutoff date** only. Post-cutoff transactions are **never** used for labeling (but are retained in output — see Output Filtering).

| Parameter | Value |
|-----------|-------|
| Cutoff date | `2020-03-31` |
| Recency window | 30 days |
| Recency window range | `(cutoff - 30 days, cutoff]` → `(2020-03-02, 2020-03-31]` |

### Purchase tier (primary axis)

Computed from `purchase_count_pre_cutoff` = count of transactions with `t_dat ≤ cutoff`.

| Tier | Rule |
|------|------|
| **new** | In `customers`, zero transactions with `t_dat ≤ cutoff` |
| **cold** | 1–2 purchases before cutoff |
| **light** | 3–5 purchases before cutoff |
| **medium** | 6–19 purchases before cutoff |
| **heavy** | 20+ purchases before cutoff |

**Boundary:** Medium = 6–19, Heavy = 20+ (no overlap at 20).

**New user note:** A customer with purchases only after cutoff is still **new** (zero pre-cutoff history). After sampling, all their transactions (including post-cutoff) are kept.

### Recency (secondary axis — non-new tiers only)

Computed from pre-cutoff transactions only.

| Label | Rule |
|-------|------|
| **active-recent** | ≥1 purchase in `(2020-03-02, 2020-03-31]` |
| **dormant** | ≥1 purchase before cutoff, but none in the 30-day window above |

**New tier:** Recency does not apply (no pre-cutoff history). New users form a single sampling cell `(new, n/a)`.

### Assignment order

1. Assign `purchase_tier` (new vs cold/light/medium/heavy).
2. For non-new tiers, assign `recency` (active-recent vs dormant).
3. Sample proportionally within each `(purchase_tier, recency)` cell.

---

## Sampling Algorithm

### Approach

Single-pass Spark stratified sample with proportional quotas (not uniform). Recommended over Pandas stats + Spark filter (breaks Glue portability) and over premature extraction to `src/` (over-scoped for v1 notebook).

### Step 1 — Build user-level features

Aggregate on full `transactions` (Spark):

```text
customer_id
purchase_count_pre_cutoff   -- count where t_dat <= cutoff
last_purchase_pre_cutoff    -- max(t_dat) where t_dat <= cutoff
```

Left-join `customers` on `customer_id` so customers with zero pre-cutoff transactions are retained.

### Step 2 — Label users

Derive `purchase_tier` and `recency` columns per rules above. Set `recency = null` (or `"n/a"`) for new tier.

### Step 3 — Compute quotas (largest remainder)

1. Count users per `(purchase_tier, recency)` cell in the full labeled population.
2. For each cell: `raw_quota = target_n × cell_count / total_labeled_population`.
3. Assign `floor(raw_quota)` to each cell; distribute remaining slots via largest-remainder method until sum equals exactly `target_n` (default 1000).
4. Cap each cell quota at its population size (if a cell is smaller than its quota, take all available users in that cell).

### Step 4 — Sample within each cell (reproducible)

For each cell with quota > 0:

```python
cell_df.orderBy(rand(random_seed), "customer_id").limit(quota)
```

- **Primary sort:** `rand(random_seed)` — deterministic given seed.
- **Tie-break:** `customer_id` ascending — stable ordering when random values tie.
- **Do not use** `sample(fraction=...)` alone — fraction-based sampling can vary across Spark versions and cluster topologies.

Union all cell samples → `sampled_customer_ids`.

Default `random_seed = 42`, overridable via `SAMPLING_RANDOM_SEED` env var.

### Step 5 — Filter and write output tables

| Output file | Filter rule |
|-------------|-------------|
| `transactions_train.csv` | `customer_id IN sampled_customer_ids` — **all dates** (pre- and post-cutoff) |
| `customers.csv` | `customer_id IN sampled_customer_ids` |
| `articles.csv` | `article_id IN DISTINCT article_ids from filtered transactions` |

Write to `dataset/sample/` as CSV (same filenames and schema as `dataset/full/`).

Skip `dataset/full/images/` entirely.

Optional: write `dataset/sample/sampling_manifest.json` with config, seed, per-cell quotas, and final counts for audit/reproducibility.

### Step 6 — Validation

Notebook ends with a validation cell that:

1. Compares purchase-tier proportions: full dataset vs sample.
2. Compares recency split within each non-new tier: full vs sample.
3. Reports row counts: users, transactions, articles.
4. Asserts `|sampled_users - target_n| ≤ tolerance` (tolerance accounts for cell population caps).

---

## Notebook Structure

**File:** `notebooks/stratified_user_sampling.ipynb`

| Cell block | Purpose |
|------------|---------|
| **1 — Config** | Dict + env var overrides (see below) |
| **2 — Spark session** | Local `master("local[*]")`; Glue branch when `AWS_EXECUTION_ENV` is set |
| **3 — Load** | Read CSVs from config paths |
| **4 — Functions** | Pure PySpark: `build_user_labels()`, `compute_quotas()`, `sample_users()`, `filter_and_write()` |
| **5 — Run pipeline** | Orchestrate Steps 1–5 |
| **6 — Validate** | Proportion comparison + assertions |
| **7 — Manifest (optional)** | Write `sampling_manifest.json` |

Functions are written in notebook cells but structured as standalone units so they can move to `pipelines/glue/stratified_user_sampling.py` later with no logic changes.

---

## Configuration

```python
CONFIG = {
    "input_path":   os.getenv("SAMPLING_INPUT_PATH",   "../dataset/full"),
    "output_path":  os.getenv("SAMPLING_OUTPUT_PATH",  "../dataset/sample"),
    "cutoff_date":  os.getenv("SAMPLING_CUTOFF_DATE",  "2020-03-31"),
    "recency_days": int(os.getenv("SAMPLING_RECENCY_DAYS", "30")),
    "target_n":     int(os.getenv("SAMPLING_TARGET_N",     "1000")),
    "random_seed":  int(os.getenv("SAMPLING_RANDOM_SEED",  "42")),
}
```

### Local vs Glue

| Concern | Local | AWS Glue |
|---------|-------|----------|
| Spark master | `local[*]` | Provided by Glue context |
| Input path | `../dataset/full` | `s3://fashion-reco-{env}/raw/` |
| Output path | `../dataset/sample` | `s3://fashion-reco-{env}/sample/` or back to `raw/` after ingest decision |
| Session init | `SparkSession.builder...` | `%glue_context` / existing `SparkContext` |

Guard with `IS_GLUE = os.getenv("AWS_EXECUTION_ENV") is not None`.

---

## Dependencies

- **PySpark only** in the notebook — no Pandas.
- Add `pyspark` to dev dependencies in `pyproject.toml` if not already present.

---

## Error Handling

| Condition | Behavior |
|-----------|----------|
| Missing input CSV | Fail fast with clear path in error message |
| Cell population < quota | Cap at population size; log warning in manifest |
| Sampled user count far below target_n | Validation cell warns (many small cells capped) |
| Output path exists | Overwrite mode (`mode("overwrite")`) with explicit config flag |

---

## Testing / Verification

Manual verification via notebook validation cell (no unit tests in v1 notebook scope):

1. Re-run notebook twice → identical `sampled_customer_ids`.
2. Tier proportions in sample within expected deviation from full (document actual deltas in validation output).
3. No orphaned transactions (every transaction row belongs to a sampled user).
4. No orphaned articles (every article appears in at least one filtered transaction).
5. Images directory not touched.

---

## Future Glue Migration

1. Move function cells to `pipelines/glue/stratified_user_sampling.py`.
2. Notebook becomes thin runner or is retired.
3. Change only `input_path` / `output_path` via Glue job parameters.
4. Register as Step Functions / Glue job step before `raw → clean` pipeline (or as one-time sample generation job).

---

## Success Criteria

- [ ] Notebook runs locally with `local[*]` on full H&M CSVs.
- [ ] ~1,000 users sampled with proportional tier and recency representation.
- [ ] Reproducible output given same config and seed.
- [ ] `dataset/sample/` contains filtered `articles.csv`, `customers.csv`, `transactions_train.csv`.
- [ ] No reference to or copy of `dataset/full/images/`.
- [ ] Code structured for Glue lift with path/config changes only.

---

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target size | ~1,000 users | User requirement (overrides 10K in infra doc for this notebook) |
| Sampling strategy | Proportional, not uniform | Preserve real-world segment distribution |
| Primary / secondary axis | Purchase tier → recency | User requirement |
| New user recency | N/A (single cell) | No pre-cutoff history to classify |
| Transaction scope in output | All dates for sampled users | Supports future training/eval with post-cutoff labels |
| Runtime | Pure PySpark, no Pandas | Glue portability per v1-hld §18.2 |
| Reproducibility | `rand(seed)` + `customer_id` tie-break | Deterministic across re-runs |
