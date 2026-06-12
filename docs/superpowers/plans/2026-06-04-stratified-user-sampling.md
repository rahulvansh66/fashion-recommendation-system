# Stratified User Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PySpark notebook that stratified-samples ~1,000 users from the full H&M dataset into `dataset/sample/`, and align infra docs to the new dev sample scale.

**Architecture:** Single-pass Spark pipeline — label all customers by purchase tier and recency (pre-cutoff), compute proportional quotas via largest remainder, sample deterministically per cell, filter three CSV tables, validate proportions.

**Tech Stack:** PySpark 3.x, Jupyter notebook, JSON manifest (no Pandas)

**Spec:** `docs/superpowers/specs/2026-06-04-stratified-user-sampling-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `notebooks/stratified_user_sampling.ipynb` | Create | Config, Spark session, sampling functions, pipeline run, validation |
| `docs/system-design/v1/v1-infrastructure-layer.md` | Modify | Dev sample scale 10K → ~1K users |
| `docs/system-design/infrastructure-layer.md` | Modify | Same scale update (parent doc) |
| `docs/system-design/v1/v1-hld.md` | Modify | Dev sample mention in design principle #7 |
| `docs/system-design/v1/v1-requirements.md` | Modify | CON-02 and NFR dev sample references |
| `docs/system-design/project-structure.md` | Modify | `dataset/sample/` comment |

---

### Task 1: Create notebook — config and Spark session

**Files:**
- Create: `notebooks/stratified_user_sampling.ipynb`

- [ ] **Step 1:** Add markdown intro + config cell with env-driven `CONFIG` dict (cutoff `2020-03-31`, recency 30, target 1000, seed 42)
- [ ] **Step 2:** Add Spark session cell with `local[*]`, Glue guard via `AWS_EXECUTION_ENV`, driver memory config for large CSVs

---

### Task 2: Create notebook — core functions

**Files:**
- Modify: `notebooks/stratified_user_sampling.ipynb`

- [ ] **Step 1:** `build_user_labels(customers, transactions, cutoff_date, recency_days)` — left join stats, tier + recency columns
- [ ] **Step 2:** `compute_quotas(labeled_df, target_n)` — largest remainder on driver (~11 cells)
- [ ] **Step 3:** `sample_users(labeled_df, quotas, random_seed)` — per-cell `orderBy(rand(seed), customer_id).limit(quota)`
- [ ] **Step 4:** `write_single_csv(df, output_path)` — coalesce(1) + rename part file helper
- [ ] **Step 5:** `filter_and_write(...)` — filter customers, all-date transactions, derived articles; skip images

---

### Task 3: Create notebook — run, validate, manifest

**Files:**
- Modify: `notebooks/stratified_user_sampling.ipynb`

- [ ] **Step 1:** Pipeline orchestration cell — load CSVs, run full flow, write to `dataset/sample/`
- [ ] **Step 2:** Validation cell — tier/recency proportion comparison (full vs sample), row counts, assert `|sampled - target_n| ≤ 50`
- [ ] **Step 3:** Manifest cell — write `sampling_manifest.json` with config, quotas, counts

---

### Task 4: Update infra and related docs

**Files:**
- Modify: `docs/system-design/v1/v1-infrastructure-layer.md`
- Modify: `docs/system-design/infrastructure-layer.md`
- Modify: `docs/system-design/v1/v1-hld.md`
- Modify: `docs/system-design/v1/v1-requirements.md`
- Modify: `docs/system-design/project-structure.md`

- [ ] **Step 1:** Replace `10K users / 5K items / 100K transactions` with `~1K users` and note articles/transactions are derived from stratified sample (produced by `notebooks/stratified_user_sampling.ipynb`)

---

### Task 5: Verify notebook runs locally

- [ ] **Step 1:** `pip install pyspark`
- [ ] **Step 2:** Execute notebook cells or run equivalent script against `dataset/full/`
- [ ] **Step 3:** Confirm `dataset/sample/` has three CSVs, ~1000 users, manifest written

---
