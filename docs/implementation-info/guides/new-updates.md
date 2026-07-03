# Planned changes vs current implementation

**Purpose:** Track intentional deviations from the current v1 temporal-split design before updating authoritative docs (`v1-requirements.md`, `ranking-model-training-guide.md`, etc.).

**Status:** Proposed — not yet reflected in v1 requirements or pipelines.

---

## Change 1 — Snap-date temporal splits (2 train · 2 val · 1 test · 5 drift)

### What we have today

Current v1 contract ([`v1-requirements.md`](../../system-design/v1/v1-requirements.md) FR-BATCH-02) uses **calendar date ranges** per split:

| Split | `t_dat` range | Role |
|-------|---------------|------|
| Train | start → **2020-03-31** | Ranker training |
| Val | **2020-04-01** → **2020-05-15** | Tuning / early stopping |
| Test | **2020-05-16** → **2020-06-30** | Final acceptance |
| Drift 1–3 | **Jul / Aug / Sep 2020** (monthly) | Model Monitor only |

Positives are purchases **inside** each split’s date range. Features are computed for that split. There is no explicit snap date or forward label week.

The Kaggle reference ([`quick-and-easy-model-build-guide.md`](./quick-and-easy-model-build-guide.md)) uses **snap dates** instead: features frozen at a snap date, label = purchases in the **7 days after** that snap. It has 2 train snaps + 1 test snap (Sept 2020), with leaky train/val overlap.

### What we plan instead

Adopt the **snap-date + forward label week** pattern from the Kaggle guide, but aligned to early-2020 dates and with **non-leaky** role separation:

| Concept | Rule |
|---------|------|
| **Snap date** | Feature cutoff — all features use `t_dat <= snap_date` |
| **Label window** | The 7 calendar days **after** the snap: `(snap_date + 1)` → `(snap_date + 7)` |
| **One row** | One `(customer_id, article_id)` pair at one snap date; `SOLD = 1` if purchased in that row’s label window |
| **One model** | Stack all train snap rows → single ranker `fit`; val/test/drift rows are **never** used in training |

#### Full snap schedule (10 snaps)

| Snap date | Role | Features (`t_dat <=`) | Label window (`SOLD = 1` if purchase here) |
|-----------|------|----------------------|---------------------------------------------|
| `2020-03-31` | **Train 1** | Mar 31 | `2020-04-01` – `2020-04-07` |
| `2020-04-07` | **Train 2** | Apr 7 | `2020-04-08` – `2020-04-14` |
| `2020-04-14` | **Val 1** | Apr 14 | `2020-04-15` – `2020-04-21` |
| `2020-04-28` | **Val 2** | Apr 28 | `2020-04-29` – `2020-05-05` |
| `2020-05-15` | **Test** | May 15 | `2020-05-16` – `2020-05-22` |
| `2020-05-31` | **Drift 1** | May 31 | `2020-06-01` – `2020-06-07` |
| `2020-06-30` | **Drift 2** | Jun 30 | `2020-07-01` – `2020-07-07` |
| `2020-07-31` | **Drift 3** | Jul 31 | `2020-08-01` – `2020-08-07` |
| `2020-08-31` | **Drift 4** | Aug 31 | `2020-09-01` – `2020-09-07` |
| `2020-09-15` | **Drift 5** | Sep 15 | `2020-09-16` – `2020-09-22` |

```text
Mar 31   Apr 7    Apr 14   Apr 28   May 15   May 31   Jun 30   Jul 31   Aug 31   Sep 15
  ●        ●        ●        ●        ●        ●        ●        ●        ●        ●
  T1       T2       V1       V2      TEST      D1       D2       D3       D4       D5
  |←L→|    |←L→|    |←L→|    |←L→|    |←L→|    |←L→|    |←L→|    |←L→|    |←L→|    |←L→|
```

#### How each split is used

| Split | Snap dates | Used in `fit()`? | Purpose |
|-------|------------|------------------|---------|
| **Train** | Mar 31, Apr 7 | Yes | Learn `P(purchase next week \| features@snap)` |
| **Val** | Apr 14, Apr 28 | No | Early stopping, hyperparameter tuning |
| **Test** | May 15 | No | Final acceptance metrics (report once) |
| **Drift** | May 31, Jun 30, Jul 31, Aug 31, Sep 15 | No | Score-only — plot metrics vs snap date to detect performance decay over time |

**Drift analysis:** Run the same evaluation metrics on all 5 drift snaps and plot them in time order (D1 → D5). Expect gradual decay as the gap from training grows. Drift snaps are **not** used for model selection or promotion gates.

#### Example (one customer, multiple snaps)

Customer Alice, article Red Dress:

| Snap date | Role | Alice’s feature state | Label week | SOLD |
|-----------|------|----------------------|------------|------|
| `2020-03-31` | Train 1 | 5 purchases through Mar 31 | Apr 1–7 | 1 (bought Apr 5) |
| `2020-04-07` | Train 2 | 6 purchases through Apr 7 | Apr 8–14 | 1 (bought Apr 12) |
| `2020-04-14` | Val 1 | History through Apr 14 | Apr 15–21 | scored only |
| `2020-05-15` | Test | History through May 15 | May 16–22 | scored only |

Same customer and article can appear in multiple rows — each row is a **different snap + label week**, not a duplicate.

#### Sampling (unchanged from v1 intent)

Keep v1 negative strategy where applicable:

- **Positives:** real `(customer_id, article_id)` purchases in that snap’s label window.
- **Negatives:** 5 window-aware negatives per positive (same customer, article not purchased in label window, not in `seen` set before label window).
- **Ratio:** 1 : 5; `scale_pos_weight = 5`.

Negatives must be drawn **per snap date** — do not pool across snaps.

#### Masks (conceptual)

```python
train_mask  = df["SNAP_DATE"].isin(["2020-03-31", "2020-04-07"])
val_mask    = df["SNAP_DATE"].isin(["2020-04-14", "2020-04-28"])
test_mask   = df["SNAP_DATE"] == "2020-05-15"
drift_mask  = df["SNAP_DATE"].isin([
    "2020-05-31", "2020-06-30", "2020-07-31", "2020-08-31", "2020-09-15"
])
```

### Why this change

| Motivation | Detail |
|------------|--------|
| **Matches online inference** | Production asks “given today’s features, what happens next week?” — snap-date rows mirror that |
| **Fixes Kaggle leakage** | Reference notebook trains on the same snap it validates on; we hold out val/test/drift strictly |
| **Richer drift signal** | 5 drift snaps (Jun → Sep) give a finer decay curve than 3 monthly buckets |
| **More training volume** | Two stacked train snaps ≈ 2× positive rows vs one cutoff |

### Docs / code to update when adopted

- [`v1-requirements.md`](../../system-design/v1/v1-requirements.md) — FR-BATCH-02 temporal table, CON-09, ranker label definition
- [`ranking-model-training-guide.md`](./ranking-model-training-guide.md) — §3 splits, §4 positives/negatives
- [`quick-and-easy-model-build-guide.md`](./quick-and-easy-model-build-guide.md) — §2.3 snap timeline (reference vs planned v1)
- Feature-engineering / training-table build jobs — emit rows keyed by `SNAP_DATE`
- Drift monitoring (FR-BATCH-05) — evaluate on 5 drift snaps instead of 3 monthly windows

### Constraints / notes

- H&M transaction data ends ~**2020-09-22**; Drift 5 label window (`Sep 16–22`) is the last usable week.
- Label window always starts **the day after** the snap date (no same-day overlap between features and labels).
- Promotion gate still uses **test snap only**; drift is monitoring-only.
