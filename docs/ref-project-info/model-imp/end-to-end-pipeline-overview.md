---
⚠️ **REFERENCE PROJECT DISCLAIMER** ⚠️

**THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION**

- **DO NOT USE** unless explicitly asked to reference old code
- **CURRENT IMPLEMENTATION** is in `system-design/` directory
- This file is for **REFERENCE ONLY** to understand legacy approaches
- All new development should follow current system design specifications

---



## 6. The Three Serving Pipelines

The system has three distinct serving pipelines. Confusing them is the most common source of misreading the code.


|                   | Pipeline 1              | Pipeline 2                 | Pipeline 3                              |
| ----------------- | ----------------------- | -------------------------- | --------------------------------------- |
| **Entry point**   | `customer_id`           | `customer_id`              | Free-text input                         |
| **Retrieval**     | Two-tower (Embedding B) | Two-tower (Embedding B)    | SentenceTransformer (Embedding A)       |
| **Re-ranking**    | CatBoost                | GPT-4o-mini                | None                                    |
| **Personalized?** | Yes                     | Yes                        | No                                      |
| **Key files**     | `ranking_predictor.py`  | `llm_ranking_predictor.py` | `recommenders.py` → `get_similar_items` |


---

### Pipeline 1: Main Personalized Recommendations (Two-Tower + CatBoost)

This is the primary flow described in Section 5. `customer_id` → QueryTower → ANN search over Embedding B → CatBoost re-ranking → sorted results.

---

### Pipeline 2: LLM Re-ranker (Two-Tower + GPT-4o-mini)

This is an alternative to Pipeline 1 — the **retrieval step is identical** (same two-tower model, same Embedding B, same ANN search). Only the re-ranking step differs: instead of CatBoost, GPT-4o-mini is used.

File: `recsys/inference/llm_ranking_predictor.py`.

The LLM receives each candidate article's structured features (age, month_sin, month_cos, product type, colour, garment group, etc.) formatted as a prompt and outputs a purchase probability. Capped at 20 candidates because one LLM call per candidate makes this slow (~15-30 seconds end-to-end).

**What `llm_ranking_predictor.py` is NOT:** it is not a text search tool and it does not use Embedding A. It is purely a drop-in replacement for CatBoost in the re-ranking step of the personalized pipeline.

---

### Pipeline 3: LLM Text Search (HyDE + Embedding A)

This is the part where **Embedding A** is used. A user types a natural-language request like "I'm going to the beach for a week-long vacation."

The actual flow (verified from `recsys/ui/recommenders.py`) is:

**Step 1 — LLM query expansion.** The raw user text is sent to GPT, which generates 3–5 specific item descriptions in H&M product language:

```
User: "I'm going to the beach for a week-long vacation"
GPT →
  "Floral print wrap dress with flutter sleeves"
  "Strappy nude block heel sandals"
  "Woven straw tote bag with leather handles"
  "Cropped denim jacket with raw hem"
```

**Step 2 — Encode each description and search.** For each generated description, `get_similar_items()` runs:

```python
description_embedding = embedding_model.encode(description)   # 384-dim
articles_fv.find_neighbors(description_embedding, k=25)       # Embedding A index
```

**Step 3 — Display by category, no re-ranking.** The top 5 matches per category are shown directly. There is no CatBoost or LLM scoring step after retrieval.

**Why this is not standard RAG.** In RAG the order is *retrieve → generate* (retrieved documents are injected into the LLM prompt to ground the answer). Here the order is *generate → retrieve* (the LLM generates hypothetical item descriptions first, and those are used as search queries). This pattern is known as **HyDE (Hypothetical Document Embeddings)**.

The reason HyDE works better than encoding the raw user query directly: the user query ("I'm going to the beach") lives in a different part of embedding space than article descriptions ("Floral print wrap dress..."). The LLM bridges that gap by producing text that is phrased the same way the catalog is phrased. Searching with the hypothetical description finds much closer neighbors in Embedding A than searching with the original request.

**What this pipeline does not do:** it does not know who the user is, does not use purchase history, and does not use the two-tower model or CatBoost.

This is also the answer to "why does `create_article_description` exist when we have `detail_desc`": it exists to feed this text search path with a richer, more normalized text string. `detail_desc` alone has gaps (some articles have no `detail_desc`) and lacks the product hierarchy context (color, group, section) that helps the SentenceTransformer make good neighbors in Embedding A space.
---

## 7. Putting It All Together: A Concrete End-to-End Example

Imagine customer `C42` opens the app on November 15, 2022.

**Offline state at that moment:**

- `C42` is in the `customers` feature group with `age=34`.
- The articles feature group has 11,820 articles, each with structured fields, an `article_description` string, a 384-dim text embedding, an image URL.
- The `candidate_embeddings` feature group has 11,820 16-dim vectors produced by the trained ItemTower.
- The trained `query_model`, `candidate_model`, and `ranking_model` are all in the registry.
- The `query` and `ranking` deployments are running.

**Runtime sequence:**

1. UI sends `{customer_id: "C42", transaction_date: "2022-11-15T12:16:25"}` to the `query` deployment.
2. `query_transformer.py` looks up `C42` and finds `age=34`, computes `month_sin=sin(11 * 2π/12) ≈ -0.5`, `month_cos=cos(...) ≈ -0.866`.
3. `query_model` (the QueryTower) takes `{customer_id: "C42", age: 34, month_sin: -0.5, month_cos: -0.866}` and outputs a 16-dim vector, call it `q = [0.12, -0.34, ...]`.
4. The request continues to the `ranking` deployment with `q` attached.
5. `ranking_transformer.py` calls `candidate_index.find_neighbors(q, k=100)`. Hopsworks searches the `candidate_embeddings` index and returns 100 article IDs.
6. The transformer queries the `transactions` feature group for `customer_id == "C42"` and removes any of those 100 IDs that `C42` already bought. Say 80 remain.
7. For each of those 80 articles, the transformer pulls structured attributes from the `articles` feature view (without the heavy text/embedding columns).
8. It attaches `age=34`, `month_sin=-0.5`, `month_cos=-0.866` to every row.
9. It sends the 80x14 feature matrix and the 80 article IDs to `ranking_predictor.py`, which runs `model.predict_proba`.
10. The CatBoost model returns 80 purchase probabilities.
11. The transformer pairs them with article IDs and sorts descending.
12. UI receives `[[0.94, "592846001"], [0.91, "536139006"], [0.88, "408554004"], ...]` and renders product cards.

The whole thing takes ~2-3 seconds.

If instead `C42` types "warm wool sweater for winter" into the LLM search box, none of the above runs. Instead (Pipeline 3):

1. GPT receives "warm wool sweater for winter" and generates specific item descriptions:
  - "Oversized chunky cable-knit wool sweater in cream"
  - "Slim-fit ribbed turtleneck in dark grey merino wool"
  - "Relaxed-fit longline knit cardigan with button front"
2. For each description, Streamlit calls `embedding_model.encode(description)` → 384-dim vector.
3. Calls `articles_fv.find_neighbors(description_embedding, k=25)` against the **Embedding A** index.
4. Renders the top 5 matches per category. No CatBoost scoring, no customer ID involved.

Three pipelines, two embedding spaces, different uses for each.
---
⚠️ **END OF REFERENCE PROJECT FILE** ⚠️

Remember: This is archived code. Use `system-design/` for current implementation.

---
