# RAG Chatbot Design & Evaluation Guide

| Field | Value |
|---|---|
| **Status** | Reference Design (Out of V1 Scope) |
| **Dataset** | H&M Articles — `detail_desc` + structured metadata |
| **Related Docs** | [`schema-info.md`](../../system-design/schema-info.md) · [`v1-hld.md`](../../system-design/v1/v1-hld.md) |

> **Note:** RAG is explicitly out of V1 scope (see `v1-hld.md §1.2`). This guide captures the design for a future iteration.

---

## Table of Contents

1. [Corpus Fields to Index](#1-corpus-fields-to-index)
2. [Chunk Construction Strategy](#2-chunk-construction-strategy)
3. [RAG Pipeline Architecture](#3-rag-pipeline-architecture)
4. [Evaluation Without Labels — Industry Best Practices](#4-evaluation-without-labels--industry-best-practices)
   - [Strategy A: Synthetic Testset Generation](#strategy-a-synthetic-testset-generation-reference-creation)
   - [Strategy B: Reference-Free Metrics (LLM-as-Judge)](#strategy-b-reference-free-metrics-no-labels-at-all)
   - [Strategy C: AWS Bedrock Model Evaluation](#strategy-c-aws-bedrock-model-evaluation-managed-llm-as-judge)
5. [How Synthetic Triples Evaluate Both Stages](#5-how-synthetic-triples-evaluate-both-stages)
6. [Recommended Evaluation Pipeline](#6-recommended-evaluation-pipeline)
7. [Key Design Choices for This Dataset](#7-key-design-choices-for-this-dataset)
8. [Relevant Libraries](#8-relevant-libraries)

---

## 1. Corpus Fields to Index

From the `articles` table (see `schema-info.md`), combine the following per-document for each article:

| Field | Role in chunk |
|---|---|
| `prod_name` | Document title / anchor |
| `product_type_name`, `product_group_name` | Category signal |
| `colour_group_name`, `perceived_colour_master_name`, `graphical_appearance_name` | Visual / style signal |
| `department_name`, `section_name`, `garment_group_name`, `index_group_name` | Taxonomy hierarchy |
| `detail_desc` | **Primary narrative text** — the main retrieval target |

> The schema has only `detail_desc` as the free-text field. There is no separate `description` column.

Store the following as **metadata alongside each chunk** for pre-filtered retrieval:

- `article_id`, `product_type_name`, `colour_group_name`, `department_name`, `index_group_name`

---

## 2. Chunk Construction Strategy

Each article maps to **one self-contained document chunk**. Since `detail_desc` is item-level (no multi-paragraph structure), the strategy is to **enrich** each chunk with structured metadata fields so the LLM receives full context:

```text
[Document for article_id=0110065001]
Product: Slim Fit Shirt (Shirts / Menswear)
Color: Dark Blue | Solid
Department: Menswear, Section: Shirts & Blouses
Description: A slim-fit shirt in airy cotton fabric with a turn-down collar,
button placket, and rounded hem. Long sleeves with buttons at the cuffs.
```

**Scale:**
- ~150–250 tokens per chunk
- 105K articles → manageable corpus (~15–25M tokens total)

**Why enrich?** `detail_desc` in H&M is often sparse or repetitive. Injecting structured attributes (color, department, type) significantly improves retrieval quality for style-based queries.

---

## 3. RAG Pipeline Architecture

```
User Query
    │
    ▼
[Query Understanding]
  ├─ Extract intent (style? color? occasion?)
  └─ Optional: metadata filter extraction (colour, department)
    │
    ▼
[Embedding Model] ──► Query Vector
    │
    ▼
[Vector Store] ──── Pre-filter (colour_group, department)
  └─ Semantic search → Top-K chunks (K=5–10)
    │
    ▼
[LLM Generator]
  └─ System prompt + retrieved chunks + user query → Response
```

### AWS-Native Stack

| Component | Service |
|---|---|
| Embeddings | Amazon Titan Embeddings v2 (`amazon.titan-embed-text-v2:0`) |
| Vector store | Amazon OpenSearch Serverless (k-NN) or Bedrock Knowledge Base |
| LLM | Claude 3 Haiku / Sonnet via Amazon Bedrock |
| Orchestration | Amazon Bedrock Knowledge Bases (fully managed RAG) or LangChain / LlamaIndex on ECS |

**Amazon Bedrock Knowledge Bases** handles ingestion, chunking, embedding, and retrieval natively. Articles are ingested as JSONL into S3, and the Knowledge Base is pointed at that S3 prefix — removing most infrastructure plumbing.

**Hybrid search** (BM25 + dense vector) outperforms pure semantic search for product name lookups. OpenSearch Serverless supports both natively. Using metadata pre-filters before k-NN search at 105K articles significantly reduces retrieval noise.

---

## 4. Evaluation Without Labels — Industry Best Practices

The H&M dataset has no ground-truth QA pairs, so traditional precision/recall metrics don't directly apply. Two complementary strategies cover both offline evaluation and production monitoring.

---

### Strategy A: Synthetic Testset Generation (Reference Creation)

**Core idea:** Use an LLM to generate QA pairs from the article corpus itself, creating pseudo-ground-truth `(question, ground_truth_answer, ground_truth_contexts)` triples.

RAGAS `TestsetGenerator` handles this end-to-end:

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_aws import BedrockChat

generator_llm = LangchainLLMWrapper(
    BedrockChat(model_id="anthropic.claude-3-haiku-20240307-v1:0")
)
generator = TestsetGenerator(llm=generator_llm)

# articles_docs = 105K article chunks as LangChain Documents
testset = generator.generate_with_langchain_docs(
    articles_docs,
    testset_size=500,
    distributions={"simple": 0.5, "reasoning": 0.3, "multi_context": 0.2}
)
```

**Distribution types:**
- `simple` — single-chunk factual: *"What materials are used in the slim-fit cotton shirts?"*
- `reasoning` — requires inference: *"Why would a Solid graphical appearance affect fabric choice?"*
- `multi_context` — spans multiple chunks: *"Which dark blue men's shirts are slim fit and have long sleeves?"*

**Quality note:** RAGAS uses a generator + critic multi-pass to filter low-quality synthetic pairs before they enter the evaluation set. The "ground truth" is LLM-generated so it can be imperfect — the critic pass reduces this noise.

With these triples you can evaluate **both retrieval and generation** (see [§5](#5-how-synthetic-triples-evaluate-both-stages)).

---

### Strategy B: Reference-Free Metrics (No Labels at All)

For production monitoring where generating a testset for every query is impractical, RAGAS provides **reference-free metrics** that use LLM-as-judge internally. No ground truth is needed at inference time.

| Metric | What it measures | Needs labels? |
|---|---|---|
| **Faithfulness** | Are all claims in the answer grounded in retrieved chunks? (factual consistency) | No |
| **Answer Relevancy** | Is the answer actually answering the question asked? | No |
| **Context Precision** | Are the retrieved chunks actually relevant to the question? | No |
| **Context Recall** | Were all relevant documents retrieved? | Yes (needs reference) |
| **Noise Sensitivity** | How much does the answer change with noisy / irrelevant context? | No |

For the pure no-label scenario, use **Faithfulness + Answer Relevancy + Context Precision**. With the synthetic testset from Strategy A, all five metrics become available.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question": [...],
    "answer": [...],       # LLM generated answer
    "contexts": [[...]],   # retrieved chunks per question
})

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=evaluator_llm,       # Claude or Titan via Bedrock
    embeddings=evaluator_embeddings,
)
```

---

### Strategy C: AWS Bedrock Model Evaluation (Managed LLM-as-Judge)

AWS Bedrock's native **Model Evaluation** service supports RAG evaluation jobs:

1. Define a **prompt dataset** (synthetic testset or production samples)
2. Choose an **evaluator model** (e.g. Claude 3 Sonnet as judge)
3. Specify **evaluation dimensions**: helpfulness, faithfulness, coherence, harmlessness

This runs as a managed batch job and produces a scored report. Best suited for **offline cadence evaluation** (weekly / release-gated) rather than real-time per-request scoring.

---

## 5. How Synthetic Triples Evaluate Both Stages

The synthetic `(question, ground_truth_answer, ground_truth_contexts)` triples serve as pseudo-ground-truth for **both stages independently**.

### Retrieval Stage

`ground_truth_contexts` = the source chunks the generator LLM used to produce the synthetic answer. These become the reference for retrieval quality.

| Metric | What you compare |
|---|---|
| **Context Recall** | Did your retriever fetch the ground-truth chunk? `# ground-truth chunks in retrieved set / total ground-truth chunks` |
| **Context Precision** | Of the K retrieved chunks, how many are relevant? `# relevant retrieved / K` |
| **Hit Rate / MRR** | Was the ground-truth chunk in top-K, and at what rank? |

This answers: **is your embedding model + vector search working for fashion queries?**

### Generation Stage

`ground_truth_answer` = the LLM-generated reference answer. This becomes the reference for generation quality.

| Metric | What you compare |
|---|---|
| **Faithfulness** | Does the generated answer stick to retrieved context? (LLM-as-judge, no reference needed) |
| **Answer Correctness** | How close is the generated answer to the reference? (semantic similarity + factual overlap) |
| **Answer Relevancy** | Does the answer address the question? (LLM-as-judge, no reference needed) |

This answers: **is the LLM generating accurate, grounded responses from retrieved context?**

### Cascade Diagnostic

The two stages are not fully independent. If retrieval fails, generation will also fail regardless of LLM quality. The diagnostic pattern:

| Context Recall | Faithfulness | Answer Correctness | Diagnosis |
|---|---|---|---|
| High | Low | Low | Retrieval ok — LLM is hallucinating |
| Low | Low | Low | Retrieval is the bottleneck |
| High | High | Low | Synthetic ground truth was noisy (QA generation quality issue) |
| High | High | High | Both stages working |

---

## 6. Recommended Evaluation Pipeline

```
Offline (once per release):
  1. TestsetGenerator → 500 synthetic QA pairs from detail_desc corpus
  2. Run RAG pipeline on each question
  3. RAGAS evaluate() → Faithfulness, Answer Relevancy, Context Precision,
     Context Recall, Answer Correctness (with synthetic ground truth)
  4. Log results to S3 / CloudWatch as baseline

Online (production monitoring, sampled):
  5. Sample ~5% of live queries
  6. Run Faithfulness + Answer Relevancy asynchronously (async Bedrock call)
  7. Alert via CloudWatch alarm if moving average drops below threshold
```

---

## 7. Key Design Choices for This Dataset

- **`detail_desc` is sparse** — many H&M articles have short or repetitive descriptions. Always enrich chunks with structured fields (color, department, type); otherwise retrieval quality degrades significantly on style-based queries.
- **Hybrid search** (BM25 + dense) outperforms pure semantic search for product name lookups (e.g. "polo shirt"). OpenSearch Serverless supports both natively.
- **Metadata pre-filtering** before semantic search is critical at 105K articles. Filtering by `product_type_name` or `department_name` before k-NN cuts retrieval noise substantially.
- **RAGAS with Bedrock**: RAGAS supports `LangchainLLMWrapper` wrapping `BedrockChat`, so no OpenAI keys are needed — the full stack stays within the AWS ecosystem.
- **Chunk granularity**: One chunk per article is appropriate here because `detail_desc` is item-scoped. If a future dataset has longer product narratives (multi-paragraph), sliding window chunking with overlap should be considered.

---

## 8. Relevant Libraries

```
ragas>=0.2           # evaluation framework (reference-free metrics + synthetic testset)
langchain-aws        # BedrockChat + BedrockEmbeddings wrappers
boto3                # Bedrock API calls
opensearch-py        # OpenSearch Serverless client
```
