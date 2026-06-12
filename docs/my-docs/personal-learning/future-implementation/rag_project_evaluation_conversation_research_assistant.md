# RAG Project Evaluation Discussion

## 1. Repository Discussed

Repository: [jamwithai/production-agentic-rag-course](https://github.com/jamwithai/production-agentic-rag-course/tree/main)

The conversation focused on how the RAG project evaluates or monitors:

- Retrieval performance
- Generation performance
- Ground-truth labels
- OpenSearch scoring
- LLM-based judging
- Whether the approach is acceptable in industry practice

---

## 2. Initial Question

The main question was:

> What strategy is used to evaluate the retrieval and generation steps in the RAG project? How is the ground-truth label generated, or was it already present in the dataset used for the project?

---

## 3. Dataset and Ground Truth

The project appears to use an arXiv-paper corpus for a research-assistant style RAG system.

There is no clear evidence that the project uses a pre-labeled benchmark dataset containing records such as:

```text
question, relevant_chunk_ids, reference_answer
```

So, the project does **not** appear to rely on pre-existing human-labeled ground truth for retrieval or generation evaluation.

Instead, the project uses:

- OpenSearch retrieval scores
- Langfuse tracing and observability
- An LLM-based document relevance grader in the agentic RAG workflow
- Prompt instructions to encourage grounded generation

---

## 4. Retrieval Evaluation Strategy

The retrieval step is handled through OpenSearch and, in the later agentic version, an LLM-based document grader.

The high-level flow is:

```text
User query
   ↓
OpenSearch retrieves top chunks
   ↓
Retrieved chunks are passed to document grader
   ↓
LLM grader returns yes/no relevance judgment
   ↓
If yes: generate answer
If no: rewrite query or retry retrieval
```

### 4.1 OpenSearch Retrieval Score

OpenSearch returns a `_score` for each retrieved result.

In the project’s keyword search setup, this score is mainly a BM25-style relevance ranking score.

It represents:

```text
How well this document or chunk matches the query compared with other retrieved chunks.
```

It is influenced by factors such as:

- Query-term matches
- Term frequency
- Rarity of terms across the index
- Field boosts, such as title, abstract, and content weighting
- Search mode, such as BM25 or hybrid search

It does **not** represent:

```text
accuracy
precision
recall
faithfulness
answer correctness
human-labeled relevance
```

### 4.2 Small Example of OpenSearch Score

User query:

```text
What is self-attention in transformers?
```

Possible retrieved chunks:

| Chunk | Content | OpenSearch score |
|---|---|---:|
| A | “Transformer models use self-attention to relate tokens...” | 12.8 |
| B | “Attention mechanisms are used in neural machine translation...” | 9.4 |
| C | “This paper discusses transformer efficiency in hardware...” | 4.1 |

Interpretation:

- Chunk A is ranked highest because it strongly matches the query.
- Chunk B is somewhat relevant.
- Chunk C has a lower score and may be less useful.

However, this score is only a retrieval ranking signal. It does not prove the chunk is truly relevant or sufficient for answering.

---

## 5. LLM-Based Retrieval Grading

The project uses an LLM grader to check whether retrieved documents are relevant to the user question.

The grader returns a binary judgment such as:

```text
binary_score: yes
```

or:

```text
binary_score: no
```

### Example

Question:

```text
What is self-attention in transformers?
```

Retrieved documents contain transformer and self-attention information.

The LLM grader may return:

```text
binary_score: yes
reasoning: The documents explain self-attention and transformer architecture.
```

Another question:

```text
What are the limitations of diffusion models?
```

If the retrieved documents are mostly about transformer limitations, the grader may return:

```text
binary_score: no
reasoning: The retrieved documents are about transformers, not diffusion models.
```

In that case, the agent may rewrite the query and retry retrieval.

---

## 6. Generation Evaluation Strategy

The project does **not** appear to use a separate LLM judge to evaluate the final generated answer.

The generation flow is closer to:

```text
Retrieved context + user question
   ↓
Generation prompt
   ↓
LLM generates answer
   ↓
Operational metadata is logged
```

The project uses prompt instructions to encourage grounding. The prompt tells the model to:

- Use only the retrieved research papers
- Cite specific papers or arXiv IDs
- Acknowledge when the context is insufficient
- Avoid making unsupported claims

But there is no clear evidence of a second LLM evaluation step like:

```text
Generated answer + retrieved context
   ↓
LLM judge
   ↓
Faithfulness / correctness / groundedness score
```

---

## 7. Metrics Actually Used or Logged

### 7.1 Retrieval-Related Signals

The project appears to track or use practical retrieval signals such as:

| Signal | Meaning |
|---|---|
| OpenSearch `_score` | Ranking score for retrieved chunks |
| `top_k` | Number of chunks requested |
| `chunks_returned` | Number of retrieved chunks |
| `unique_papers` | Number of distinct papers represented |
| `total_hits` | Total search matches |
| `search_mode` | BM25 or hybrid search mode |
| `retrieval_attempts` | Number of retrieval attempts |
| LLM document grade | Binary yes/no relevance judgment |

### 7.2 Generation-Related Signals

Generation is mainly monitored through operational metadata such as:

| Signal | Meaning |
|---|---|
| `answer_length` | Length of generated answer |
| `sources_used` | Number of sources available or cited |
| `context_length` | Size of context passed to the LLM |
| `prompt_length` | Size of the full prompt |
| `model_used` | LLM used for generation |
| `execution_time_ms` | Generation latency |
| Langfuse trace data | Observability and debugging information |

These are useful engineering metrics, but they are not the same as correctness or faithfulness metrics.

---

## 8. Correction About Precision and Recall

During the discussion, the phrase “Quality Metrics: Precision, recall, and relevance scoring” was mentioned.

After re-checking the repository, the corrected understanding is:

- The main README mentions precision, recall, and relevance scoring as course or architecture concepts.
- However, there is no clear evidence that the project actually implements an offline benchmark that computes Precision@k, Recall@k, MRR, or nDCG using ground-truth labels.

Therefore, the better statement is:

```text
The repo mentions precision and recall conceptually, but the implemented system primarily uses OpenSearch scores, retrieval statistics, Langfuse tracing, and LLM-based document relevance grading.
```

---

## 9. What Is Not Implemented as a Formal Evaluation Suite

The following formal retrieval metrics were not clearly found as implemented benchmark metrics:

| Retrieval metric | Status |
|---|---|
| Precision@k | Not clearly implemented |
| Recall@k | Not clearly implemented |
| MRR | Not clearly implemented |
| nDCG | Not clearly implemented |
| Hit Rate | Not clearly implemented |

The following formal generation metrics were also not clearly found as implemented metrics:

| Generation metric | Status |
|---|---|
| Faithfulness | Not clearly implemented |
| Answer relevance | Not clearly implemented |
| Answer correctness | Not clearly implemented |
| Citation accuracy | Not clearly implemented |
| BLEU / ROUGE | Not clearly implemented |
| RAGAS-style evaluation | Not clearly implemented |
| LLM judge for final answer | Not clearly implemented |

---

## 10. Was the Answer Grounded in Retrieved Context?

The project does not appear to explicitly evaluate final-answer grounding after generation.

Instead, it relies on:

1. Retrieval relevance grading before generation
2. A generation prompt that instructs the LLM to use only retrieved documents
3. Source citation instructions
4. Langfuse tracing and observability

So the implied logic is:

```text
If retrieved documents are relevant
and the prompt tells the LLM to use only those documents
then the answer is expected to be grounded.
```

But that expectation is not the same as a measured groundedness score.

A stronger grounding check would be:

```text
Generated answer + retrieved context
   ↓
LLM judge or evaluator
   ↓
Groundedness / faithfulness score
```

That does not appear to be implemented in the project.

---

## 11. Industry Practice Assessment

The project’s approach is acceptable as a baseline or early production-style RAG implementation.

It is useful for:

- Building a working RAG system
- Observing retrieval behavior
- Debugging with traces
- Detecting clearly irrelevant retrievals
- Iterating on prompts and search strategy

However, for mature industry systems, especially customer-facing or high-stakes systems, this is usually not enough.

A stronger industry setup would add:

| Area | Stronger industry practice |
|---|---|
| Retrieval | Recall@k, Precision@k, MRR, nDCG, Hit Rate |
| Generation | Faithfulness, answer relevance, correctness, citation accuracy |
| Ground truth | Human-labeled or human-reviewed golden dataset |
| Regression testing | Fixed eval set run after retrieval/prompt/model changes |
| Monitoring | Langfuse or similar observability plus automated eval scores |
| Human review | Sampling and manual validation of production answers |

---

## 12. Final Summary

The project uses a practical, observability-driven RAG evaluation approach rather than a complete benchmark-driven evaluation framework.

### Retrieval

Retrieval is evaluated or monitored using:

```text
OpenSearch relevance score
retrieval statistics
LLM-based document relevance grading
Langfuse tracing
```

### Generation

Generation is monitored using:

```text
prompt constraints
source/citation instructions
answer length
context length
sources used
execution time
Langfuse traces
```

### Ground Truth

Ground-truth labels are not clearly present in the dataset.

The project does not appear to use a formal labeled evaluation dataset for retrieval or generation.

### Main Limitation

The project checks whether retrieved documents are relevant, but it does not clearly check whether the final generated answer is faithful or grounded after generation.

A production-grade extension would add an evaluation layer such as:

```text
Generated answer + retrieved context + reference answer if available
   ↓
LLM judge / RAGAS / custom evaluator
   ↓
faithfulness, answer relevance, correctness, citation accuracy
```
