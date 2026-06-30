# PolicyMind AI — Benchmark Scores

> This file is updated after every version that changes retrieval or generation.
> Scores are produced by running `pytest tests/rag_eval/` against the insurance benchmark dataset.
> Never update scores manually — run the evaluation and copy the output.

---

## Benchmark Dataset

**File:** `tests/rag_eval/insurance_benchmark.json`  
**Questions:** 15+ real insurance policy Q&A pairs  
**Document types:** Health insurance, property insurance, motor insurance  
**Covers:** Waiting periods, exclusions, sub-limits, conditions, claim procedures

---

## Thresholds (CI enforced from V10 onward)

| Metric | Minimum Threshold | Target |
|---|---|---|
| Faithfulness | 0.85 | ≥ 0.87 |
| Answer Relevancy | 0.80 | ≥ 0.82 |
| Contextual Precision | 0.75 | ≥ 0.79 |
| Contextual Recall | 0.75 | ≥ 0.84 |

---

## Score History

### V0 — Pre-Stabilization (estimated, not measured)

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | ~0.60 | Noisy context, no reranking |
| Answer Relevancy | ~0.65 | Fixed chunking loses clause context |
| Contextual Precision | ~0.55 | Single-stage cosine only |
| Contextual Recall | ~0.58 | Vocabulary mismatch fails |
| **Overall** | **~0.60** | |

> ⚠️ Replace estimates with real scores after V1 completes.

---

### V1 — Stable Foundation

> Run after fixing all 5 bugs. This is the true baseline.

| Metric | Score | Delta vs V0 | Notes |
|---|---|---|---|
| Faithfulness | — | — | Run: `pytest tests/rag_eval/ -k faithfulness` |
| Answer Relevancy | — | — | |
| Contextual Precision | — | — | |
| Contextual Recall | — | — | |
| **Overall** | — | — | |

**Date measured:**  
**Commit:**  
**Command used:** `pytest tests/rag_eval/ -v --tb=short`

---

### V2 — Architecture Refactor

> Scores should be identical to V1. If they differ, something broke during refactor.

| Metric | Score | Delta vs V1 | Notes |
|---|---|---|---|
| Faithfulness | — | — | Should be 0.00 delta |
| Answer Relevancy | — | — | Should be 0.00 delta |
| Contextual Precision | — | — | Should be 0.00 delta |
| Contextual Recall | — | — | Should be 0.00 delta |

**Date measured:**  
**Commit:**

---

### V3 — Gemini Embeddings + ChromaDB

> Expected improvement: better embedding quality, persistent storage.

| Metric | Score | Delta vs V2 | Notes |
|---|---|---|---|
| Faithfulness | — | — | |
| Answer Relevancy | — | — | |
| Contextual Precision | — | — | Gemini embeddings vs Voyage AI |
| Contextual Recall | — | — | |

**Date measured:**  
**Commit:**  
**Embedding model:** gemini-embedding-001  
**Vector store:** ChromaDB

---

### V4 — Parent-Child Chunking

> Expected improvement: better context completeness → higher faithfulness.

| Metric | Score | Delta vs V3 | Notes |
|---|---|---|---|
| Faithfulness | — | — | Parent context = more complete clauses |
| Answer Relevancy | — | — | |
| Contextual Precision | — | — | |
| Contextual Recall | — | — | |

**Date measured:**  
**Commit:**  
**Chunk config:** parent=1500t, child=256t, overlap=32t

---

### V5 — Hybrid Retrieval (BM25 + Dense + RRF)

> Expected improvement: higher recall — captures exact keyword matches that dense misses.

| Metric | Score | Delta vs V4 | Notes |
|---|---|---|---|
| Faithfulness | — | — | |
| Answer Relevancy | — | — | |
| Contextual Precision | — | — | |
| Contextual Recall | — | — | Main target: recall improvement |

**Date measured:**  
**Commit:**  
**RRF config:** k=60, dense_weight=0.7, sparse_weight=0.3

---

### V6 — HyDE Query Expansion

> Expected improvement: better retrieval on vocabulary-mismatch queries (colloquial → formal insurance language).

| Metric | Score | Delta vs V5 | Notes |
|---|---|---|---|
| Faithfulness | — | — | |
| Answer Relevancy | — | — | Main target: relevancy on hard queries |
| Contextual Precision | — | — | |
| Contextual Recall | — | — | |

**Date measured:**  
**Commit:**  
**HyDE config:** temperature=0.3, max_tokens=200, fallback=original query

---

### V7 — Cross-Encoder Reranking

> Expected improvement: higher precision — cross-encoder catches negations, conditions, exceptions bi-encoders miss.

| Metric | Score | Delta vs V6 | Notes |
|---|---|---|---|
| Faithfulness | — | — | |
| Answer Relevancy | — | — | |
| Contextual Precision | — | — | Main target: precision improvement |
| Contextual Recall | — | — | May decrease slightly (top-20 → top-8) |

**Date measured:**  
**Commit:**  
**Reranker:** BAAI/bge-reranker-v2-m3, FP16, top-20 → top-8

---

### V8 — Context Compression

> Expected improvement: lower hallucination rate, higher faithfulness — only relevant sentences sent to Gemini.

| Metric | Score | Delta vs V7 | Notes |
|---|---|---|---|
| Faithfulness | — | — | Main target: faithfulness improvement |
| Answer Relevancy | — | — | |
| Contextual Precision | — | — | |
| Contextual Recall | — | — | |

**Date measured:**  
**Commit:**  
**Compression:** temperature=0.0, max_tokens=300, min_chunks=2 fallback  
**Token reduction:** —% average (log and fill in)

---

### V9 — Agentic RAG (LangGraph)

> Expected improvement: better performance on difficult/ambiguous queries via retrieve→grade→rewrite loop.

| Metric | Score | Delta vs V8 | Notes |
|---|---|---|---|
| Faithfulness | — | — | |
| Answer Relevancy | — | — | |
| Contextual Precision | — | — | |
| Contextual Recall | — | — | Main target: hard query improvement |

**Date measured:**  
**Commit:**  
**Agent config:** max_retries=2, confidence_threshold=0.4

---

### V10 — Evaluation Suite Baseline

> V10 adds the full evaluation harness. Re-run all prior versions if possible for apples-to-apples comparison.

| Metric | Score | Delta vs V9 | Notes |
|---|---|---|---|
| Faithfulness | — | — | First formally enforced score |
| Answer Relevancy | — | — | |
| Contextual Precision | — | — | |
| Contextual Recall | — | — | |

**Date measured:**  
**Commit:**  
**Benchmark questions:** — (fill in count)  
**CI:** rag-eval job active on main branch

---

### V11+ — Production Features

> Production features (Redis, streaming, rate limiting) should not affect RAG scores.
> Re-run to confirm no regression.

| Version | Faithfulness | Relevancy | Precision | Recall | Notes |
|---|---|---|---|---|---|
| V11 | — | — | — | — | Confirm no regression from production features |
| V12 | — | — | — | — | Confirm Qdrant scores match ChromaDB |
| V13 | — | — | — | — | CI baseline locked |
| V14 | — | — | — | — | Production deployment smoke test |

---

## Latency Benchmarks (Locust — added V11)

**Test config:** 50 users, 5/s ramp, 5 min run

| Version | p50 (cache miss) | p95 (cache miss) | p50 (cache hit) | p95 (cache hit) |
|---|---|---|---|---|
| V11 | — | — | — | — |
| V12 | — | — | — | — |
| V14 (prod) | — | — | — | — |

**SLA targets:**
- p95 cache miss: < 4500 ms
- p95 cache hit: < 200 ms

---

## How to Run Evaluations

```bash
# Full RAG eval (needs real API keys)
cd backend
pytest tests/rag_eval/ -v --tb=short

# Single metric
pytest tests/rag_eval/ -v -k "faithfulness"

# With score output saved
pytest tests/rag_eval/ -v --tb=short 2>&1 | tee docs/eval_results/v$(VERSION).txt

# Load test
locust -f tests/locustfile.py --headless -u 50 -r 5 --run-time 5m --host http://localhost:8000
```

---

## Notes on Interpretation

- **Faithfulness** measures whether the answer is grounded in retrieved context (no hallucination). Most important metric.
- **Answer Relevancy** measures whether the answer addresses the question asked.
- **Contextual Precision** measures whether the retrieved chunks are actually relevant (noise in context = lower precision).
- **Contextual Recall** measures whether all relevant information was retrieved (missing clauses = lower recall).

A system can have high Faithfulness but low Recall (answers what it retrieved, but retrieved the wrong thing). Both must be high.