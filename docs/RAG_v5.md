# RAG v5 — Hybrid Retrieval (BM25 + Dense + RRF)

**Date:** 2026-06-30
**Scope:** Implementing Hybrid Retrieval combining dense vector search and sparse keyword search, fused via Reciprocal Rank Fusion (RRF).

---

## Goal

- Implement sparse search using BM25 (`rank-bm25`) to catch exact keyword matches.
- Compute cosine similarity dense search in memory over child chunk embeddings.
- Combine and rank candidate chunks using Reciprocal Rank Fusion (RRF) with configurable parameters.
- Resolve parent text blocks for fused candidates to pass complete context to the LLM.
- Remove outdated heuristic-based scoring helpers to clean up dead code.

---

## Previous Limitation (why this was needed)

Previously, retrieval relied solely on dense vector search (cosine similarity). Dense embeddings are excellent at capturing high-level semantic meaning, but fail in two critical insurance query scenarios:
1. **Exact Numbers/Codes**: Queries asking about specific percentages, monetary limits, or section numbers (e.g. "Section 4(d)") are often missed by dense search because the embedding space merges distinct numbers together.
2. **Out-of-vocabulary Keywords**: Rare medical terms or specific clauses might get buried under more semantically dominant words.

Keyword-based search (BM25) provides the exact token matching needed to solve these failures, while dense search ensures semantic variants are still retrieved.

---

## Concepts Introduced

1. **Sparse Search (BM25Okapi)**: Term-frequency inverse-document-frequency ranking using the standard `rank-bm25` package.
2. **Reciprocal Rank Fusion (RRF)**: A rank-based fusion algorithm that combines multiple ranked lists without needing score normalization. It calculates a unified score using:
   $$\text{score}(d) = \frac{\alpha}{k + \text{rank}_{\text{dense}}} + \frac{\beta}{k + \text{rank}_{\text{bm25}}}$$
   where $k=60$ acts as a smoothing factor, $\alpha=0.7$ weighs the dense ranking, and $\beta=0.3$ weighs the sparse ranking.

---

## Design Decisions

- **In-Memory Dense Search**: Dense search similarity computations are performed in memory on numpy matrices over the already cached child chunk embeddings. This provides blazing fast, zero-latency vector scoring without needing extra roundtrips or database calls.
- **De-coupled BM25 Index Serialization**: At ingestion time, the BM25 index is constructed and saved as a local pickle file (`backend/cache/bm25_{doc_id}.pkl`).
- **Lazy BM25 Construction**: When loading cached documents from ChromaDB, if the corresponding BM25 index file is missing, the system lazily builds it on-the-fly from the cached chunk texts. This prevents redundant Gemini API embedding requests, bypassing cost and rate limit issues.
- **Pruning Heuristic Dead Code**: Removed the old hand-crafted heuristics (such as content density, length, and section-specific rules) from `vector_store.py` in compliance with Rule 15 (No dead code, file size < 300 lines).

---

## Trade-offs

- **Memory vs. Disk Storage**: Saving BM25 indexes as `.pkl` files adds slight disk usage in `backend/cache/`, but it allows instant index loading and scoring, removing runtime recalculation overhead.
- **Simultaneous Retrieval Latency**: Running both BM25 and dense search sequentially adds minimal latency (~1-3 milliseconds) because all indexing and vector data are kept in memory.

---

## Changed Files

| File | Changes |
|---|---|
| `backend/config.py` | Added RRF parameters (`rrf_k`, `rrf_dense_weight`, `rrf_sparse_weight`) to Settings. |
| `backend/requirements.txt` | Added `rank-bm25==0.2.2` runtime dependency. Removed unused `voyageai==0.3.4` dependency. |
| `backend/services/retrieval/bm25_index.py` | **[NEW]** Tokenizer, serialization, and retrieval search logic for rank-bm25 indexes. |
| `backend/services/retrieval/hybrid_search.py` | **[NEW]** In-memory dense matrix cosine search and Reciprocal Rank Fusion rankings combiner. |
| `backend/services/retrieval/vector_store.py` | Refactored `advanced_universal_retrieval` to fuse Dense and BM25 search. Removed dead heuristic scoring functions. |
| `backend/services/ingestion/chunker.py` | Updated `load_or_create_chunks` to build BM25 indexes and lazily rebuild them if missing on cache loads. |
| `backend/tests/unit/test_hybrid_search.py` | **[NEW]** Unit tests for RRF rank preferences and disjoint sets handling. |

---

## New Behavior

Retrieval queries now run dense and sparse searches simultaneously over the child chunks:
1. Dense search ranks chunks by cosine similarity.
2. BM25 search ranks chunks by keyword term match.
3. RRF combines rankings (with dense weight $0.7$ and sparse weight $0.3$).
4. Fused chunks map to parent text and return the complete clauses to the generator.

---

## Testing Added

1. **Unit Tests (`test_hybrid_search.py`)**:
   - `test_rrf_prefers_docs_ranked_high_by_both`: Asserts that candidates ranked highly in both dense and sparse retrieval win the top position.
   - `test_rrf_handles_disjoint_sets`: Asserts that candidates present in only one of the search lists are still correctly scored and returned without duplicates.
2. **Verification Script (`verify_v5_e2e.py`)**: Runs complete end-to-end ingestion and retrieval, asserting RRF rank fusion, file caching, parent mapping, and weight tuning effects.

---

## Benchmark Results (V3 onward)

Since the evaluation harness is not yet implemented (scheduled for **V10**), baseline RAGAS/DeepEval scores are not measured. 
Manual evaluation on the CMS SBC insurance PDF confirms:
- Hybrid search successfully surfaces chunks matching exact term lists (like "Coverage Examples" or "Other Covered Services").
- Dense vector search scores reach up to `0.8083`.
- BM25 keyword scores reach up to `6.2370`.
- RRF combines them seamlessly without scale mismatches.
- Ranking orders change dynamically when adjusting RRF weights in memory (e.g. Sparse-Heavy vs. Dense-Heavy).

---

## Known Limitations

- Sparse search does not perform synonym matching (e.g., query "hospital stay" won't match BM25 document "inpatient care" unless they share terms). Query expansion (HyDE) in **V6** will address this limitation.
- RRF weights are currently static, although configurable via `config.py`.

---

## Next Version Preview

The next version (**V6 — HyDE Query Expansion**) will introduce Hypothetical Document Embeddings. It will generate a hypothetical policy response using Gemini, embed it, and use that embedding to perform dense vector search, resolving synonym and vocabulary mismatches.
