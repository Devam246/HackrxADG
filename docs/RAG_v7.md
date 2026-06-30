# RAG v7 — Cross-Encoder Reranking

**Date:** 2026-06-30
**Scope:** Add a lightweight cross-encoder reranking step after initial retrieval so the generator receives the highest-signal chunks before context assembly.

---

## Goal

- Add a cross-encoder reranker that scores the top retrieval candidates against the original user query.
- Preserve graceful fallback behavior when the reranker model cannot be loaded or inference fails.
- Respect a configurable `rerank_top_n` limit throughout retrieval and generation.
- Keep the change lightweight and easy to operate without breaking the existing RRF-based pipeline.

---

## Previous Limitation (why this was needed)

The retrieval stack already fused BM25 and dense results using reciprocal rank fusion, but the final candidate list could still contain borderline false positives. A reranking pass over the candidate pool helps surface the most relevant chunks before the generator builds its context window.

---

## Concepts Introduced

1. **Cross-Encoder Reranking**: The reranker evaluates each candidate chunk against the original query and reorders the shortlist by relevance.
2. **Lazy Model Loading**: The reranker model is loaded only when reranking is needed, reducing import-time overhead.
3. **Graceful Fallback**: If loading or inference fails, the pipeline preserves the original candidate order and logs the issue.

---

## Design Decisions

- **Sparse/Dense Retrieval Remains the Same**: RRF still provides the initial candidate set.
- **Reranking Happens After Initial Retrieval**: The reranker only evaluates the top candidate pool, keeping runtime cost controlled.
- **Configurable Limit**: The `rerank_top_n` setting controls how many candidates are kept after reranking.
- **No Hard Failure on Model Issues**: Errors fall back to the original ordering without breaking the request.

---

## Changed Files

| File | Changes |
|---|---|
| `backend/services/retrieval/reranker.py` | Added `CrossEncoderReranker` with lazy loading and fallback behavior. |
| `backend/services/retrieval/vector_store.py` | Applied reranking in `advanced_universal_retrieval`. |
| `backend/services/generation/generator.py` | Respected `rerank_top_n` when selecting context chunks. |
| `backend/api/v1/routes_query.py` | Passed reranking settings through the query route. |
| `backend/models/domain.py` | Added `rerank_score` to `Chunk`. |
| `backend/tests/unit/test_reranker.py` | Added unit tests for success, load failure, and inference failure. |
| `docs/structure.md` | Updated the version and backend file inventory. |

---

## New Behavior

1. The retrieval pipeline builds an initial candidate pool.
2. `CrossEncoderReranker` scores those candidates against the original query.
3. The best `rerank_top_n` chunks are selected and passed into generation.
4. If model loading or scoring fails, the original order is preserved.

---

## Testing

- Verified with `pytest -q backend/tests`.
- Result: `10 passed in 94.04s`.

---

## Notes

- The reranker uses `BAAI/bge-reranker-v2-m3` by default when available.
- The implementation is intentionally defensive so retrieval remains robust in environments without the reranker dependencies installed.
