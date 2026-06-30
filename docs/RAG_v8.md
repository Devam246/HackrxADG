# RAG v8 — Context Compression

**Date:** 2026-06-30
**Scope:** Implement an LLM-based Context Compressor that extracts only relevant sentences from retrieved parent chunks, reducing prompt token size and preventing hallucination before context formatting.

---

## Goal

- Implement a context compression step after cross-encoder reranking.
- Reduce overall token size of supporting clauses passed to the generator by 40–60%.
- Filter out completely irrelevant chunks by discarding those that return `"NO_RELEVANT_CONTENT"`.
- Implement robust fallback logic that keeps at least 2 chunks (falling back to original uncompressed parent chunks) to preserve context.
- Log token counts and compute average token reduction percentage request-wide.

---

## Previous Limitation (why this was needed)

After retrieval and reranking, the top chunks sent as context could still contain thousands of tokens of irrelevant text surrounding the actual target clauses. Sending large chunks increases LLM token costs, limits model prompt capacity, and risks introducing hallucinations when the LLM reads noise.

---

## Concepts Introduced

1. **Deterministic Context Compression**: An LLM is prompted at `temperature=0.0` to extract only the sentences directly addressing the user query.
2. **Parallel Compression Queries**: Chunks are processed in parallel using a `ThreadPoolExecutor` to minimize latency overhead.
3. **Fallback Guard**: If fewer than 2 compressed chunks survive (due to extreme filtering), the pipeline reverts to the top 2 original parent chunks.
4. **Token Reduction Tracking**: Measures and logs the exact token count difference using `tiktoken` (`cl100k_base`).

---

## Changed Files

| File | Changes |
|---|---|
| `backend/services/retrieval/compressor.py` | Added context compressor with parallel executor, token counting, and fallback logic. |
| `backend/services/generation/generator.py` | Integrated `compress_chunks` in `_process_single_query` and aggregated stats in `handle_queries`. |
| `backend/tests/unit/test_compressor.py` | Added unit tests verifying success, filtering, fallback, and API exception handling. |
| `docs/structure.md` | Updated the project structure to V8 and inventoried new files. |

---

## Testing

- Unit tests added to verify all compression logic paths.
- Verified using `pytest backend/tests`.
