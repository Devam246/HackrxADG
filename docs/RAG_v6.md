# RAG v6 — HyDE Query Expansion

**Date:** 2026-06-30
**Scope:** Implementing Hypothetical Document Embeddings (HyDE) for query expansion to resolve vocabulary and phrasing mismatches during semantic retrieval.

---

## Goal

- Generate a hypothetical policy excerpt that answers the user's question, using the configured Gemini model (`gemini-2.5-flash`).
- Use the embedding of the generated hypothetical excerpt for the dense vector search, projecting the query into the document clause vector space.
- Keep using the original query text for BM25 keyword matching, ensuring that both semantic alignment and exact keyword signals remain strong.
- Silently log any generation errors and fall back to embedding the original query text.
- Rename the legacy Voyage embedding helper (`embed_voyage`) to `embed_text` and align it with Gemini's task-type specifications.

---

## Previous Limitation (why this was needed)

Standard semantic retrieval relies on matching the user's question embedding directly to document chunk embeddings. If a user asks a question using colloquial phrasing (e.g. `"What is the phone number for Filipino language help?"`), it might fail to rank highly against a formal policy clause that uses different terminology (e.g. `"[Tagalog (Tagalog): ...]"`), since the query and document vectors reside in slightly different semantic sub-spaces. 

---

## Concepts Introduced

1. **Hypothetical Document Embeddings (HyDE)**: Instead of embedding the query directly, a generative model writes a dummy/hypothetical response page (e.g., an excerpt in formal insurance language). Embedding this dummy document allows document-to-document vector matching, which has significantly higher semantic alignment.
2. **Gemini Task Type Differentiation**: Projects query-time vectors with `task_type="retrieval_query"` and chunk-indexing vectors with `task_type="retrieval_document"`, complying with Gemini embedding model standards.

---

## Design Decisions

- **Embedding Rename**: Renamed the legacy Voyage naming wrapper `embed_voyage` to a model-agnostic `embed_text`.
- **Query vs Document Embeddings**: Integrated the `task_type` parameter into `embed_text` so that query-time embeddings project into the query space and ingest-time embeddings project into the document space.
- **Isolated Prompt Logic**: Placed `HYDE_PROMPT` inside `backend/services/generation/prompts.py` in compliance with **Rule 6** (no inline prompts in retrieval services).
- **Graceful Fallback**: Implemented a robust try-except wrapper. If the Gemini API limits or fails, the exception is logged with full traceback details via `logger.exception("hyde_generation_failed", ...)` and the pipeline silently falls back to encoding the original query text, preventing user-facing crashes.

---

## Trade-offs

- **LLM Cost & Latency**: Query expansion adds one Gemini API call before retrieval. We minimized latency by configuring `temperature = 0.3` and capping `max_output_tokens = 200`. On the free tier, this can occasionally trigger rate limits (HTTP 429), which are gracefully handled by the silent fallback.

---

## Changed Files

| File | Changes |
|---|---|
| `backend/services/retrieval/embedder.py` | Renamed `embed_voyage` to `embed_text` and added the `task_type` parameter. |
| `backend/services/ingestion/chunker.py` | Updated all legacy `embed_voyage` imports and calls to `embed_text` with `task_type="retrieval_document"`. |
| `backend/services/generation/prompts.py` | Added the standard `HYDE_PROMPT` template. |
| `backend/services/retrieval/hyde.py` | **[NEW]** Created the HyDE service, implementing `generate_hypothetical_excerpt` with Gemini and fallback. |
| `backend/services/generation/generator.py` | Integrated HyDE expansion in `_process_single_query` before generating the query embedding. |
| `backend/tests/unit/test_hyde.py` | **[NEW]** Created unit tests for HyDE expansion success and fallback. |
| `docs/structure.md` | Updated structure tree to reflect V6. |

---

## New Behavior

1. User query enters `_process_single_query`.
2. HyDE generates a hypothetical policy excerpt answering the query (or falls back to the original query on failure).
3. The hypothetical excerpt is embedded using `task_type="retrieval_query"`.
4. `advanced_universal_retrieval` runs `search_dense` using the HyDE embedding and `search_bm25` using the original query string.
5. RRF fuses the lists and maps child chunks to parents.

---

## Testing Added

1. **Unit Tests (`test_hyde.py`)**:
   - `test_generate_hypothetical_excerpt_success`: Asserts that the prompt is formatted correctly and returns the generated text.
   - `test_generate_hypothetical_excerpt_failure_fallback`: Mocks a Gemini API failure to verify that the original query is returned and the error is logged.
2. **Verification Script (`verify_v6_e2e.py`)**:
   - Ingests the CMS PDF.
   - Verifies the user-requested query `"Is rehab covered for alcohol addiction?"` against Chunk 3 (verifies it ranks in the top 3 under HyDE).
   - Verifies the vocabulary mismatch query `"Can I set aside pre-tax dollars from my paycheck to pay for medical costs?"` against Chunk 10 (describing HSAs/FSAs/HRAs).
   - Asserts that HyDE improves the dense search rank of Chunk 10 (elevating it from Rank 6 to Rank 3) and correctly returns it in the top 3.

---

## Benchmark Results (V3 onward)

Baseline RAGAS/DeepEval benchmarks are scheduled for **V10**.
Manual verification on the CMS template PDF using live non-mocked API generation confirms:
- Case 1: Original query `"Is rehab covered for alcohol addiction?"` ranks the rehab/substance services chunk (Chunk 3) as **1st** (score: 0.6400). HyDE maintains its **1st** rank with an elevated similarity score of **0.7009**.
- Case 2: Original query `"Can I set aside pre-tax dollars from my paycheck to pay for medical costs?"` ranks the tax-advantaged savings account chunk (Chunk 10) as **6th** (score: 0.6173). HyDE-expanded query elevates it to **3rd** (score: 0.6202).
- RRF successfully fuses this with BM25, ranking Chunk 10 at **3rd** place in the final combined list.

---

## Known Limitations

- High query rate on the free tier can trigger Gemini rate limit exceptions, causing retrieval to fall back to standard semantic search.

---

## Next Version Preview

The next version (**V7 — rerank-lite Rerank**) will introduce a lightweight cross-encoder reranking step using Gemini's classification layer to evaluate the top candidates returned by RRF, filtering out false positives before context construction.
