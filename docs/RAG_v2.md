# RAG v2 - Modular Architecture Refactor

**Date:** 2026-06-30
**Scope:** Reorganizing the monolithic `rag_pipeline.py` and `main.py` routes into modular folders. No functional changes, no new features.

---

## Goal

Split the monolithic `rag_pipeline.py` (approx. 100 KB) and route handlers in `main.py` into modular packages under `backend/` to improve code organization, scalability, and maintainability. Ensure all tests continue to pass cleanly.

---

## Modular Layout mapping (What Moved Where)

| Original File / Code Block | New File | Description |
|---|---|---|
| `main.py` timing / GZip / CORS middleware, lifespan | `backend/main.py` | Remains as application factory and middleware configuration only. |
| `main.py` `verify_token` | `backend/api/v1/deps.py` | Authentication dependency. |
| `main.py` system endpoints (`/`, `/health`, `/ping`) | `backend/api/v1/routes_health.py` | Health and diagnostic routes. |
| `main.py` query endpoints (`/hackrx/run`, `/cache-stats`) | `backend/api/v1/routes_query.py` | Core query logic routes. |
| `main.py` `DocumentCache` class & instance | `backend/utils/cache.py` | In-memory cache manager. |
| `main.py` `structlog` configuration | `backend/utils/logging.py` | Shared logging configuration. |
| `main.py` schemas (`HackathonRequest`, etc.) | `backend/models/schemas.py` | Pydantic request/response validation schemas. |
| (Placeholder definitions) | `backend/models/domain.py` | `Chunk` and `Document` domain dataclasses (left unused until V4). |
| (Placeholder definition) | `backend/utils/security.py` | Empty `SSRFGuard` placeholder stub (V11 scope). |
| `rag_pipeline.py` `download_file` | `backend/services/ingestion/downloader.py` | Safe remote file downloader. |
| `rag_pipeline.py` `extract_text_from_pdf` / `docx` / `eml` | `backend/services/ingestion/parsers.py` | Document loaders and parsers. |
| `rag_pipeline.py` `DocumentClassifier` | `backend/services/ingestion/classifier.py` | Content classifier. |
| `rag_pipeline.py` text cleaning, structured sections, and chunking | `backend/services/ingestion/chunker.py` | Text cleaner, chunkers, and orchestrator `load_or_create_chunks`. |
| `rag_pipeline.py` `embed_voyage`, PCA dimensionality reduction | `backend/services/retrieval/embedder.py` | Embedding shims and PCA. `embed_voyage` has TODO: V3 comment. |
| `rag_pipeline.py` local memory index shims, query parsing, candidate filtering, scores | `backend/services/retrieval/vector_store.py` | Vector store indexing shims, candidate retrieval, scoring functions. |
| `rag_pipeline.py` LLM batch generation, query preprocessors, query handlers | `backend/services/generation/generator.py` | LLM invocation client and batch query dispatcher. |
| `rag_pipeline.py` JSON parser, confidence evaluation | `backend/services/generation/postprocessor.py` | Response parser and confidence calculator. |
| `rag_pipeline.py` prompt builders | `backend/services/generation/prompts.py` | Prompt templates. |

---

## Confirming No Behavior Changed

- **Routing:** `/hackrx/run`, `/health`, `/ping`, `/cache-stats`, and `/` behave exactly as before.
- **Middleware & Lifespan:** GZip, CORS, request timing middleware, and lifespan app setup behave identically.
- **Shims & Fallbacks:** Kept Voyage AI dummy local hash embedding shim (labeled with `# TODO: V3 — replace with real gemini-embedding-001`) and blank spaCy loader fallback.
- **Cache:** LRU cache continues to cache document chunk structures and embeddings transparently.

---

## Verification

Running tests and code verification from `backend/`:

```text
venv/Scripts/python -m pytest
============================== 1 passed in 5.61s ==============================

venv/Scripts/python -m ruff check .
All checks passed!

venv/Scripts/python -m ruff format --check .
22 files already formatted
```

---

## Limitations

- The vector index and embedding remain local shims (Gemini embeddings and ChromaDB vector store migration is V3 scope).
- Security policies, SSRFGuard, and Rate Limiting remain unimplemented stubs (V11 scope).
