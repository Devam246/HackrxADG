# PolicyMind AI - Project Structure

**Current Version:** V6 HyDE Query Expansion
**Last Updated:** 2026-06-30

This file reflects what exists on disk right now.

---

## Root

```text
policymind-ai/
├── AGENT.md
├── Procfile
├── PROJECT_ANALYSIS.md
├── README.md
├── start.sh
├── .gitignore
├── backend/
└── docs/
```

Notes:
- The backend is modularly structured under `backend/`.
- There is no frontend yet.
- Archived planning files live under `docs/ARCHIVE/`.

---

## Backend

```text
backend/
├── .env                         # Local secrets only; do not commit
├── config.py                    # Pydantic Settings
├── main.py                      # FastAPI app (app factory only)
├── pyproject.toml               # Ruff + pytest config
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Dev/test dependencies added in V1
├── api/
│   └── v1/
│       ├── deps.py              # Auth dependencies
│       ├── routes_health.py     # System health and ping routes
│       └── routes_query.py      # HackRx run query routes
├── models/
│   ├── domain.py                # Domain-level dataclasses (Chunk with dict-shimming, Document)
│   └── schemas.py               # Pydantic request/response schemas
├── services/
│   ├── ingestion/
│   │   ├── classifier.py        # Document type classifier
│   │   ├── downloader.py        # Safe downloader
│   │   ├── parsers.py           # Text parsers (PDF, DOCX, EML)
│   │   └── chunker.py           # ParentChildChunker & parent store mapping
│   ├── retrieval/
│   │   ├── embedder.py          # Gemini embedding wrapper (embed_text, task_type added in V6)
│   │   ├── vector_store.py      # ChromaDB client abstraction (RRF retrieval)
│   │   ├── bm25_index.py        # BM25Okapi sparse index and persistence
│   │   ├── hybrid_search.py     # Cosine similarity dense search & RRF fusion logic
│   │   └── hyde.py              # HyDE query expansion (added in V6)
│   └── generation/
│       ├── generator.py         # LLM batch generator and orchestration (HyDE integrated in V6)
│       ├── postprocessor.py     # JSON parser and confidence evaluation
│       └── prompts.py           # Batch, universal, and HYDE prompt builders
├── utils/
│   ├── cache.py                 # LRU Document cache
│   ├── logging.py               # structlog configuration
│   └── security.py              # Stub SSRF guard
└── tests/
    ├── conftest.py              # Test configuration and path setup
    ├── test_smoke.py            # Smoke test for POST /hackrx/run
    └── unit/
        ├── test_chunker.py      # Chunker unit tests
        ├── test_hybrid_search.py # RRF fusion unit tests
        └── test_hyde.py         # HyDE unit tests (added in V6)
```

Generated local directories that may exist but are not source:

```text
backend/
├── .pytest_cache/
├── .ruff_cache/
├── __pycache__/
├── cache/                       # parents_{doc_id}.json & bm25_{doc_id}.pkl stores
├── chroma_db/                   # Persistent ChromaDB data (git-ignored)
└── venv/
```

---

## Not Yet Created

These are future-version targets and do not exist in V6:

```text
backend/agent/                   # LangGraph Agentic graph (V9 scope)
frontend/
.github/workflows/
```

---

## Tests

```text
backend/tests/
├── conftest.py                  # Test env defaults and import path setup
├── test_smoke.py                # POST /hackrx/run smoke test with monkeypatched services
└── unit/
    ├── test_chunker.py          # Chunker unit tests verifying hierarchy and token sizes
    ├── test_hybrid_search.py    # BM25 + Dense + RRF fusion verification unit tests
    └── test_hyde.py             # HyDE expansion and fallback unit tests
```

---

## Docs

```text
docs/
├── ARCHIVE/
│   ├── final.md
│   ├── improvements2.md
│   └── imrovements.md
├── benchmarks.md
├── guide.md
├── RAG_v0.md
├── RAG_v1.md
├── RAG_v2.md
├── RAG_v3.md
├── RAG_v4.md
├── RAG_v5.md
├── RAG_v6.md
├── rules.md
└── structure.md
```

---

## Key File Descriptions

| File | Purpose | V6 status |
|---|---|---|
| `backend/services/retrieval/hyde.py` | HyDE expansion generator | Generates hypothetical policy excerpts with silent fallback logging |
| `backend/services/generation/prompts.py` | Prompt templates | Houses the standard `HYDE_PROMPT` to keep prompt logic separated |
| `backend/services/retrieval/embedder.py` | Embedding generation | Renamed `embed_voyage` to `embed_text` and added query vs document task types |
| `backend/tests/unit/test_hyde.py` | HyDE unit testing | Asserts successful generation and fallback behaviors |
