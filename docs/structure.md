# PolicyMind AI - Project Structure

**Current Version:** V2 architecture refactor
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
├── config.py                    # Pydantic Settings added in V1
├── main.py                      # FastAPI app (app factory only)
├── pyproject.toml               # Ruff + pytest config
├── requirements.txt             # Runtime dependencies cleaned in V1
├── requirements-dev.txt         # Dev/test dependencies added in V1
├── api/
│   └── v1/
│       ├── deps.py              # Auth dependencies
│       ├── routes_health.py     # System health and ping routes
│       └── routes_query.py      # HackRx run query routes
├── models/
│   ├── domain.py                # Domain-level dataclasses (Chunk, Document)
│   └── schemas.py               # Pydantic request/response schemas
├── services/
│   ├── ingestion/
│   │   ├── classifier.py        # Document type classifier
│   │   ├── downloader.py        # Safe downloader
│   │   ├── parsers.py           # Text parsers (PDF, DOCX, EML)
│   │   └── chunker.py           # Text cleaning, sectioning, and chunking
│   ├── retrieval/
│   │   ├── embedder.py          # Voyage embedder shim (V3 placeholder) and PCA
│   │   └── vector_store.py      # In-memory indexes, candidate filters, scores
│   └── generation/
│       ├── generator.py         # LLM batch generator and orchestration
│       ├── postprocessor.py     # JSON parser and confidence evaluation
│       └── prompts.py           # Batch and domain-specific prompt builders
├── utils/
│   ├── cache.py                 # LRU Document cache
│   ├── logging.py               # structlog configuration and spaCy model loading
│   └── security.py              # Stub SSRF guard
└── tests/
    ├── conftest.py              # Test configuration and path setup
    └── test_smoke.py            # Smoke test for POST /hackrx/run
```

Generated local directories that may exist but are not source:

```text
backend/
├── .pytest_cache/
├── .ruff_cache/
├── __pycache__/
├── cache/
└── venv/
```

---

## Not Yet Created

These are future-version targets and do not exist in V2:

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
└── test_smoke.py                # POST /hackrx/run smoke test with monkeypatched services
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
├── rules.md
└── structure.md
```

---

## Key File Descriptions

| File | Purpose | V2 status |
|---|---|---|
| `backend/main.py` | FastAPI entry point and middleware | Converted to application factory only |
| `backend/api/v1/` | FastAPI routes | Separated into system routes and query routes |
| `backend/services/` | Modularized business logic | Monolith `rag_pipeline.py` split into ingestion, retrieval, and generation packages |
| `backend/models/` | Type definitions and schema structures | Separated Pydantic schemas and placeholder domain dataclasses |
| `backend/utils/` | Shared utilities | Logging, cache, and a stub security file created |
| `backend/tests/test_smoke.py` | API smoke test | Patched to mock modules at new modular lookup paths |
