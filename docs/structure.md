# PolicyMind AI - Project Structure

**Current Version:** V10 Evaluation Suite
**Last Updated:** 2026-06-30

This file reflects what exists on disk right now.

---

## Root

```text
hackrxadg/
├── AGENT.md
├── Procfile
├── PROJECT_ANALYSIS.md
├── README.md
├── start.sh
├── .gitignore
├── .github/
│   └── workflows/
│       └── rag_eval.yml         # RAG eval CI job (runs on main branch)
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
├── config.py                    # Pydantic settings and reranking defaults
├── main.py                      # FastAPI app factory
├── pyproject.toml               # Ruff + pytest config
├── requirements.txt             # Runtime dependencies including reranker packages
├── requirements-dev.txt         # Dev/test dependencies
├── api/
│   └── v1/
│       ├── deps.py              # Auth dependencies
│       ├── routes_health.py     # System health routes
│       └── routes_query.py      # HackRx query routes with rerank_top_n support
├── models/
│   ├── domain.py                # Chunk and Document dataclasses (includes rerank_score)
│   └── schemas.py               # Pydantic request/response schemas
├── services/
│   ├── ingestion/
│   │   ├── classifier.py        # Document type classifier
│   │   ├── downloader.py        # Safe downloader
│   │   ├── parsers.py           # Text parsers (PDF, DOCX, EML)
│   │   └── chunker.py           # ParentChildChunker and parent-store mapping
│   ├── retrieval/
│   │   ├── embedder.py          # Gemini embedding wrapper
│   │   ├── vector_store.py      # ChromaDB client abstraction with reranking
│   │   ├── bm25_index.py        # BM25Okapi sparse index and persistence
│   │   ├── hybrid_search.py     # Dense search + RRF fusion logic
│   │   ├── hyde.py              # HyDE query expansion (V6)
│   │   ├── reranker.py          # CrossEncoderReranker with lazy loading and fallback
│   │   └── compressor.py        # Contextual compressor using Gemini (V8)
│   └── generation/
│       ├── generator.py         # LLM wrapper delegating to agentic pipeline
│       ├── postprocessor.py     # JSON parser and confidence evaluation
│       └── prompts.py           # Batch, universal, and HyDE prompt builders
├── agent/
│   ├── __init__.py
│   └── rag_graph.py         # LangGraph Retrieve→Grade→Rewrite→Generate (V9)
├── utils/
│   ├── cache.py                 # LRU document cache
│   ├── logging.py               # structlog configuration
│   └── security.py              # SSRF guard stub
└── tests/
    ├── conftest.py              # Test env defaults and import path setup
    ├── test_smoke.py            # Smoke test for POST /hackrx/run
    ├── unit/
    │   ├── test_chunker.py
    │   ├── test_hybrid_search.py
    │   ├── test_hyde.py
    │   ├── test_reranker.py
    │   ├── test_compressor.py
    │   └── test_agent.py
    └── rag_eval/
        ├── __init__.py
        ├── insurance_benchmark.json # 15-question Q&A dataset
        └── test_rag_metrics.py      # DeepEval + RAGAS (skipped without API key)
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

These are future-version targets and do not exist in V11:

```text
frontend/
```

---

## Tests

```text
backend/tests/
├── conftest.py                  # Test env defaults and import path setup
├── test_smoke.py                # Smoke test for POST /hackrx/run
├── unit/
│   ├── test_chunker.py          # Chunker unit tests verifying hierarchy and token sizes
│   ├── test_hybrid_search.py    # BM25 + dense + RRF fusion verification unit tests
│   ├── test_hyde.py             # HyDE expansion and fallback unit tests
│   └── test_reranker.py        # Cross-encoder reranking unit tests
└── rag_eval/
    ├── insurance_benchmark.json # 15-question Q&A dataset
    └── test_rag_metrics.py      # DeepEval + RAGAS (skipped without API key)
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
├── RAG_v7.md
├── RAG_v8.md
├── RAG_v9.md
├── RAG_v10.md
├── RAG_v11.md
├── rules.md
└── structure.md
```

---

## Key File Descriptions

| File | Purpose | V7 status |
|---|---|---|
| `backend/services/retrieval/reranker.py` | Cross-encoder reranking | Lazy-loads the reranker model and falls back gracefully |
| `backend/services/retrieval/vector_store.py` | Retrieval orchestration | Applies reranking after the initial candidate pool is built |
| `backend/services/retrieval/compressor.py` | Context compression | Extracts relevant sentences using Gemini and filters out irrelevant chunks |
| `backend/services/generation/generator.py` | Response generation | Delegates to agentic pipeline; aggregates answers and `needs_human_review` flags |
| `backend/agent/rag_graph.py` | Agentic RAG graph | LangGraph Retrieve→Grade→Rewrite→Generate state machine |
| `backend/tests/unit/test_agent.py` | Agent graph unit testing | Covers direct path, rewrite cycle, and exhausted-retry human-review paths |
| `backend/tests/rag_eval/insurance_benchmark.json` | Benchmark dataset | 15 insurance Q&A pairs for metric evaluation |
| `backend/tests/rag_eval/test_rag_metrics.py` | RAG evaluation suite | DeepEval (4 metrics) + RAGAS batch scoring; auto-skipped without API key |
| `.github/workflows/rag_eval.yml` | CI job | Runs RAG eval suite on pushes to main branch |
| `backend/tests/unit/test_compressor.py` | Compressor unit testing | Covers success, filtering, fallback, and API error paths |
| `backend/tests/unit/test_agent.py` | Agent graph unit testing | Covers direct path, rewrite cycle, and exhausted-retry human-review paths |
