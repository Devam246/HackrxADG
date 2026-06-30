# PolicyMind AI - Project Structure

**Current Version:** V1 stable foundation
**Last Updated:** 2026-06-30

This file reflects what exists on disk right now. It intentionally does not list the V2 modular service folders as existing, because V2 has not started.

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
- The backend is flat inside `backend/`.
- There is no frontend yet.
- There is no root-level `pyproject.toml`, `requirements.txt`, `main.py`, or `rag_pipeline.py`; those files are under `backend/`.
- Archived planning files live under `docs/ARCHIVE/`.

---

## Backend

```text
backend/
├── .env                         # Local secrets only; do not commit
├── config.py                    # Pydantic Settings added in V1
├── main.py                      # FastAPI app; still monolithic until V2
├── pyproject.toml               # Ruff + pytest config
├── rag_pipeline.py              # Monolithic RAG pipeline; V2 will split this
├── requirements.txt             # Runtime dependencies cleaned in V1
├── requirements-dev.txt         # Dev/test dependencies added in V1
└── tests/
    ├── conftest.py
    └── test_smoke.py
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

These are future-version targets and do not exist in V1:

```text
backend/api/
backend/services/
backend/models/
backend/utils/
backend/agent/
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
├── rules.md
└── structure.md
```

---

## Key File Descriptions

| File | Purpose | V1 status |
|---|---|---|
| `backend/main.py` | FastAPI entry point and hackathon endpoint | Fixed import crash, structured logging in place |
| `backend/rag_pipeline.py` | Monolithic download, parse, chunk, retrieve, generate pipeline | Duplicate retrieval overrides removed; FAISS import replaced with NumPy shim |
| `backend/config.py` | Central settings object | Uses `Field(validation_alias=...)`; no deprecated `SettingsConfigDict(fields=...)` |
| `backend/requirements.txt` | Runtime dependencies | Banned V1 dependencies removed |
| `backend/requirements-dev.txt` | Dev/test dependencies | Added pytest, pytest-cov, pytest-mock, ruff |
| `backend/pyproject.toml` | Tooling config | Ruff and pytest configured for backend-flat layout |
| `backend/tests/test_smoke.py` | API smoke test | Monkeypatches service calls and bearer token |

---

## V2 Reminder

Do not create `api/`, `services/`, `models/`, `utils/`, or `agent/` during V1. Those folders belong to the V2 architecture refactor.
