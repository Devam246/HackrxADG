# RAG v1 - Stable Foundation

**Date:** 2026-06-30
**Scope:** Runtime stabilization only. V2 modular architecture was not started.

---

## Goal

Make the existing monolithic FastAPI RAG backend start reliably, remove the known dangerous runtime patterns, clean the dependency surface, and add a smoke test/tooling baseline.

---

## What Was Actually Found

- The import crash was real: `main.py` imported `build_enhanced_inverted_index`, but `rag_pipeline.py` only defined `build_inverted_index`.
- There were two duplicate `advanced_universal_retrieval` blocks, not one. The final duplicate overrode the working implementation and contained invalid placeholder code.
- `faiss-cpu` was banned and removed from requirements, but `rag_pipeline.py` still imported and used `faiss`. This would break a clean install without `faiss-cpu`.
- Comments and log strings had encoding corruption/mojibake. This made targeted patching unreliable and required a full-file replacement for `main.py`.
- The spaCy model may be missing in local environments, causing startup failure before any request is handled.
- The smoke test had a bearer token mismatch because `main.EXPECTED_BEARER_TOKEN` was initialized at import time before the test patched request headers.
- `SettingsConfigDict(fields=...)` was invalid/deprecated for pydantic-settings v2, produced a warning, and did nothing.

---

## Changed Files

| File | Changes |
|---|---|
| `backend/main.py` | Replaced old import with `build_inverted_index`; rewrote app file with structured logging and central settings access while keeping the existing `/hackrx/run` endpoint. |
| `backend/rag_pipeline.py` | Removed duplicate retrieval overrides; added UUID temp downloads; removed global doc type state; replaced `print()` with structlog calls; replaced FAISS usage with a NumPy L2 shim; added spaCy fallback. |
| `backend/config.py` | Added pydantic-settings config with `Field(validation_alias=...)`; removed invalid `SettingsConfigDict(fields=...)`; removed leftover `voyage_api_key`. |
| `backend/requirements.txt` | Removed banned/unused V1 packages including `supabase`, `cohere`, `faiss-cpu`, `langchain-core`, `langchain-text-splitters`, `pdfplumber`, and `voyageai`; added `structlog` and `pydantic-settings`. |
| `backend/requirements-dev.txt` | Added pytest and ruff tooling dependencies. |
| `backend/pyproject.toml` | Added ruff and pytest config for the backend-flat layout; excluded generated venv/cache dirs; silenced unrelated third-party SWIG warnings. |
| `backend/tests/conftest.py` | Added test environment defaults and import path setup. |
| `backend/tests/test_smoke.py` | Added POST `/hackrx/run` smoke test with monkeypatched services and bearer token. |
| `docs/structure.md` | Updated to actual backend-flat V1 layout. |
| `docs/RAG_v1.md` | Added this V1 completion record. |

---

## New Behavior

- Importing the FastAPI app no longer fails due to the old inverted-index function name.
- Retrieval no longer resolves to the final broken duplicate implementation.
- Concurrent downloads use unique temporary filenames: `rag_{doc_id}_{uuid}{ext}`.
- Document type is request-local inside `load_or_create_chunks`.
- Clean installs no longer require `faiss-cpu`; the old index helper now returns a small NumPy-backed object with a compatible `.search()` method.
- Clean installs no longer require `voyageai`; the old `embed_voyage` helper name is kept as a local deterministic embedding shim for V1 monolith compatibility.
- Missing `en_core_web_sm` no longer prevents startup; the pipeline falls back to `spacy.blank("en")`.
- Configuration is centralized in `backend/config.py` and uses valid pydantic-settings v2 aliases.
- Logs use `structlog` instead of direct `print()` calls.

---

## Verification

Run from `backend/`:

```text
.\venv\Scripts\python.exe -c "import main; print('import ok')"
import ok

.\venv\Scripts\python.exe -m pytest -q
1 passed in 4.29s

.\venv\Scripts\python.exe -m ruff check .
All checks passed!

.\venv\Scripts\python.exe -m ruff format --check .
5 files already formatted
```

---

## Limitations

- The backend is still a monolith. No V2 service/module split was started.
- The public endpoint remains the existing `/hackrx/run`; `/api/v1/*` routes belong to later architecture work.
- The smoke test uses monkeypatched services and does not hit real document downloads, embeddings, or Gemini.
- The V1 local embedding shim is only a stabilization fallback; Gemini embedding migration belongs to a later roadmap version.
