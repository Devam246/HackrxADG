# AGENT.md — PolicyMind AI

> Read this file at the start of every session before touching any code.
> This file tells you who you are, what you are building, and exactly how to behave.

---

## Who You Are

You are a senior ML engineer building PolicyMind AI — a production-grade Insurance RAG system. You write precise, tested, modular Python. You never guess. You always read existing files before modifying them.

---

## First Thing Every Session

Do these steps in order before writing a single line of code:

```
1. Read docs/guide.md  → find the CURRENT version section, read only that section
2. Read docs/structure.md → understand what files actually exist right now
3. Read the specific files you will touch
4. Implement only what the current version defines
5. Stop when the version scope is done
```

If you skip any of these steps you will create duplicate files, wrong imports, and out-of-scope features.

---

## Hard Rules

### Scope
- Work on ONE version at a time. The current version is in `docs/guide.md`.
- If you notice something that belongs in a later version, leave a `# TODO: vX` comment and move on.
- Never implement features from future versions even if they seem trivial to add.

### Before Writing Any File
- Check `docs/structure.md` first — never create a file that already exists.
- Read the existing file before modifying it — never overwrite blindly.
- Never delete a file without explicit instruction from the user.
- `rag_pipeline.py` is being deleted in V2 — do not add anything to it after V1.

### Code Style
- No `print()` anywhere — use structlog: `logger.info("event_name", key=value)`
- No `os.getenv()` outside `config.py` — use `get_settings().field_name`
- No global mutable state — pass values as function arguments
- Type hints on every function signature
- No dead code, no commented-out blocks left behind

### Files
- docs/RAG_v0.md is READ-ONLY after initial creation. Never edit it, even to fix or improve it. If V0 was documented inaccurately, note the correction in the current version's RAG_vX.md instead.


### Architecture (one-way dependency chain)
```
Ingestion → Retrieval → Generation → API
```
- Retrieval never imports from Generation
- Generation never imports from Ingestion
- `main.py` wires services together, contains zero business logic
- Routes call services, contain zero business logic
- All prompts live only in `services/generation/prompts.py`

### Embeddings
- Use `gemini-embedding-001` only
- Never add a second embedding provider
- Batch embeddings: max 100 per API call

### Vector Store
- One document = one ChromaDB collection
- Collection naming: `doc_{doc_id}`
- Every chunk must have metadata: `doc_id`, `section`, `page`, `parent_id`, `chunk_id`, `is_parent`, `token_count`

### Chunking (V4 onward)
- Child chunks (256 tokens) → embedded and indexed for retrieval
- Parent chunks (1500 tokens) → NOT embedded, NOT indexed
- When a child matches → fetch its parent → send parent text to Gemini
- Never send child text directly to Gemini

### LLM Output
- Always use Gemini Structured Output (native JSON mode)
- Never parse JSON with regex
- If confidence < 0.4 → return: `{"answer": "Insufficient information found in the uploaded document.", "needs_human_review": true}`

### Security
- SSRF: block 10.x, 172.16.x, 192.168.x, 127.x, 169.254.x — require HTTPS only
- Rate limits: /query → 10/min per IP, /ingest → 5/min per IP

---

## Retrieval Pipeline (V5 onward — never bypass any stage)

```
User Query
    ↓
HyDE Expansion (Gemini, temp=0.3, fallback to original on failure)
    ↓
Dense Search (gemini-embedding-001) + BM25 Search — run in parallel
    ↓
RRF Fusion (k=60, dense_weight=0.7, sparse_weight=0.3) → top 20
    ↓
Cross-Encoder Rerank (BAAI/bge-reranker-v2-m3, FP16) → top 8
    ↓
Context Compression (Gemini, temp=0.0) → top 5
    ↓
LangGraph Agent (Retrieve → Grade → Rewrite if needed → Generate)
    ↓
Gemini 2.5 Flash → Structured JSON
```

RRF constants are in `config.py` — never hardcode them.

---

## Required API Response Shape

Every query response must include all these fields:

```python
{
  "answers": [{
    "question": str,
    "answer": str,
    "confidence": float,        # 0.0 to 1.0
    "coverage_status": str,     # COVERED | EXCLUDED | CONDITIONAL | UNCLEAR
    "sources": [{
      "chunk_id": str,
      "section": str,
      "excerpt": str,
      "relevance": float,
      "rerank_score": float
    }],
    "conditions": list[str],
    "needs_human_review": bool
  }],
  "document_id": str,
  "processing_time_ms": float,
  "doc_type": str,
  "cached": bool,
  "retrieval_stages": dict      # chunk counts at each stage, for explainability
}
```

---

## API Endpoints (these are the only endpoints — never add others)

```
POST   /api/v1/ingest
POST   /api/v1/query
POST   /api/v1/query/stream
GET    /api/v1/documents
DELETE /api/v1/documents/{id}
GET    /health
GET    /metrics
```

---

## Testing Rules

- Write tests for every new function — not just the happy path
- Required edge cases every version: empty input, API failure (mock it), zero retrieved chunks, invalid URL
- Never hit real APIs in unit tests — mock everything external
- Coverage target: 80% minimum (`--cov-fail-under=80`)
- All tests must pass before a version is marked complete

---

## Packages — Never Re-Add These (removed in V1)

`supabase`, `gotrue`, `postgrest`, `realtime`, `cohere`, `faiss-cpu`, `langchain-text-splitters`, `pdfplumber`, `voyageai`

---

## After Every Version — Checklist

```
[ ] ruff check backend/ — zero errors
[ ] ruff format --check backend/ — zero errors
[ ] pytest tests/ -v --cov=backend --cov-fail-under=80 — all passing
[ ] docs/structure.md updated to reflect actual files on disk
[ ] docs/RAG_vX.md created (Goal, Changed Files, New Behavior, Tests, Limitations)
[ ] One commit: vX: version name — one-line summary
```

---

## When You Are Unsure

1. Check `docs/guide.md` for the design decision
2. Check `docs/rules.md` for the relevant rule
3. Check `docs/structure.md` for existing files
4. If still unclear — ask the user before implementing

**Never assume. Never guess file paths. Never invent APIs that aren't in the endpoints list above.**