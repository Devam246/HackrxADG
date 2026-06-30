# PolicyMind AI — Development Rules

---

## Philosophy

This project is a **personal flagship project**, not a hackathon submission.

The primary objective is **retrieval accuracy and answer faithfulness**, followed by production-quality engineering, then UI polish.

Every architectural decision must answer one question:

> **Does this improve retrieval quality, faithfulness, explainability, or maintainability?**

If not, it belongs in a later version.

---

## Rule 1: Source of Truth

`guide.md` is the single source of truth for architecture, technology choices, version scope, and design decisions.

- Read the current version in `guide.md` before writing any code
- Understand the existing implementation before changing it
- Preserve all previously working functionality unless the current version explicitly replaces it
- If a significantly better technique exists (library changes, newer production approach, API deprecation, measurable improvement), update both the implementation **and** `guide.md`
- The guide and implementation must **never** diverge

---

## Rule 2: Version Workflow

Every version follows this exact order. No exceptions.

```
1. Read current version in guide.md
2. Inspect current implementation (read relevant files)
3. Plan the smallest coherent change
4. Implement only the current version scope
5. Write or update tests
6. Run tests — all must pass
7. Update structure.md
8. Create docs/RAG_vX.md
9. Commit (one commit per completed version)
10. Begin next version
```

**Never** work on multiple versions simultaneously.  
**Never** skip versions.  
**Never** merge version scopes.  
**Never** commit failing tests.

---

## Rule 3: Accuracy First

Accuracy is the highest priority. Optimize for:

- Retrieval Precision
- Retrieval Recall
- Faithfulness (no hallucinations)
- Context Precision
- Context Recall
- Citation Quality

**Never** add infrastructure "because it is production" if it does not improve one of:
- Answer quality
- Retrieval quality
- Explainability
- Reliability

If a feature does not improve any of these, it belongs in a later version.

---

## Rule 4: The Retrieval Pipeline Is Fixed

After V5, the retrieval pipeline is the backbone. Every later version builds on it. No version bypasses any stage unless it explicitly replaces that stage.

```
User Query
  ↓
HyDE Query Expansion          (V6+)
  ↓
Dense Retrieval (Gemini)      (V3+)
+
BM25 Retrieval                (V5+)
  ↓
Reciprocal Rank Fusion        (V5+)
  ↓
Cross-Encoder Reranker        (V7+)
  ↓
Context Compression           (V8+)
  ↓
LangGraph Agent               (V9+)
  ↓
Gemini 2.5 Flash
  ↓
Structured JSON Response
```

---

## Rule 5: Embeddings

The project standard embedding model is:

```
gemini-embedding-001
```

Reasons:
- No additional API provider (same Gemini ecosystem)
- Simpler key management
- Free quota for development
- Strong quality on domain text

**Never** introduce a second embedding provider unless there is measurable benchmark improvement. If switching, update `guide.md` first.

---

## Rule 6: LLM

Generation model:

```
gemini-2.5-flash
```

- Use Gemini Structured Output (native JSON mode) wherever possible
- **Never** rely on regex parsing for production JSON responses
- All prompts live in `services/generation/prompts.py` — never inline prompts in route handlers or agent code

---

## Rule 7: Chunking

The project uses Parent–Child Chunking (from V4 onward).

Rules:
- **Child chunks** (256 tokens) are embedded and indexed — used for retrieval
- **Parent chunks** (1500 tokens) are NOT embedded and NOT indexed — used as LLM context
- When a child chunk matches, fetch its parent and send the parent text to Gemini
- **Never** send child chunks directly to the LLM
- Prefer semantic boundaries (headings, clause numbers, sections, tables) over arbitrary character limits
- `overlap = 32 tokens` on child chunks

---

## Rule 8: Vector Database

```
Development:   ChromaDB  (local, zero-ops)
Production:    Qdrant    (cloud, managed)
```

Rules:
- One document = one collection. Never mix documents inside one collection.
- Collection naming: `doc_{doc_id}`
- Every chunk must contain metadata: `doc_id`, `section`, `page`, `parent_id`, `chunk_id`, `is_parent`, `token_count`
- ChromaDB HNSW config: `{"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:M": 32}`
- The vector store is accessed only through `services/retrieval/vector_store.py` — never directly from routes or agents

---

## Rule 9: Hybrid Retrieval (V5+)

After V5, never rely solely on cosine similarity.

Every retrieval must follow:

```
Dense Search (Gemini embeddings)
+
BM25 Sparse Search
  ↓
Reciprocal Rank Fusion (k=60, dense_weight=0.7, sparse_weight=0.3)
  ↓
Top 20 candidates
```

RRF constants:
- `k = 60` (standard smoothing from original RRF paper)
- `dense_weight = 0.7`
- `sparse_weight = 0.3`

These are tunable via `config.py` — never hardcode them.

---

## Rule 10: Agent Rules (V9+)

Generation never starts immediately after retrieval. The agent grades retrieval quality first.

```
Retrieve
  ↓
Grade (does the context actually answer the question?)
  ↓
Sufficient? ──Yes──→ Generate
     │
    No
     ↓
Rewrite Query
  ↓
Retrieve Again (max 2 retries)
  ↓
Generate (with needs_human_review: true if still insufficient)
```

Rules:
- Maximum 2 rewrite cycles per query — no infinite loops
- If after 2 retries context is still insufficient, generate with `needs_human_review: true`
- Confidence below 0.4 always triggers `needs_human_review: true`
- **Never** allow Gemini to answer from general knowledge — only from retrieved context

---

## Rule 11: Hallucination Prevention

**Never** fabricate information.

If retrieval confidence is below threshold, return:
```json
{
  "answer": "Insufficient information found in the uploaded document.",
  "confidence": 0.0,
  "needs_human_review": true
}
```

Confidence is a **filter**, not decoration. Low confidence suppresses the answer.

---

## Rule 12: Explainability

Every answer must expose:
- `confidence` (float 0–1)
- `sources` (list of chunk citations with section, page, excerpt, relevance score)
- `coverage_status` ("COVERED" | "EXCLUDED" | "CONDITIONAL" | "UNCLEAR")
- `rerank_score` per source
- `needs_human_review` (bool)
- `retrieval_stages` (chunks at each stage, for debugging)
- `conditions` (list of conditions if CONDITIONAL)

No black-box answers. Every answer must be traceable to a source chunk.

---

## Rule 13: Evaluation

Every retrieval improvement must be evaluated before the version is considered complete.

Use:
- **DeepEval** for unit-style assertions with enforced thresholds
- **RAGAS** for batch scoring

Track per-version in `docs/benchmarks.md`:
- Faithfulness (threshold ≥ 0.85)
- Contextual Precision (threshold ≥ 0.75)
- Contextual Recall (threshold ≥ 0.75)
- Answer Relevancy (threshold ≥ 0.80)

**No retrieval improvement is complete without a benchmark comparison.**

---

## Rule 14: Testing

Every version must include unit tests, and after V10, integration tests must also pass.

Required edge cases to test:
- Empty question list
- Unsupported file format
- Invalid or malformed URL
- Gemini API failure (mock and assert graceful fallback)
- Zero retrieved chunks
- Empty document
- SSRF URL (should reject with 422)
- Unauthenticated request (should reject with 401)

**Never** mark a version complete with failing tests.

Coverage target: ≥ 80% (enforced in CI via `--cov-fail-under=80`).

---

## Rule 15: Code Quality

Correctness before optimization.

Mandatory rules:
- No dead code
- No duplicate functions
- No `print()` — use `structlog` everywhere
- Type hints on all function signatures
- Modular services with clear ownership
- Small, focused files (target < 300 lines per file)
- Dependency injection (no global state after V1)
- All configuration through `config.py` Pydantic Settings

**Never** call `os.getenv()` outside `config.py`. If you need an env var, add it to `Settings`.

---

## Rule 16: Service Boundaries

```
Ingestion services  →  Retrieval services  →  Generation services  →  API layer
```

Direction is one-way:
- Retrieval **never** imports from Generation
- Generation **never** imports from Ingestion
- `main.py` only orchestrates — it wires services together, does not contain logic
- Routes call services — routes do not contain business logic

---

## Rule 17: Security Rules

SSRF protection is mandatory from V11 onward (V1 if the URL downloader exists earlier):

Blocked ranges:
```python
10.0.0.0/8       # private
172.16.0.0/12    # private
192.168.0.0/16   # private
127.0.0.0/8      # loopback
169.254.0.0/16   # AWS metadata endpoint
```

Additional rules:
- Require HTTPS — reject all plain `http://` URLs for document downloads
- Resolve hostname to IP before allowing (DNS rebinding prevention)
- Rate limit all endpoints: query → 10/min, ingest → 5/min
- Bearer token required on all non-health endpoints

---

## Rule 18: Configuration

All environment variables go in `config.py` as Pydantic `Settings` fields.

The system must **fail fast** on startup if required variables are missing. Never start with default empty strings for secrets.

Required variables:
```
GEMINI_API_KEY
HACKATHON_BEARER_TOKEN
```

Optional with defaults:
```
EMBEDDING_MODEL=gemini-embedding-001
LLM_MODEL=gemini-2.5-flash
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
REDIS_URL=redis://localhost:6379
CHROMA_PERSIST_DIR=./chroma_db
RETRIEVAL_TOP_K=20
RERANK_TOP_N=8
FINAL_CONTEXT_CHUNKS=5
PARENT_CHUNK_SIZE=1500
CHILD_CHUNK_SIZE=256
CHUNK_OVERLAP=32
CACHE_TTL_SECONDS=3600
SEMANTIC_CACHE_THRESHOLD=0.04
```

---

## Rule 19: Structure Documentation

`structure.md` must be updated after every version that changes the file layout.

Contents of `structure.md`:
- Full directory tree with one-line description per file
- Current version number
- Date last updated

---

## Rule 20: Version Documentation

Every version creates `docs/RAG_vX.md` containing:

```
# RAG vX — [Version Name]

## Goal
## Previous Limitation (why this was needed)
## Concepts Introduced
## Design Decisions
## Trade-offs
## Changed Files
## New Behavior
## Testing Added
## Benchmark Results (V3 onward)
## Known Limitations
## Next Version Preview
```

Documentation must reflect the **actual implementation**, not copy from `guide.md`.

---

## Rule 21: Git Workflow

- One commit per completed version
- Commit message format: `v{N}: {version name} — {one-line summary}`
- Examples:
  - `v1: stable foundation — fix 5 runtime crashes, add structlog and pydantic settings`
  - `v5: hybrid retrieval — BM25 + dense + RRF fusion, recall +12%`
- Commit only tested, passing code
- Never mix unrelated changes in one commit
- If an earlier bug is found during a later version: create a hotfix commit, document it separately, then continue

---

## Rule 22: API Design

These are the only endpoints. Never add hackathon-specific or ad-hoc endpoints.

```
POST   /api/v1/ingest            — ingest a document by URL
POST   /api/v1/query             — synchronous query
POST   /api/v1/query/stream      — streaming query (SSE)
GET    /api/v1/documents         — list ingested documents
DELETE /api/v1/documents/{id}    — remove a document
GET    /health                   — health check (no auth required)
GET    /metrics                  — Prometheus metrics (V11+)
```

---

## Rule 23: Dependency Management

`requirements.txt` — runtime only:
```
fastapi
uvicorn[standard]
google-generativeai
chromadb==0.5.20
rank-bm25==0.2.2
FlagEmbedding==1.2.11
sentence-transformers==3.3.1
redis==5.0.8
slowapi==0.1.9
sse-starlette==2.1.3
pydantic-settings==2.5.2
structlog==24.4.0
prometheus-fastapi-instrumentator==7.0.0
langchain-core
langgraph
langsmith
tiktoken
python-multipart
httpx
pymupdf
python-docx
```

`requirements-dev.txt` — dev/test only:
```
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-cov==5.0.0
httpx==0.27.2
deepeval==1.5.0
ragas==0.2.6
respx==0.21.1
ruff==0.7.0
locust
```

Packages to **never** re-add (removed in V1):
- `supabase`, `gotrue`, `postgrest`, `realtime` — not used
- `cohere` — not used
- `faiss-cpu` — replaced by ChromaDB/Qdrant
- `langchain-text-splitters` — replaced by custom chunker
- `pdfplumber` — redundant, PyMuPDF handles PDF
- `voyageai` — replaced by Gemini embeddings in V3

---

## Rule 24: Ruff Configuration

`pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "RUF"]

[tool.ruff.per-file-ignores]
"tests/**" = ["S101"]   # allow assert in tests

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

---

## Rule 25: Deployment Checklist (V14)

Before any deployment:

- [ ] All runtime bugs fixed
- [ ] All tests passing (unit + integration + rag_eval)
- [ ] Benchmark thresholds met (see `docs/benchmarks.md`)
- [ ] ChromaDB → Qdrant migrated
- [ ] Redis connected and tested
- [ ] LangSmith connected and tracing
- [ ] Health endpoint returns 200
- [ ] End-to-end smoke test passing
- [ ] Secrets in Railway env vars — never in code or committed files
- [ ] SSRF protection tested
- [ ] Rate limiting tested under load

---

## Rule 26: Frontend Comes Last

Frontend is intentionally the final version.

Priority order:
```
Accuracy → Evaluation → Production Backend → Deployment → Frontend
```

The frontend should visualize:
- Citations (linked to source sections)
- Retrieved chunks per stage
- Confidence score
- Reranker scores
- Retrieval stage breakdown (how many chunks at each step)
- Query latency per stage

The frontend **never** drives backend design decisions.

---

## Rule 27: The Guiding Question

Before implementing anything, ask:

> **"Does this make the system produce more accurate, more trustworthy, and more explainable answers?"**

If the answer is **no**, it does not belong in the current version.