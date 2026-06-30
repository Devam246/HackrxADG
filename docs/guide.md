# PolicyMind AI — Master Guide

> A production-grade, highly accurate, explainable Insurance RAG system.

---

## Vision

Build a production-quality Retrieval-Augmented Generation system that answers complex insurance policy questions with high accuracy, grounded citations, full explainability, and measurable evaluation benchmarks.

This is **not** another chatbot. It is a demonstration of:

- Production RAG Architecture
- Hybrid Retrieval (Dense + Sparse + RRF)
- Agentic Retrieval (LangGraph)
- Explainable AI (citations, scores, stages)
- Rigorous Evaluation (RAGAS + DeepEval)
- Production Engineering (Redis, streaming, rate limiting, SSRF)

**Priority order is fixed:**

```
Accuracy
  ↓
Retrieval Quality
  ↓
Explainability
  ↓
Production Engineering
  ↓
Frontend
```

---

## Final Target Architecture

```
                Documents
                    │
                    ▼
          Document Parser
       (PDF, DOCX, EML, HTML)
                    │
                    ▼
         Section Detection
                    │
                    ▼
      Parent–Child Chunking
      (child=256t, parent=1500t)
                    │
                    ▼
     gemini-embedding-001
                    │
                    ▼
           Qdrant (prod)
        ChromaDB (dev)
                    │
                    ▼
─────────────────────────────────

            User Query
                    │
                    ▼
      HyDE Query Expansion
                    │
                    ▼
Dense Retrieval       BM25 Retrieval
       │                    │
       └──────────┬─────────┘
                  ▼
      Reciprocal Rank Fusion
                  ▼
    Cross Encoder Reranker
      (BAAI/bge-reranker-v2-m3)
                  ▼
      Context Compression
                  ▼
      LangGraph Agentic RAG
   (Retrieve → Grade → Rewrite)
                  ▼
        gemini-2.5-flash
                  ▼
      Structured JSON Output
                  ▼
Confidence • Citations • Sources
                  ▼
  Redis Cache + LangSmith Tracing
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Embeddings | gemini-embedding-001 |
| LLM | gemini-2.5-flash |
| Vector DB (dev) | ChromaDB |
| Vector DB (prod) | Qdrant Cloud |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Agent | LangGraph |
| Evaluation | RAGAS + DeepEval |
| Tracing | LangSmith |
| Caching | Redis (Upstash in prod) |
| Frontend | React + TypeScript + Tailwind |
| Deployment | Railway (backend), Vercel (frontend), Qdrant Cloud, Upstash |

---

## Project Structure (Target)

```
backend/
├── main.py                        # FastAPI app factory + middleware only
├── config.py                      # Pydantic Settings — all env vars
│
├── api/v1/
│   ├── routes_query.py            # POST /api/v1/query, /query/stream
│   ├── routes_ingest.py           # POST /api/v1/ingest
│   ├── routes_documents.py        # GET/DELETE /api/v1/documents
│   ├── routes_health.py           # GET /health, /metrics
│   └── deps.py                    # Auth, rate limiting dependencies
│
├── services/
│   ├── ingestion/
│   │   ├── downloader.py          # UUID-named safe downloader + SSRF guard
│   │   ├── parsers.py             # PDF, DOCX (tables), EML, HTML extraction
│   │   ├── classifier.py          # Domain classifier (insurance, general)
│   │   └── chunker.py             # ParentChildChunker
│   │
│   ├── retrieval/
│   │   ├── embedder.py            # Gemini embedding wrapper with batching
│   │   ├── vector_store.py        # ChromaDB/Qdrant client abstraction
│   │   ├── bm25_index.py          # BM25Okapi sparse retriever
│   │   ├── hybrid_search.py       # RRF fusion
│   │   ├── reranker.py            # BGE cross-encoder
│   │   ├── hyde.py                # HyDE query expansion
│   │   └── compressor.py          # Contextual compression
│   │
│   └── generation/
│       ├── generator.py           # Gemini wrapper with streaming
│       ├── prompts.py             # Insurance-domain prompt templates
│       └── postprocessor.py       # JSON parsing, confidence, citations
│
├── agent/
│   └── rag_graph.py               # LangGraph Retrieve→Grade→Rewrite→Generate
│
├── models/
│   ├── schemas.py                 # Pydantic request/response models
│   └── domain.py                  # Internal dataclasses (Chunk, Document)
│
└── utils/
    ├── cache.py                   # Redis semantic cache
    ├── logging.py                 # structlog configuration
    └── security.py                # SSRF guard, URL sanitizer, token validator

tests/
├── conftest.py
├── unit/
│   ├── test_chunker.py
│   ├── test_hybrid_search.py
│   └── test_security.py
├── integration/
│   └── test_api_query.py
├── rag_eval/
│   └── test_rag_metrics.py
└── locustfile.py

docs/
├── guide.md                       # This file — single source of truth
├── rules.md                       # Development rules
├── structure.md                   # Current file structure (updated each version)
├── RAG_v1.md
├── RAG_v2.md
...
```

---

## Environment Variables

```
GEMINI_API_KEY=
EMBEDDING_MODEL=gemini-embedding-001
LLM_MODEL=gemini-2.5-flash

QDRANT_URL=
QDRANT_API_KEY=

REDIS_URL=redis://localhost:6379
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=policymind-ai

CHROMA_PERSIST_DIR=./chroma_db

HACKATHON_BEARER_TOKEN=
```

---

## Pydantic Settings (config.py target)

```python
class Settings(BaseSettings):
    bearer_token: str
    gemini_api_key: str

    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    retrieval_top_k: int = 20
    rerank_top_n: int = 8
    final_context_chunks: int = 5
    parent_chunk_size: int = 1500
    child_chunk_size: int = 256
    chunk_overlap: int = 32

    chroma_persist_dir: str = "./chroma_db"
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.04

    langsmith_api_key: str = ""
    langsmith_project: str = "policymind-ai"

    class Config:
        env_file = ".env"
```

---

## APIs

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

## Benchmark Targets (DeepEval + RAGAS)

| Metric | Threshold | Target |
|---|---|---|
| Faithfulness | 0.85 | ≥ 0.87 |
| Answer Relevancy | 0.80 | ≥ 0.82 |
| Contextual Precision | 0.75 | ≥ 0.79 |
| Contextual Recall | 0.75 | ≥ 0.84 |

SLA targets (Locust load test):
- p95 latency with cache hit: < 200 ms
- p95 latency on cache miss: < 4.5 s
- Concurrent users: 50, ramp rate: 5/s

---

## Projected Score After All Versions

| Category | Before | After |
|---|---|---|
| Architecture | 3/10 | 9/10 |
| Code Quality | 2/10 | 9/10 |
| Maintainability | 2/10 | 8/10 |
| Scalability | 2/10 | 9/10 |
| Performance | 5/10 | 9/10 |
| Security | 4/10 | 9/10 |
| Testing | 0/10 | 9/10 |
| RAG Quality | 4/10 | 9/10 |
| **Overall** | **~2.5/10** | **~9/10** |

---

---

# Version Roadmap

---

## V1 — Stable Foundation

**Goal:** Fix all runtime crashes and remove dangerous patterns. The system must start and handle requests without crashing.

**Current blockers:**
- Import crash: `build_enhanced_inverted_index` does not exist
- Chunking `TypeError`: `doc_id` missing from `create_chunk`
- Retrieval crash: duplicate function definitions override the correct ones
- Race condition: temp file name collisions under concurrent load
- Thread-safety: global `CURRENT_DOC_TYPE` shared across requests

**Tasks:**

Fix 1 — Import crash (5 min):
```python
# main.py line 22
from rag_pipeline import build_inverted_index  # was build_enhanced_inverted_index
```

Fix 2 — Chunking TypeError (10 min):
```python
def create_chunk(section, text, chunk_index, keywords, doc_id="unknown"):
    ...
```

Fix 3 — Retrieval crash (15 min):
```bash
grep -n "^def " rag_pipeline.py | awk -F: '{print $2}' | sort | uniq -d
# Delete duplicate block (lines ~2246–2673)
```

Fix 4 — Race condition on downloads (20 min):
```python
import uuid, tempfile
from pathlib import Path

def download_file(url, doc_id):
    ext = _detect_extension(url)
    tmp = Path(tempfile.gettempdir()) / f"rag_{doc_id}_{uuid.uuid4().hex}{ext}"
    return tmp
```

Fix 5 — Thread-safety (15 min):
```python
def process_document(text, doc_id):
    doc_type = classify_document(text)   # thread-local, no global
    chunks = adaptive_chunk(text, doc_type, doc_id)
    return chunks, doc_type
```

**Also:**
- Replace all `print()` with `structlog` logging
- Replace all `os.getenv()` calls with a single `config.py` using Pydantic Settings
- Clean `requirements.txt`: remove `supabase`, `cohere`, `faiss-cpu`, `langchain-core`, `langchain-text-splitters`, `pdfplumber`
- Add `ruff` and `pyproject.toml` with lint config
- Set up basic `pytest` structure with `conftest.py`

**Deliverables:**
- Server starts without crashing
- All five bugs fixed
- `config.py` with Pydantic Settings
- Structured logging in place
- `requirements.txt` cleaned
- `ruff` passing with zero errors
- Smoke test: POST /api/v1/query returns 200

**Document:** `docs/RAG_v1.md`

---

## V2 — Architecture Refactor

**Goal:** Convert `rag_pipeline.py` (100 KB monolith) into the modular structure defined above. Zero new features — only reorganization.

**Tasks:**
- Create `services/ingestion/` (downloader, parsers, classifier)
- Create `services/retrieval/` (embedder stub, vector_store stub)
- Create `services/generation/` (generator, prompts, postprocessor)
- Create `models/schemas.py` with typed Pydantic request/response models
- Create `models/domain.py` with `Chunk` and `Document` dataclasses
- Create `utils/` (cache stub, logging, security stub)
- Move all routes into `api/v1/`
- `main.py` becomes app factory only

**Rules:**
- Retrieval never imports Generation
- Generation never imports Ingestion
- `main.py` only orchestrates
- All existing functionality must still work after refactor

**Deliverables:**
- Modular file structure matching the target layout
- All existing tests still pass
- `structure.md` updated
- `docs/RAG_v2.md`

---

## V3 — Gemini Embeddings + ChromaDB

**Goal:** Replace Voyage AI + NumPy cache with Gemini embeddings + ChromaDB persistent vector store.

**Why:**
- Voyage AI: extra API dependency, cost, key management
- NumPy `.npy` + `.pkl`: no metadata filtering, no querying, infinite growth, not production-viable
- ChromaDB: persistent, queryable, filterable, HNSW-indexed, zero-ops for dev

**Tasks:**
- Implement `services/retrieval/embedder.py` using `google-generativeai` for `gemini-embedding-001`
- Implement `services/retrieval/vector_store.py` using ChromaDB
- One document = one collection (naming: `doc_{doc_id}`)
- Every chunk must store: `doc_id`, `section`, `page`, `parent_id`, `chunk_id`, `is_parent`, `token_count`
- Batch embeddings (max 100 per call)
- Remove all Voyage AI imports and dependencies

**ChromaDB collection config:**
```python
metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:M": 32}
```

**Deliverables:**
- `embedder.py` using Gemini
- `vector_store.py` using ChromaDB
- All retrieval working end-to-end with new stack
- Existing benchmark scores maintained or improved
- `docs/RAG_v3.md`

---

## V4 — Parent–Child Chunking

**Goal:** Improve retrieval context completeness.

**Why:** Small chunks (256 tokens) retrieve precisely. But sending a 256-token fragment to the LLM loses surrounding context. Parent chunks (1500 tokens) preserve the full clause, section, or paragraph context.

**Rules:**
- Child chunks are embedded and indexed (used for retrieval)
- Parent chunks are NOT embedded and NOT indexed
- When a child chunk matches, fetch its parent and send the parent to Gemini
- Never send child chunks directly to the LLM
- Prefer semantic boundaries (headings, clauses, sections) over character limits

**Implementation:**
```python
class ParentChildChunker:
    def __init__(self, parent_size=1500, child_size=256, overlap=32):
        ...

    def chunk(self, text, doc_id, doc_type):
        # Returns list of Chunk dataclasses
        # Each child has: chunk_id, parent_id, text, is_parent=False
        # Each parent has: chunk_id, parent_id=None, text, is_parent=True, child_ids=[]
```

**Tasks:**
- Implement `ParentChildChunker` in `services/ingestion/chunker.py`
- Update `vector_store.py` to store only child chunks with parent_id metadata
- Add parent lookup: `get_parent(child_id)` → returns parent text
- Update generation pipeline to always use parent text for LLM context
- Add section-aware splitting (detect headings, clause numbers)

**Unit tests:**
```python
def test_parent_child_creates_hierarchy(sample_policy_text):
    chunks = chunker.chunk(text, "doc123", "insurance")
    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]
    assert len(parents) >= 1
    assert len(children) >= len(parents)
    assert all(c.parent_id is not None for c in children)

def test_no_empty_chunks(sample_policy_text):
    chunks = chunker.chunk(text, "doc456", "insurance")
    assert all(len(c.text.strip()) > 0 for c in chunks)
```

**Deliverables:**
- `chunker.py` with `ParentChildChunker`
- Parent-child metadata in ChromaDB
- Parent text sent to LLM (not children)
- Unit tests passing
- `docs/RAG_v4.md`

---

## V5 — Hybrid Retrieval (BM25 + Dense + RRF)

**Goal:** Improve recall. Dense retrieval misses exact keyword matches. BM25 misses semantic variants. Together they cover both failure modes.

**Why RRF:** BM25 scores (TF-IDF floats) and cosine similarity scores (0–1) are not comparable. Rank-based fusion (RRF) normalizes both into a single ranking without scale mismatch.

**RRF formula:**
```
score(d) = α/(k + rank_dense) + β/(k + rank_bm25)
k = 60  (standard smoothing constant from original RRF paper)
dense_weight = 0.7
sparse_weight = 0.3
```

**Tasks:**
- Implement `services/retrieval/bm25_index.py` using `rank-bm25` (BM25Okapi)
- Build BM25 index per document at ingest time (on child chunk texts)
- Implement `services/retrieval/hybrid_search.py` with `reciprocal_rank_fusion()`
- Update query pipeline: run dense + BM25 simultaneously, fuse with RRF, return top 20

**Unit tests:**
```python
def test_rrf_prefers_docs_ranked_high_by_both():
    dense = [(0, 0.95), (1, 0.80), (2, 0.60)]
    sparse = [(2, 8.5), (0, 7.1), (3, 5.0)]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert fused[0][0] == 0   # top in dense, 2nd in sparse → should win

def test_rrf_handles_disjoint_sets():
    dense = [(0, 0.9), (1, 0.8)]
    sparse = [(5, 10.0), (6, 9.0)]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert set(idx for idx, _ in fused) == {0, 1, 5, 6}
```

**Deliverables:**
- `bm25_index.py` building and persisting index per doc_id
- `hybrid_search.py` with `reciprocal_rank_fusion()`
- Query pipeline using RRF output as top-20 candidates
- Unit tests passing
- Benchmark comparison (V4 vs V5 recall scores)
- `docs/RAG_v5.md`

---

## V6 — HyDE Query Expansion

**Goal:** Improve retrieval on vocabulary-mismatch queries.

**Problem:** Users ask "Will my knee replacement be covered?" but the policy says "Orthopaedic surgery — 2-year waiting period applies." Standard retrieval fails completely on this.

**How HyDE works:** Generate a hypothetical policy excerpt that would answer the question, then retrieve by document-to-document similarity. Policy language retrieves policy language.

**Prompt:**
```
You are an insurance policy expert.
Write a short paragraph (3–5 sentences) that would appear in a professional 
insurance policy and directly answer this question using formal insurance 
terminology. No preamble.

Question: {question}
Hypothetical policy excerpt:
```

**Rules:**
- HyDE is best-effort. If it fails, fall back to original query silently.
- Temperature: 0.3 (slight variation, not random)
- Max tokens: 200

**Tasks:**
- Implement `services/retrieval/hyde.py`
- Integrate before dense retrieval in the pipeline
- Use HyDE embedding for dense search, original query for BM25
- Graceful fallback if Gemini call fails

**Deliverables:**
- `hyde.py` with fallback
- Updated retrieval pipeline
- Test: queries with vocabulary mismatch now retrieve correct chunks
- Benchmark comparison (V5 vs V6)
- `docs/RAG_v6.md`

---

## V7 — Cross-Encoder Reranking

**Goal:** Improve precision. Single highest-ROI retrieval improvement.

**Why cross-encoders beat bi-encoders for reranking:** Bi-encoders compress query and document independently into vectors. Cross-encoders read query + document together — joint attention captures exact interactions (negations, conditions, exceptions) that bi-encoders miss.

**Model:** `BAAI/bge-reranker-v2-m3` (~1.2 GB, load once with `@lru_cache`)

**Pipeline after V7:**
```
HyDE → Dense + BM25 → RRF → top 20
  ↓
Cross-encoder rerank → top 8
  ↓
Context compression → top 5
  ↓
Gemini
```

**Tasks:**
- Implement `services/retrieval/reranker.py` with `CrossEncoderReranker`
- Load model once with `@lru_cache(maxsize=1)`, use FP16
- Score all 20 RRF candidates, keep top 8
- Add `rerank_score` to chunk metadata (exposed in response for explainability)
- Handle model load failure gracefully (fall back to RRF ordering)

**Deliverables:**
- `reranker.py` with lazy model loading
- Top-20 → top-8 reranking in pipeline
- `rerank_score` exposed in API response
- Benchmark comparison (V6 vs V7 precision scores)
- `docs/RAG_v7.md`

---

## V8 — Context Compression

**Goal:** Reduce hallucinations and cut prompt tokens by 40–60%.

**Why:** After reranking, the top-8 parent chunks are still ~12,000 tokens. Most of that is irrelevant to the specific question. Compressing to only relevant sentences reduces hallucination risk and cost.

**Prompt:**
```
Extract ONLY the sentences from the policy excerpt below that directly 
answer: "{query}"
If none are relevant, return "NO_RELEVANT_CONTENT".

Policy excerpt:
{chunk_text}
```

**Rules:**
- Temperature: 0.0 (deterministic extraction, not generation)
- Max output tokens: 300 per chunk
- Discard chunks that return "NO_RELEVANT_CONTENT"
- Keep minimum 2 chunks even if compression removes context (fallback to full parent)

**Tasks:**
- Implement `services/retrieval/compressor.py`
- Apply after reranking (top 8 → compressed → top 5 for generation)
- Track token count before/after compression (log the reduction)

**Deliverables:**
- `compressor.py`
- Token reduction logging (avg % reduction per request)
- Faithfulness improvement vs V7
- `docs/RAG_v8.md`

---

## V9 — Agentic RAG (LangGraph)

**Goal:** Enable intelligent multi-hop retrieval for questions that single-pass retrieval cannot answer.

**Why:** Some queries require the system to recognize "I didn't find enough context" and reformulate the question before trying again. A simple pipeline cannot do this.

**LangGraph workflow:**
```
Retrieve (run full V8 pipeline)
  ↓
Grade (LLM checks: do retrieved chunks actually answer the question?)
  ↓
Enough context? ──Yes──→ Generate → Structured Response
       │
      No
       ↓
Rewrite Query (LLM reformulates with different terminology)
  ↓
Retrieve again (max 2 retries)
  ↓
Generate (even if imperfect — return with low confidence flag)
```

**Grader prompt:**
```
Does the following retrieved context contain enough information to 
answer this insurance question accurately?
Question: {question}
Context: {context}
Respond with only: YES or NO
```

**Rules:**
- Max 2 rewrite cycles (prevent infinite loops)
- If after 2 retries context is still insufficient, return answer with `needs_human_review: true`
- Confidence below 0.4 always triggers `needs_human_review: true`
- Never answer from Gemini general knowledge — only from retrieved context

**Deliverables:**
- `agent/rag_graph.py` with LangGraph state machine
- Rewrite prompt tuned for insurance vocabulary
- `needs_human_review` field in response
- Test: ambiguous queries now trigger rewrite correctly
- `docs/RAG_v9.md`

---

## V10 — Evaluation Suite

**Goal:** Measure every retrieval improvement with enforced thresholds. This version makes the project genuinely benchmark-worthy.

**Tools:** DeepEval (unit-style metric assertions) + RAGAS (batch evaluation)

**Benchmark dataset** (`tests/rag_eval/insurance_benchmark.json`):

```json
[
  {
    "input": "What is the waiting period for cataract surgery?",
    "expected_output": "Cataract surgery is excluded during the first 24 months of coverage.",
    "retrieval_context": [
      "3.1 Cataract surgery is excluded during the first 24 months of coverage.",
      "3.2 Other ophthalmology procedures require a 12-month waiting period."
    ]
  },
  {
    "input": "Is maternity covered under the policy?",
    "expected_output": "Maternity expenses are covered after a 9-month waiting period.",
    "retrieval_context": [
      "3.2 Maternity expenses are covered after a waiting period of 9 months.",
      "Normal delivery: up to INR 25,000. Caesarean: up to INR 50,000."
    ]
  },
  {
    "input": "Does this policy cover commercial basement flood damage?",
    "expected_output": "No, the policy explicitly excludes coverage for flood damage to basements.",
    "retrieval_context": [
      "Section 4.2: Water damage from external flooding to underground levels is excluded."
    ]
  }
]
```

**Metrics and thresholds:**

| Metric | Threshold | Target |
|---|---|---|
| Faithfulness | 0.85 | ≥ 0.87 |
| Answer Relevancy | 0.80 | ≥ 0.82 |
| Contextual Precision | 0.75 | ≥ 0.79 |
| Contextual Recall | 0.75 | ≥ 0.84 |

**Tasks:**
- Create `tests/rag_eval/test_rag_metrics.py` with DeepEval `assert_test` (fails CI if below threshold)
- Create `tests/rag_eval/insurance_benchmark.json` with ≥ 15 questions
- Integrate RAGAS for batch scoring
- Set up GitHub Actions `rag-eval` job (runs only on merges to main)
- Store per-version benchmark scores in `docs/benchmarks.md`

**Deliverables:**
- Full DeepEval test suite with thresholds
- `insurance_benchmark.json` with ≥ 15 Q&A pairs
- `docs/benchmarks.md` with V1–V10 scores
- CI badge on README
- `docs/RAG_v10.md`

---

## V11 — Production Features

**Goal:** Production readiness: caching, tracing, streaming, rate limiting, SSRF protection, Prometheus metrics.

**Tasks:**

Redis semantic cache (`utils/cache.py`):
- Exact cache: hash(doc_url + sorted questions) → cached response
- Semantic cache: if query embedding is within cosine distance 0.04 of a cached query → return cached answer without LLM call
- TTL: 3600 seconds

SSRF protection (`utils/security.py`):
- Block all private IP ranges: 10.x, 172.16.x, 192.168.x, 127.x, 169.254.x
- Require HTTPS only
- Resolve hostname before allowing request

Rate limiting (slowapi):
- `POST /api/v1/query`: 10/minute per IP
- `POST /api/v1/ingest`: 5/minute per IP

SSE Streaming (`routes_query.py`):
- `/api/v1/query/stream` emits status events during processing
- Events: `downloading`, `chunking`, `embedding`, `retrieving`, `reranking`, `compressing`, `generating`, `result`

LangSmith tracing:
- Trace every LangGraph run
- Tag with doc_type, question count, cache hit/miss

Prometheus metrics:
- `rag_queries_total` (labels: doc_type, cached)
- `rag_query_duration_seconds` (histogram by stage)
- `rag_retrieval_chunks_returned` (histogram)
- Expose at `/metrics`

Security tests:
```python
def test_blocks_loopback():
    assert SSRFGuard().is_safe_url("http://127.0.0.1/secrets") is False

def test_blocks_aws_metadata():
    assert SSRFGuard().is_safe_url("http://169.254.169.254/latest/meta-data/") is False

def test_allows_public_https():
    assert SSRFGuard().is_safe_url("https://example.com/policy.pdf") is True

def test_blocks_plain_http():
    assert SSRFGuard().is_safe_url("http://example.com/policy.pdf") is False
```

Load test (Locust):
- 50 concurrent users, 5/s ramp, 5 min run
- Target: p95 < 200 ms (cache hit), p95 < 4.5 s (cache miss)

**Deliverables:**
- Redis cache (exact + semantic)
- SSRF guard with tests
- Rate limiting
- SSE streaming endpoint
- LangSmith tracing connected
- Prometheus metrics
- Locust results meeting SLA
- GitHub Actions CI running all test layers
- `docs/RAG_v11.md`

---

## V12 — Qdrant Migration (Production Vector DB)

**Goal:** Replace ChromaDB with Qdrant for production deployment.

**Why:** ChromaDB is excellent for development but Qdrant offers managed cloud, filtering APIs, quantization, and horizontal scalability.

**Tasks:**
- Add Qdrant client to `vector_store.py` behind an abstraction interface
- Toggle via `VECTOR_DB=chroma|qdrant` in settings
- Migrate existing collections
- Test all retrieval paths with Qdrant backend
- Configure Qdrant Cloud connection (URL + API key)

**Deliverables:**
- `vector_store.py` supports both ChromaDB and Qdrant via settings toggle
- Qdrant Cloud connected
- All retrieval tests passing with Qdrant
- `docs/RAG_v12.md`

---

## V13 — CI/CD + Full Test Suite

**Goal:** Full CI pipeline with unit, integration, and RAG eval stages. Coverage ≥ 80%.

**GitHub Actions jobs:**

```
lint       → ruff check + ruff format --check
unit       → pytest tests/unit/ --cov --cov-fail-under=80
integration → pytest tests/integration/ (with Redis service)
rag-eval   → pytest tests/rag_eval/ (main branch only, uses real API keys)
```

**Deliverables:**
- `.github/workflows/ci.yml`
- Codecov integration
- Coverage badge on README
- All jobs green
- `docs/RAG_v13.md`

---

## V14 — Deployment

**Goal:** Live production system.

**Stack:**
- Backend → Railway (auto-deploy from main)
- Frontend → Vercel
- Vector DB → Qdrant Cloud
- Redis → Upstash
- Tracing → LangSmith cloud

**Checklist before deploy:**
- All runtime bugs fixed ✓
- Tests passing ✓
- Benchmark thresholds met ✓
- ChromaDB → Qdrant migrated ✓
- Redis connected ✓
- LangSmith connected ✓
- Health endpoint returning 200 ✓
- End-to-end smoke test passing ✓
- Secrets in Railway env vars (never in code) ✓

**Deliverables:**
- Live URL
- Health check passing
- `docs/RAG_v14.md`

---

## V15 — Frontend

**Goal:** Visualize the full RAG pipeline — citations, retrieval stages, confidence, reranker scores.

**Pages:**
- Landing — project overview + live demo CTA
- Chat — real-time streaming query interface
- Analytics — benchmark scores, latency charts, cache hit rate
- Explainability — per-answer: retrieved chunks, similarity scores, reranker scores, compression ratio, retrieval path
- Admin — document management, collection stats

**Rules:**
- Frontend never drives backend design
- Explainability is the primary differentiator (not the chat UI)
- Every answer shows its full retrieval trace

**Deliverables:**
- React + TypeScript + Tailwind app
- Deployed to Vercel
- All pages implemented
- `docs/RAG_v15.md`

---

## Future Ideas (Not in Current Roadmap)

- GraphRAG + Knowledge Graph integration
- OCR for scanned PDFs
- Multi-language policy support
- Fine-tuned reranker on insurance domain
- Local LLM support (Ollama)
- Multi-agent workflows (one agent per policy clause type)
- Automatic policy comparison (diff two policies)
- Regulatory compliance assistant
- Voice interface
- Table and image understanding