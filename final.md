# Final Improvement Plan — Insurance RAG Platform
### Synthesized from Strategy 1 (imrovements.md) + Strategy 2 (improvements2.md) against Project Analysis

---

## Current State Summary

The project is a pre-alpha RAG API for insurance document Q&A built on FastAPI + Voyage AI + Gemini. It has a solid conceptual foundation — hybrid retrieval, multi-format parsing, multi-layer caching — but is blocked by **4 critical runtime crashes**, a race condition, a 100 KB monolith, zero tests, and an outdated README. The three pillars below address the highest-ROI transformations.

---

## Pillar 1: Good Results on Trusted Benchmarks and Rigorous Testing Strategy

### Why This Is the Top Priority
The project has **zero tests**. The README claims pytest coverage with LLM mocks — it does not exist. Every code change is a blind flight. Testing is also the most recruiter-visible improvement because it can be demonstrated with CI badges, coverage reports, and metric scores.

---

### 1.1 Fix the Runtime Before Testing Anything

All four crashes must be resolved first — a test suite on broken code measures nothing.

**Fix 1 — Import crash (5 min):**
```python
# main.py line 22 — change:
from rag_pipeline import build_enhanced_inverted_index  # does not exist
# to:
from rag_pipeline import build_inverted_index
```

**Fix 2 — Chunking TypeError (10 min):**
```python
# rag_pipeline.py — make doc_id optional or thread it through all callers
def create_chunk(section, text, chunk_index, keywords, doc_id="unknown"):
    ...
```

**Fix 3 — Retrieval crash (15 min):**
```bash
# Verify duplicate functions and delete the bottom overrides
grep -n "^def " rag_pipeline.py | awk -F: '{print $2}' | sort | uniq -d
# Delete the entire duplicate block (lines ~2246–2673) — the top definitions are correct
```

**Fix 4 — Race condition on downloads (20 min):**
```python
import uuid, tempfile
from pathlib import Path

def download_file(url, doc_id):
    ext = _detect_extension(url)
    tmp = Path(tempfile.gettempdir()) / f"rag_{doc_id}_{uuid.uuid4().hex}{ext}"
    # download and write to tmp ...
    return tmp
```

**Fix 5 — Thread-safety on CURRENT_DOC_TYPE (15 min):**
```python
# Remove the global. Pass doc_type as a local argument through the call chain.
def process_document(text, doc_id):
    doc_type = classify_document(text)   # thread-local
    chunks = adaptive_chunk(text, doc_type, doc_id)
    return chunks, doc_type
```

---

### 1.2 Unit Tests (pytest)

Create `tests/conftest.py` with shared fixtures and mocks so every test is isolated and reproducible:

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def sample_policy_text():
    return """
    SECTION 3: EXCLUSIONS
    3.1 Cataract surgery is excluded during the first 24 months of coverage.
    3.2 Maternity expenses are covered after a 9-month waiting period.
    """

@pytest.fixture
def mock_voyage_client():
    with patch("services.retrieval.embedder.voyageai.Client") as mock:
        mock.return_value.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
        yield mock.return_value

@pytest.fixture
def mock_gemini_model():
    with patch("services.generation.generator.genai.GenerativeModel") as mock:
        mock.return_value.generate_content_async = AsyncMock(
            return_value=MagicMock(text='{"answers": [{"question": "test", "answer": "24 months", "confidence": 0.9, "sources": [], "coverage_status": "EXCLUDED", "conditions": [], "needs_human_review": false}], "document_type": "insurance", "policy_name": null, "processing_notes": ""}')
        )
        yield mock.return_value
```

Critical unit tests to write immediately:

```python
# tests/unit/test_chunker.py
from services.ingestion.chunker import ParentChildChunker

def test_parent_child_creates_hierarchy(sample_policy_text):
    chunker = ParentChildChunker(parent_size=500, child_size=128, overlap=16)
    chunks = chunker.chunk(sample_policy_text, "doc123", "insurance")
    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]
    assert len(parents) >= 1
    assert len(children) >= len(parents)
    assert all(c.parent_id is not None for c in children)

def test_no_empty_chunks(sample_policy_text):
    chunker = ParentChildChunker(parent_size=200, child_size=50, overlap=10)
    chunks = chunker.chunk(sample_policy_text, "doc456", "insurance")
    assert all(len(c.text.strip()) > 0 for c in chunks)


# tests/unit/test_hybrid_search.py
from services.retrieval.hybrid_search import reciprocal_rank_fusion

def test_rrf_prefers_docs_ranked_high_by_both():
    dense = [(0, 0.95), (1, 0.80), (2, 0.60)]
    sparse = [(2, 8.5), (0, 7.1), (3, 5.0)]
    fused = reciprocal_rank_fusion(dense, sparse)
    fused_ids = [idx for idx, _ in fused]
    assert fused_ids[0] == 0   # top in dense, 2nd in sparse → should win

def test_rrf_handles_disjoint_sets():
    dense = [(0, 0.9), (1, 0.8)]
    sparse = [(5, 10.0), (6, 9.0)]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert set(idx for idx, _ in fused) == {0, 1, 5, 6}


# tests/unit/test_security.py
from app.utils.security import SSRFGuard

def test_blocks_loopback():
    assert SSRFGuard().is_safe_url("http://127.0.0.1/secrets") is False

def test_blocks_aws_metadata():
    assert SSRFGuard().is_safe_url("http://169.254.169.254/latest/meta-data/") is False

def test_blocks_private_ranges():
    assert SSRFGuard().is_safe_url("http://192.168.1.1/") is False
    assert SSRFGuard().is_safe_url("http://10.0.0.1/") is False

def test_allows_public_https():
    assert SSRFGuard().is_safe_url("https://example.com/policy.pdf") is True

def test_blocks_plain_http():
    assert SSRFGuard().is_safe_url("http://example.com/policy.pdf") is False
```

---

### 1.3 Integration Tests for API Routes

```python
# tests/integration/test_api_query.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_query_requires_auth(async_client):
    r = await async_client.post("/api/v1/query", json={
        "documents": "https://example.com/policy.pdf",
        "questions": ["What is covered?"]
    })
    assert r.status_code == 401

@pytest.mark.asyncio
async def test_query_rejects_ssrf_url(async_client):
    r = await async_client.post(
        "/api/v1/query",
        headers={"Authorization": "Bearer test-token"},
        json={"documents": "http://127.0.0.1:8080/internal", "questions": ["test"]}
    )
    assert r.status_code == 422
    assert "unsafe" in r.json()["detail"].lower()

@pytest.mark.asyncio
async def test_query_returns_structured_response(
    async_client, mock_voyage_client, mock_gemini_model
):
    r = await async_client.post(
        "/api/v1/query",
        headers={"Authorization": "Bearer test-token"},
        json={"documents": "https://example.com/policy.pdf", "questions": ["Cataract waiting period?"]}
    )
    assert r.status_code == 200
    data = r.json()
    assert "answers" in data
    assert "confidence" in data["answers"][0]
    assert "coverage_status" in data["answers"][0]
    assert "sources" in data["answers"][0]
```

---

### 1.4 RAG Metric Evaluation with DeepEval — The Benchmark Suite

This is the headline item. Automated RAG evaluation with enforced thresholds proves the pipeline quality and makes the project genuinely resume-worthy.

```python
# tests/rag_eval/test_rag_metrics.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

INSURANCE_GROUND_TRUTH = [
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

@pytest.mark.parametrize("td", INSURANCE_GROUND_TRUTH)
def test_faithfulness(td, run_rag_pipeline):
    actual_output, retrieved_context = run_rag_pipeline(td["input"])
    tc = LLMTestCase(input=td["input"], actual_output=actual_output,
                     retrieval_context=retrieved_context)
    metric = FaithfulnessMetric(threshold=0.85)
    assert_test(tc, [metric])   # fails CI if score < 0.85

@pytest.mark.parametrize("td", INSURANCE_GROUND_TRUTH)
def test_answer_relevancy(td, run_rag_pipeline):
    actual_output, retrieved_context = run_rag_pipeline(td["input"])
    tc = LLMTestCase(input=td["input"], actual_output=actual_output,
                     retrieval_context=retrieved_context)
    assert_test(tc, [AnswerRelevancyMetric(threshold=0.80)])

@pytest.mark.parametrize("td", INSURANCE_GROUND_TRUTH)
def test_contextual_precision(td, run_rag_pipeline):
    actual_output, retrieved_context = run_rag_pipeline(td["input"])
    tc = LLMTestCase(input=td["input"], actual_output=actual_output,
                     expected_output=td["expected_output"],
                     retrieval_context=retrieved_context)
    assert_test(tc, [ContextualPrecisionMetric(threshold=0.75)])
```

Target benchmark scores to report:

|
 Metric 
|
 Threshold 
|
 Target Score 
|
|
---
|
---
|
---
|
|
 Faithfulness 
|
 0.85 
|
 ≥ 0.87 
|
|
 Answer Relevancy 
|
 0.80 
|
 ≥ 0.82 
|
|
 Contextual Precision 
|
 0.75 
|
 ≥ 0.79 
|
|
 Contextual Recall 
|
 0.75 
|
 ≥ 0.84 
|

---

### 1.5 Load Testing with Locust

```python
# tests/locustfile.py
from locust import HttpUser, task, between
import random

class RAGLoadTester(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def stream_query(self):
        payload = {"query": random.choice([
            "What are the premium penalties for late submissions?",
            "Is lightning damage covered under general liability?",
            "What is the sub-limit for cataract surgery?",
        ])}
        with self.client.post("/api/v1/query/stream",
                              json=payload,
                              headers={"Authorization": "Bearer token"},
                              stream=True) as r:
            r.success() if r.status_code == 200 else r.failure(r.status_code)

    @task(1)
    def ingest_document(self):
        self.client.post("/api/v1/ingest",
                         json={"url": "https://example.com/policy.pdf"},
                         headers={"Authorization": "Bearer token"})
```

Run: `locust -f tests/locustfile.py --headless -u 50 -r 5 --run-time 5m --host http://localhost:8000`
Target SLA: p95 response start < 200 ms with semantic cache hit, < 4.5 s on cache miss.

---

### 1.6 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI — Test & Lint

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install ruff && ruff check backend/ && ruff format --check backend/

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r backend/requirements-dev.txt
      - run: |
          cd backend
          pytest tests/unit/ -v --cov=app --cov-report=xml --cov-fail-under=80
      - uses: codecov/codecov-action@v4
        with: {file: backend/coverage.xml}

  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis: {image: redis:7-alpine, ports: ["6379:6379"]}
    env:
      HACKATHON_BEARER_TOKEN: test-ci-token
      REDIS_URL: redis://localhost:6379
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r backend/requirements.txt -r backend/requirements-dev.txt
      - run: cd backend && pytest tests/integration/ -v --timeout=60

  rag-eval:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'   # only on merges to main
    env:
      VOYAGE_API_KEY: ${{ secrets.VOYAGE_API_KEY }}
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r backend/requirements.txt deepeval ragas
      - run: cd backend && pytest tests/rag_eval/ -v --tb=short
```

---

## Pillar 2: High Performance

### 2.1 Three-Stage Retrieval Pipeline (The Core Upgrade)

Replace the current single-stage NumPy cosine loop with a three-stage funnel:

```
Query
  ↓
[Stage 1] HyDE Expansion + BM25 (sparse) + Voyage AI (dense) → RRF Fusion → top 20 candidates
  ↓
[Stage 2] BGE-Reranker-v2-m3 cross-encoder → top 8 candidates
  ↓
[Stage 3] Contextual compression (LLM extracts only relevant sentences) → top 5 for generation
  ↓
Gemini 2.5 Flash → Structured JSON response
```

**Stage 1 — Hybrid BM25 + Dense Search with RRF:**

```python
# services/retrieval/hybrid_search.py
def reciprocal_rank_fusion(dense_results, sparse_results, k=60,
                           dense_weight=0.7, sparse_weight=0.3):
    """
    RRF: score(d) = α/(k + rank_dense) + β/(k + rank_bm25)
    k=60 is the standard smoothing constant from the original RRF paper.
    Rank-based fusion handles mismatched BM25/cosine scales correctly.
    """
    scores = {}
    for rank, (chunk_idx, _) in enumerate(dense_results):
        scores[chunk_idx] = scores.get(chunk_idx, 0.0) + dense_weight / (k + rank + 1)
    for rank, (chunk_idx, _) in enumerate(sparse_results):
        scores[chunk_idx] = scores.get(chunk_idx, 0.0) + sparse_weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Stage 2 — Cross-Encoder Reranking (single highest-ROI improvement):**

```python
# services/retrieval/reranker.py
from FlagEmbedding import FlagReranker
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_reranker():
    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)  # ~1.2 GB, load once

class CrossEncoderReranker:
    def rerank(self, query, candidates, top_n=8):
        """
        Cross-encoder reads query + document together (bi-encoders compress separately).
        This direct joint modeling is why cross-encoders beat bi-encoders at reranking.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = _load_reranker().compute_score(pairs, normalize=True)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [{"rerank_score": float(s), **c} for s, c in ranked[:top_n]]
```

**Stage 3 — Contextual Compression (reduces prompt tokens by 40–60%):**

```python
# services/retrieval/compressor.py
COMPRESSION_PROMPT = """Extract ONLY the sentences from the policy excerpt below
that directly answer: "{query}"
If none are relevant, return "NO_RELEVANT_CONTENT".

Policy excerpt:
{chunk_text}"""

class ContextualCompressor:
    async def compress(self, query, chunks, model):
        compressed = []
        for chunk in chunks:
            response = await model.generate_content_async(
                COMPRESSION_PROMPT.format(query=query, chunk_text=chunk["text"]),
                generation_config={"max_output_tokens": 300, "temperature": 0.0}
            )
            extracted = response.text.strip()
            if extracted != "NO_RELEVANT_CONTENT":
                chunk["compressed_text"] = extracted
                compressed.append(chunk)
        return compressed
```

---

### 2.2 HyDE Query Expansion (bridges vocabulary gap)

Insurance users ask "Will my knee replacement be covered?" but the policy says "Orthopaedic surgery — 2-year waiting period applies." Standard retrieval fails on this vocabulary mismatch. HyDE generates a hypothetical policy excerpt and retrieves by document-to-document similarity instead.

```python
# services/retrieval/hyde.py
INSURANCE_HYDE_PROMPT = """You are an insurance policy expert.
Write a short paragraph (3–5 sentences) that would appear in a professional insurance policy
and directly answer this question using formal insurance terminology. No preamble.

Question: {question}
Hypothetical policy excerpt:"""

class HyDEExpander:
    async def expand_query(self, query):
        try:
            response = await self.model.generate_content_async(
                INSURANCE_HYDE_PROMPT.format(question=query),
                generation_config={"max_output_tokens": 200, "temperature": 0.3}
            )
            return response.text.strip()
        except Exception:
            return query   # graceful fallback to original query
```

---

### 2.3 Replace NumPy Cache with ChromaDB (Persistent Vector Store)

The current `.npy` + `.pkl` disk cache has no querying capability, no metadata filtering, and grows infinitely. ChromaDB solves all three.

```python
# services/retrieval/vector_store.py
import chromadb

class VectorStore:
    def __init__(self, persist_dir):
        self.client = chromadb.PersistentClient(path=persist_dir)

    def get_or_create_collection(self, doc_id):
        return self.client.get_or_create_collection(
            name=f"doc_{doc_id}",
            metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:M": 32}
        )

    def upsert_chunks(self, doc_id, chunks, embeddings):
        col = self.get_or_create_collection(doc_id)
        col.upsert(
            ids=[f"{doc_id}_chunk_{c['chunk_index']}" for c in chunks],
            embeddings=embeddings,
            documents=[c['text'] for c in chunks],
            metadatas=[{
                "doc_id": doc_id, "section": c.get("section", ""),
                "is_parent": c.get("is_parent", False),
                "parent_id": c.get("parent_id", ""),
                "token_count": c.get("token_count", 0)
            } for c in chunks]
        )

    def dense_search(self, doc_id, query_embedding, top_k=20):
        col = self.get_or_create_collection(doc_id)
        return col.query(query_embeddings=[query_embedding], n_results=top_k,
                         include=["documents", "metadatas", "distances"])
```

---

### 2.4 Semantic Cache with Redis (near-zero latency on repeated queries)

```python
# app/utils/cache.py
import hashlib, json
import redis.asyncio as aioredis

class CacheManager:
    def __init__(self, redis_url, ttl=3600):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl

    def _key(self, doc_url, questions):
        payload = f"{doc_url}::{sorted(questions)}"
        return f"rag:query:{hashlib.sha256(payload.encode()).hexdigest()}"

    async def get(self, doc_url, questions):
        v = await self.redis.get(self._key(doc_url, questions))
        return json.loads(v) if v else None

    async def set(self, doc_url, questions, result):
        await self.redis.setex(self._key(doc_url, questions), self.ttl, json.dumps(result))
```

For semantic caching (Strategy 2 addition): when a query's embedding is within cosine distance 0.04 of a cached query embedding, return the cached answer immediately without any LLM call.

---

### 2.5 Real-Time Streaming via Server-Sent Events

```python
# app/api/v1/routes_query.py
from sse_starlette.sse import EventSourceResponse
import json

@router.post("/query/stream")
async def stream_query(request: QueryRequest, token=Depends(verify_token)):
    async def events():
        yield {"event": "status", "data": json.dumps({"stage": "downloading", "message": "Fetching document..."})}
        doc_id = await ingest_service.process(str(request.documents))
        yield {"event": "status", "data": json.dumps({"stage": "retrieving", "message": "Running hybrid retrieval..."})}
        yield {"event": "status", "data": json.dumps({"stage": "reranking", "message": "Cross-encoder reranking..."})}
        result = await query_service.answer(doc_id, request.questions)
        yield {"event": "result", "data": result.model_dump_json()}

    return EventSourceResponse(events())
```

---

### 2.6 Parent-Child Chunking (retrieval precision + context completeness)

Small child chunks (256 tokens) are indexed for precise retrieval. When a child matches, the parent (1500 tokens) is returned to the LLM, preserving full context around the matched clause.

```python
# services/ingestion/chunker.py
class ParentChildChunker:
    """
    Child chunks: ~256 tokens. Used for embedding and retrieval (precision).
    Parent chunks: ~1500 tokens. Used as LLM context (completeness).
    """
    def __init__(self, parent_size=1500, child_size=256, overlap=32):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def chunk(self, text, doc_id, doc_type):
        all_chunks = []
        for p_idx, parent_text in enumerate(self._split(text, self.parent_size)):
            parent_id = f"{doc_id}_parent_{p_idx}"
            children = []
            for c_idx, child_text in enumerate(self._split(parent_text, self.child_size)):
                child_id = f"{parent_id}_child_{c_idx}"
                children.append({"chunk_id": child_id, "text": child_text,
                                  "is_parent": False, "parent_id": parent_id,
                                  "doc_id": doc_id, "chunk_index": c_idx})
            all_chunks.extend(children)
            all_chunks.append({"chunk_id": parent_id, "text": parent_text,
                                "is_parent": True, "parent_id": None,
                                "doc_id": doc_id, "child_ids": [c["chunk_id"] for c in children]})
        return all_chunks
```

---

### 2.7 Prometheus Metrics (observable performance)

```python
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

RAG_QUERIES_TOTAL = Counter("rag_queries_total", "Total queries", ["doc_type", "cached"])
RAG_QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds", "Query processing time",
    ["stage"], buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

---

## Pillar 3: Clean Code and Good Code Structure

### 3.1 Split the 100 KB Monolith

The single `rag_pipeline.py` file handles downloading, parsing, classifying, chunking, embedding, retrieving, reranking, and generating. This violates every principle of maintainable code. Split into focused modules:

```
backend/app/
├── main.py                      # FastAPI app factory + middleware only
├── config.py                    # All env vars in one validated Pydantic Settings object
│
├── api/v1/
│   ├── routes_query.py          # POST /api/v1/query, /query/stream
│   ├── routes_ingest.py         # POST /api/v1/ingest
│   ├── routes_health.py         # GET  /api/v1/health, /metrics
│   └── deps.py                  # FastAPI dependency injectors (auth, rate limiting)
│
├── services/
│   ├── ingestion/
│   │   ├── downloader.py        # UUID-named safe file downloader + SSRF guard
│   │   ├── parsers.py           # PDF, DOCX (with tables), EML extraction
│   │   ├── classifier.py        # Domain classifier
│   │   └── chunker.py           # ParentChildChunker
│   │
│   ├── retrieval/
│   │   ├── embedder.py          # Voyage AI wrapper with batching
│   │   ├── vector_store.py      # ChromaDB client
│   │   ├── bm25_index.py        # BM25Okapi sparse retriever
│   │   ├── hybrid_search.py     # RRF fusion
│   │   ├── reranker.py          # BGE cross-encoder
│   │   ├── hyde.py              # HyDE query expansion
│   │   └── compressor.py        # Contextual compression
│   │
│   └── generation/
│       ├── generator.py         # Gemini wrapper with streaming
│       ├── prompts.py           # Insurance-domain prompt templates
│       └── postprocessor.py     # JSON parsing, confidence scoring, citations
│
├── models/
│   ├── schemas.py               # Pydantic request/response models
│   └── domain.py                # Internal dataclasses (Chunk, Document)
│
└── utils/
    ├── cache.py                 # Redis + LRU cache abstraction
    ├── logging.py               # structlog configuration
    └── security.py              # SSRF guard, URL sanitizer, token validator
```

---

### 3.2 Centralized Config (Pydantic Settings)

Replace 15+ scattered `os.getenv()` calls with a single validated config object that fails fast on startup if required vars are missing:

```python
# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    bearer_token: str
    voyage_api_key: str
    gemini_api_key: str

    embedding_model: str = "voyage-3.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "gemini-2.5-flash-lite"

    retrieval_top_k: int = 20
    rerank_top_n: int = 8
    final_context_chunks: int = 5
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    parent_chunk_size: int = 1500

    chroma_persist_dir: str = "./chroma_db"
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()
```

---

### 3.3 Structured Logging (replace all print() calls)

```python
# app/utils/logging.py
import structlog, logging

def configure_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

logger = structlog.get_logger()

# Usage:
logger.info("document_ingested", doc_id=doc_id, doc_type=doc_type, chunk_count=len(chunks))
logger.error("retrieval_failed", doc_id=doc_id, error=str(e), exc_info=True)
```

---

### 3.4 Typed Pydantic Schemas for Requests and Responses

The current response exposes plain text answers. Expose structured data with citations, confidence, and coverage status:

```python
# app/models/schemas.py
from pydantic import BaseModel, Field

class SourceCitation(BaseModel):
    chunk_id: str
    section: str
    excerpt: str
    relevance: float = Field(ge=0.0, le=1.0)

class AnswerResult(BaseModel):
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SourceCitation]
    coverage_status: str  # "COVERED" | "EXCLUDED" | "CONDITIONAL" | "UNCLEAR"
    conditions: list[str] = []
    needs_human_review: bool = False

class QueryResponse(BaseModel):
    answers: list[AnswerResult]
    document_id: str
    processing_time_ms: float
    doc_type: str
    policy_name: str | None
    retrieval_stages: dict  # chunks at each stage for debugging
    cached: bool = False
```

---

### 3.5 Clean `requirements.txt`

**Remove immediately (unused — confirmed by project analysis):**
```
supabase, gotrue, postgrest, realtime
cohere
faiss-cpu
langchain-core, langchain-text-splitters
pdfplumber  (redundant — PyMuPDF already handles PDF)
```

**Add:**
```
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
tiktoken
```

**Dev only (`requirements-dev.txt`):**
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

---

### 3.6 Ruff Linting Config (`pyproject.toml`)

```toml
[
tool.ruff
]
target-version = "py311"
line-length = 100
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "RUF"]

[
tool.ruff.per-file-ignores
]
"tests/**" = ["S101"]   # allow assert in tests

[
tool.pytest.ini_options
]
asyncio_mode = "auto"
testpaths = ["tests"]

[
tool.coverage.report
]
fail_under = 80
show_missing = true
```

---

### 3.7 Rate Limiting and SSRF Protection

```python
# Rate limiting (app/main.py)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/query/stream")
@limiter.limit("10/minute")
async def stream_query(request: Request, ...): ...

# SSRF guard (app/utils/security.py)
import ipaddress, socket
from urllib.parse import urlparse

BLOCKED_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # AWS metadata endpoint
]

class SSRFGuard:
    def is_safe_url(self, url):
        parsed = urlparse(url)
        if parsed.scheme != "https":
            return False
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
        except (socket.gaierror, ValueError):
            return False
        return not any(ip in r for r in BLOCKED_RANGES)
```

---

## Implementation Order

The three pillars are interdependent. Follow this sequence:

**Week 1 (Stability + Structure)**
1. Fix the 5 runtime bugs (Pillar 1.1) — 1 hour total
2. Split `rag_pipeline.py` into modules (Pillar 3.1) — 4 hours
3. Centralized config + structured logging (Pillar 3.2, 3.3) — 2 hours
4. Clean `requirements.txt`, add Ruff (Pillar 3.5, 3.6) — 30 min

**Week 2 (Testing)**
5. Write unit tests with mocks (Pillar 1.2) — 4 hours
6. Write integration tests (Pillar 1.3) — 2 hours
7. Set up GitHub Actions CI (Pillar 1.6) — 1 hour
8. Add DeepEval RAG evaluation suite (Pillar 1.4) — 3 hours

**Week 3 (Performance)**
9. ChromaDB vector store (Pillar 2.3) — 3 hours
10. BM25 + RRF hybrid search (Pillar 2.1) — 2 hours
11. BGE cross-encoder reranking (Pillar 2.1) — 2 hours
12. HyDE query expansion (Pillar 2.2) — 1 hour
13. Parent-child chunking (Pillar 2.6) — 2 hours
14. Redis cache + SSE streaming (Pillar 2.4, 2.5) — 2 hours
15. SSRF protection + rate limiting (Pillar 3.7) — 1 hour
16. Load testing with Locust (Pillar 1.5) — 1 hour

**Total estimated effort: ~32 hours**

---

## Projected Scores After Implementation

|
 Category 
|
 Before 
|
 After 
|
|
---
|
---
|
---
|
|
 Architecture 
|
 3/10 
|
 9/10 
|
|
 Code Quality 
|
 2/10 
|
 9/10 
|
|
 Maintainability 
|
 2/10 
|
 8/10 
|
|
 Scalability 
|
 2/10 
|
 9/10 
|
|
 Performance 
|
 5/10 
|
 9/10 
|
|
 Security 
|
 4/10 
|
 9/10 
|
|
 Testing 
|
 0/10 
|
 9/10 
|
|
 RAG Quality 
|
 4/10 
|
 9/10 
|
|
**
Overall
**
|
**
~2.5/10
**
|
**
~9/10
**
|

---

*Synthesized from Strategy 1 (imrovements.md) + Strategy 2 (improvements2.md) against PROJECT_ANALYSIS.md — June 2026*
