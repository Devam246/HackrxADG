# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Optimized for Render deployment and HackRx 6 submission

import asyncio
import time
import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Header, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# ✅ Import updated pipeline functions
from rag_pipeline import (
    load_or_create_chunks,
    handle_queries,
    build_enhanced_inverted_index
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EXPECTED_BEARER_TOKEN = os.getenv("HACKATHON_BEARER_TOKEN")
MAX_QUESTIONS_PER_REQUEST = 50
REQUEST_TIMEOUT = 300

# ✅ In-memory cache to speed up repeated requests for the same document
# Structure: { cache_key: (chunks, chunk_embeddings, doc_id, doc_type, inv_index) }
# ✅ LRU in-memory cache (max 10 docs) to speed up repeated requests
from collections import OrderedDict

class DocumentCache(OrderedDict):
    def __init__(self, capacity=10):
        super().__init__()
        self.capacity = capacity
    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return super().get(key)
        return default
    def set(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)  # evict oldest

document_cache = DocumentCache(capacity=10)

# --- Data Models ---
class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]
    processing_time: Optional[float] = None
    document_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    environment: str

class PingResponse(BaseModel):
    message: str
    timestamp: float

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting HackRx 6 RAG API...")
    yield
    logger.info("🛑 Shutting down RAG API...")
    document_cache.clear()

# --- FastAPI App Instance ---
app = FastAPI(
    title="HackRx 6 RAG API",
    description="Optimized RAG API for HackRx 6 - Lightning fast document Q&A system",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    logger.info(f"📥 {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = "3.0.0"
        logger.info(f"✅ {request.method} {request.url.path} - {process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

# --- Authentication ---
async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise ValueError("Invalid scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    if token != EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or expired token")

# --- Utility Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="3.0.0",
        environment=os.getenv("ENVIRONMENT", "production")
    )

@app.get("/ping", response_model=PingResponse, tags=["System"])
async def ping():
    return PingResponse(message="pong", timestamp=time.time())

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "HackRx 6 RAG API",
        "version": "3.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ping": "/ping"
    }

@app.get("/cache-stats", tags=["System"])
async def cache_stats(_: None = Depends(verify_token)):
    return {
        "cached_documents": len(document_cache),
        "cache_keys": list(document_cache.keys()),
        "memory_usage": f"{len(str(document_cache))} bytes (approx)"
    }

# --- Main RAG Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse, tags=["RAG"])
async def run_rag_system(request_data: HackathonRequest, _: None = Depends(verify_token)):
    start_time = time.perf_counter()
    document_url = str(request_data.documents)
    questions = request_data.questions

    if len(questions) == 0:
        raise HTTPException(status_code=400, detail="At least one question is required")
    if len(questions) > MAX_QUESTIONS_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Max {MAX_QUESTIONS_PER_REQUEST} questions allowed")

    logger.info(f"📄 Document: {document_url}")
    logger.info(f"❓ Questions: {len(questions)}")

        # Use stable SHA256-based doc_id for cache key
    from rag_pipeline import get_doc_id
    doc_id = get_doc_id(document_url)
    cache_key = doc_id

    try:
         # ✅ Try LRU memory cache first
        cache_entry = document_cache.get(cache_key)
        if cache_entry:
            logger.info("⚡ Using in-memory cache")
            chunks, chunk_embeddings, doc_id, doc_type, inv_index = cache_entry

        else:
            logger.info("📊 Loading document (disk cache aware)...")
            load_start = time.perf_counter()

            # ✅ Load or create chunks + embeddings (disk cache aware)
            chunks, chunk_embeddings, doc_id, doc_type = await asyncio.get_event_loop().run_in_executor(
                None, load_or_create_chunks, document_url
            )

            # ✅ Build inverted index
            inv_index = build_enhanced_inverted_index(chunks)

             # ✅ Store in LRU memory cache
            document_cache.set(cache_key, (chunks, chunk_embeddings, doc_id, doc_type, inv_index))

            load_time = time.perf_counter() - load_start
            logger.info(f"📊 Document loaded in {load_time:.2f}s")

        # ✅ Process queries
        logger.info("🤖 Handling queries...")
        query_start = time.perf_counter()

        answers = await asyncio.get_event_loop().run_in_executor(
            None, handle_queries, questions, chunks, chunk_embeddings, inv_index, doc_type, doc_id, 3
        )

        query_time = time.perf_counter() - query_start
        total_time = time.perf_counter() - start_time

        logger.info(f"✅ Queries processed in {query_time:.2f}s")
        logger.info(f"🚀 Total time: {total_time:.2f}s")


        # (no manual eviction needed; DocumentCache enforces capacity)

        return HackathonResponse(answers=answers, processing_time=round(total_time, 3), document_id=doc_id)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
