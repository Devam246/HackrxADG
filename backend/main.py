import asyncio
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import List, Optional

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, HttpUrl

from config import get_settings
from rag_pipeline import build_inverted_index, get_doc_id, handle_queries, load_or_create_chunks

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger(__name__)
settings = get_settings()

EXPECTED_BEARER_TOKEN = settings.bearer_token
MAX_QUESTIONS_PER_REQUEST = 50


class DocumentCache(OrderedDict):
    def __init__(self, capacity: int = 10) -> None:
        super().__init__()
        self.capacity = capacity

    def get(self, key: str, default=None):
        if key in self:
            self.move_to_end(key)
            return super().get(key)
        return default

    def set(self, key: str, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)


document_cache = DocumentCache(capacity=10)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_starting", service="hackrx_rag_api")
    yield
    logger.info("app_stopping", service="hackrx_rag_api")
    document_cache.clear()


app = FastAPI(
    title="HackRx 6 RAG API",
    description="Optimized RAG API for HackRx 6 - Lightning fast document Q&A system",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

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
    logger.info("request_started", method=request.method, path=request.url.path)
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("request_failed", error=str(exc))
        raise

    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-API-Version"] = "3.0.0"
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        duration_seconds=round(process_time, 3),
    )
    return response


async def verify_token(authorization: Optional[str] = Header(None)) -> None:
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


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="3.0.0",
        environment=settings.environment,
    )


@app.get("/ping", response_model=PingResponse, tags=["System"])
async def ping() -> PingResponse:
    return PingResponse(message="pong", timestamp=time.time())


@app.get("/", tags=["System"])
async def root() -> dict:
    return {
        "message": "HackRx 6 RAG API",
        "version": "3.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ping": "/ping",
    }


@app.get("/cache-stats", tags=["System"])
async def cache_stats(_: None = Depends(verify_token)) -> dict:
    return {
        "cached_documents": len(document_cache),
        "cache_keys": list(document_cache.keys()),
        "memory_usage": f"{len(str(document_cache))} bytes (approx)",
    }


@app.post("/hackrx/run", response_model=HackathonResponse, tags=["RAG"])
async def run_rag_system(
    request_data: HackathonRequest,
    _: None = Depends(verify_token),
) -> HackathonResponse:
    start_time = time.perf_counter()
    document_url = str(request_data.documents)
    questions = request_data.questions

    if not questions:
        raise HTTPException(status_code=400, detail="At least one question is required")
    if len(questions) > MAX_QUESTIONS_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Max {MAX_QUESTIONS_PER_REQUEST} questions allowed")

    doc_id = get_doc_id(document_url)
    cache_key = doc_id
    logger.info("rag_request_received", document_url=document_url, question_count=len(questions), doc_id=doc_id)

    try:
        cache_entry = document_cache.get(cache_key)
        if cache_entry:
            logger.info("document_cache_hit", doc_id=doc_id)
            chunks, chunk_embeddings, doc_id, doc_type, inv_index = cache_entry
        else:
            logger.info("document_loading", doc_id=doc_id)
            load_start = time.perf_counter()
            loop = asyncio.get_event_loop()
            chunks, chunk_embeddings, doc_id, doc_type = await loop.run_in_executor(
                None,
                load_or_create_chunks,
                document_url,
            )
            inv_index = build_inverted_index(chunks)
            document_cache.set(cache_key, (chunks, chunk_embeddings, doc_id, doc_type, inv_index))
            logger.info(
                "document_loaded",
                doc_id=doc_id,
                duration_seconds=round(time.perf_counter() - load_start, 2),
            )

        logger.info("queries_started", doc_id=doc_id, question_count=len(questions))
        query_start = time.perf_counter()
        answers = await asyncio.get_event_loop().run_in_executor(
            None,
            handle_queries,
            questions,
            chunks,
            chunk_embeddings,
            inv_index,
            doc_type,
            doc_id,
            3,
        )

        total_time = time.perf_counter() - start_time
        logger.info(
            "queries_completed",
            doc_id=doc_id,
            duration_seconds=round(time.perf_counter() - query_start, 2),
        )
        logger.info("rag_request_completed", doc_id=doc_id, duration_seconds=round(total_time, 2))
        return HackathonResponse(answers=answers, processing_time=round(total_time, 3), document_id=doc_id)
    except Exception as exc:
        logger.exception("rag_request_failed", doc_id=doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error") from exc
