from contextlib import asynccontextmanager
import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import structlog

from api.v1.routes_health import router as health_router
from api.v1.routes_query import router as query_router
from utils.cache import document_cache
from utils.logging import configure_logging

# Configure logging at startup
configure_logging()
logger = structlog.get_logger(__name__)


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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend index.html not found</h1>", status_code=404)


# Include Routers
app.include_router(health_router)
app.include_router(query_router)

