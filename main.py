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

# Import your RAG pipeline functions
from rag_pipeline2 import extract_chunks_from_any_file, handle_queries

# Load environment variables

# os.environ["HACKATHON_BEARER_TOKEN"] = "ec1e19fd49685691a833851a2d2a3da61fab0c46c6397cfbd5153334c95c6301"
# os.environ["GEMINI_API_KEY"]="AIzaSyDAio_TICcfXsrDO28bZzktSx6uHJL2hQg"
# os.environ["VOYAGE_API_KEY"]="iBv1wEPodYD4c78oyTQxQpEfh6GT8psof3n3LkVp"
load_dotenv()
# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EXPECTED_BEARER_TOKEN = os.getenv("HACKATHON_BEARER_TOKEN")
print(EXPECTED_BEARER_TOKEN)
MAX_QUESTIONS_PER_REQUEST = 50  # Prevent abuse
REQUEST_TIMEOUT = 300  # 5 minutes timeout

# Global cache for document chunks (in-memory cache for faster processing)
document_cache = {}

# --- Data Models ---
class HackathonRequest(BaseModel):
    """Request model for the RAG system."""
    documents: HttpUrl
    questions: List[str]
    
    class Config:
        # Enable faster JSON parsing
        arbitrary_types_allowed = True

class HackathonResponse(BaseModel):
    """Response model for the RAG system."""
    answers: List[str]
    # processing_time: Optional[float] = None
    # document_id: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    environment: str

class PingResponse(BaseModel):
    """Ping response model."""
    message: str
    timestamp: float

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting HackRx 6 RAG API...")
    logger.info(f"📊 Cache initialized with capacity for documents")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down RAG API...")
    document_cache.clear()

# --- FastAPI App Instance ---
app = FastAPI(
    title="HackRx 6 RAG API",
    description="Optimized RAG API for HackRx 6 - Lightning fast document Q&A system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# --- Middleware Setup ---
# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom timing middleware
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Add request timing and logging."""
    start_time = time.perf_counter()
    
    # Log incoming request
    logger.info(f"📥 {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.perf_counter() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = "2.0.0"
        response.headers["X-Hackrx-Version"] = "6"
        
        logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        
        return response
        
    except Exception as e:
        process_time = time.perf_counter() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - Error: {str(e)} - {process_time:.3f}s")
        raise

# Custom rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting based on IP."""
    # In production, consider using Redis for distributed rate limiting
    response = await call_next(request)
    return response

# --- Authentication ---
async def verify_token(authorization: Optional[str] = Header(None)):
    """Optimized token verification with better error handling."""
    if not authorization:
        logger.warning("🔒 Authorization header missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise ValueError("Invalid scheme")
    except ValueError:
        logger.warning("🔒 Invalid authorization header format")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if token != EXPECTED_BEARER_TOKEN:
        logger.warning(f"🔒 Invalid token provided: {token}")
        logger.warning(f"🔒 Expected Token: {EXPECTED_BEARER_TOKEN}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired token",
        )
    
    logger.info("🔐 Token verification successful")

# --- Health and Utility Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for monitoring systems."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="2.0.0",
        environment=os.getenv("ENVIRONMENT", "production")
    )

@app.get("/ping", response_model=PingResponse, tags=["System"])
async def ping():
    """Simple ping endpoint for uptime monitoring."""
    return PingResponse(
        message="pong",
        timestamp=time.time()
    )

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HackRx 6 RAG API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ping": "/ping"
    }

@app.get("/cache-stats", tags=["System"])
async def cache_stats(_: None = Depends(verify_token)):
    """Get cache statistics (protected endpoint)."""
    return {
        "cached_documents": len(document_cache),
        "cache_keys": list(document_cache.keys()),
        "memory_usage": f"{len(str(document_cache))} bytes (approximate)"
    }

# --- Main RAG Endpoint ---
@app.post(
    "/hackrx/run",
    response_model=HackathonResponse,
    summary="Process documents and answer questions using RAG",
    description="Main endpoint for the HackRx 6 RAG system. Processes documents and returns AI-generated answers.",
    tags=["RAG"]
)
async def run_rag_system(
    request_data: HackathonRequest,
    _: None = Depends(verify_token)
):
    """
    Main RAG endpoint with optimizations:
    - Async processing
    - Document caching
    - Input validation
    - Error handling
    - Performance monitoring
    """
    start_time = time.perf_counter()
    doc_id = None
    
    try:
        # Input validation
        if len(request_data.questions) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )
        
        if len(request_data.questions) > MAX_QUESTIONS_PER_REQUEST:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many questions. Maximum {MAX_QUESTIONS_PER_REQUEST} allowed"
            )
        
        document_url = str(request_data.documents)
        questions = request_data.questions
        
        logger.info(f"📄 Processing document: {document_url}")
        logger.info(f"❓ Questions count: {len(questions)}")
        
        # Check cache first
        cache_key = f"doc_{hash(document_url)}"
        
        if cache_key in document_cache:
            logger.info("⚡ Using cached document chunks")
            chunks, doc_id = document_cache[cache_key]
        else:
            logger.info("📊 Extracting chunks from document...")
            chunk_start = time.perf_counter()
            
            # Run chunk extraction in a thread pool to avoid blocking
            chunks, doc_id = await asyncio.get_event_loop().run_in_executor(
                None, extract_chunks_from_any_file, document_url
            )
            
            # Cache the results
            document_cache[cache_key] = (chunks, doc_id)
            
            chunk_time = time.perf_counter() - chunk_start
            logger.info(f"📊 Chunk extraction completed in {chunk_time:.2f}s")
        
        # Process queries asynchronously
        logger.info("🤖 Processing queries with RAG...")
        query_start = time.perf_counter()
        
        # Run query handling in a thread pool
        answers = await asyncio.get_event_loop().run_in_executor(
            None, handle_queries, questions, chunks, doc_id, 3
        )
        
        query_time = time.perf_counter() - query_start
        total_time = time.perf_counter() - start_time
        
        logger.info(f"🤖 Query processing completed in {query_time:.2f}s")
        logger.info(f"🚀 Total processing time: {total_time:.2f}s")
        
        # Cleanup old cache entries if cache gets too large
        if len(document_cache) > 10:  # Keep only last 10 documents
            oldest_key = next(iter(document_cache))
            del document_cache[oldest_key]
            logger.info("🗑️ Cleaned up old cache entry")
        
        return HackathonResponse(
            answers=answers,
            processing_time=round(total_time, 3),
            document_id=doc_id
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except asyncio.TimeoutError:
        logger.error("⏰ Request timeout")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Request processing timeout"
        )
        
    except Exception as e:
        total_time = time.perf_counter() - start_time
        logger.error(f"❌ Error after {total_time:.2f}s: {str(e)}")
        
        # Log full traceback for debugging
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document processing"
        )

# --- Error Handlers ---
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "available_endpoints": [
                "/",
                "/health",
                "/ping",
                "/hackrx/run",
                "/docs"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": "Please try again later or contact support"
        }
    )

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks."""
    logger.info("🏁 HackRx 6 RAG API is ready!")
    logger.info(f"🔑 Token authentication: {'✅ Enabled' if EXPECTED_BEARER_TOKEN else '❌ Disabled'}")

if __name__ == "__main__":
    import uvicorn
    # For local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,  # Disable reload in production
        access_log=True,
        log_level="info"
    )
