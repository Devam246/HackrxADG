import time
from fastapi import APIRouter

from config import get_settings
from models.schemas import HealthResponse, PingResponse

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="3.0.0",
        environment=settings.environment,
    )


@router.get("/ping", response_model=PingResponse, tags=["System"])
async def ping() -> PingResponse:
    return PingResponse(message="pong", timestamp=time.time())


@router.get("/", tags=["System"])
async def root() -> dict:
    return {
        "message": "HackRx 6 RAG API",
        "version": "3.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ping": "/ping",
    }
