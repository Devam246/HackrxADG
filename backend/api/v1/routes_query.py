import asyncio
import time
from fastapi import APIRouter, Depends, HTTPException
import structlog

from api.v1.deps import verify_token
from models.schemas import HackathonRequest, HackathonResponse
from services.generation.generator import handle_queries
from services.ingestion.chunker import get_doc_id, load_or_create_chunks
from services.retrieval.vector_store import build_inverted_index
from utils.cache import document_cache

logger = structlog.get_logger(__name__)
router = APIRouter()

MAX_QUESTIONS_PER_REQUEST = 50


@router.get("/cache-stats", tags=["System"])
async def cache_stats(_: None = Depends(verify_token)) -> dict:
    return {
        "cached_documents": len(document_cache),
        "cache_keys": list(document_cache.keys()),
        "memory_usage": f"{len(str(document_cache))} bytes (approx)",
    }


@router.post("/hackrx/run", response_model=HackathonResponse, tags=["RAG"])
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
        raise HTTPException(
            status_code=400,
            detail=f"Max {MAX_QUESTIONS_PER_REQUEST} questions allowed",
        )

    doc_id = get_doc_id(document_url)
    cache_key = doc_id
    logger.info(
        "rag_request_received",
        document_url=document_url,
        question_count=len(questions),
        doc_id=doc_id,
    )

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
        logger.info(
            "rag_request_completed",
            doc_id=doc_id,
            duration_seconds=round(total_time, 2),
        )
        return HackathonResponse(
            answers=answers,
            processing_time=round(total_time, 3),
            document_id=doc_id,
        )
    except Exception as exc:
        logger.exception("rag_request_failed", doc_id=doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error") from exc
