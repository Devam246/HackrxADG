from concurrent.futures import ThreadPoolExecutor
import re
import time
from typing import List, Optional
import google.generativeai as genai
import google.api_core.exceptions
import structlog
import tiktoken

from config import get_settings
from models.domain import Chunk

logger = structlog.get_logger(__name__)

COMPRESSION_PROMPT = """Extract ONLY the sentences from the policy excerpt below that directly 
answer: "{query}"
If none are relevant, return "NO_RELEVANT_CONTENT".

Policy excerpt:
{chunk_text}"""


def count_tokens(text: str) -> int:
    """Helper to count tokens in a string using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to a rough word-based count if tiktoken fails
        return len(text.split())


def compress_single_chunk(query: str, chunk: Chunk, model) -> Optional[Chunk]:
    """Compresses a single chunk using the LLM with rate limiting and retry handling."""
    backoff = 2.0
    max_attempts = 3
    prompt = COMPRESSION_PROMPT.format(query=query, chunk_text=chunk.text)

    for attempt in range(max_attempts):
        try:
            response = model.generate_content(
                contents=[prompt],
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 300,
                },
            )

            # Safeguard if response has no valid text candidates
            if not response.candidates or not response.candidates[0].content.parts:
                return None

            content = response.candidates[0].content.parts[0].text
            if not content:
                return None

            clean_res = content.strip()
            # If the output indicates no relevant content, return None (discard)
            if "NO_RELEVANT_CONTENT" in clean_res or clean_res.upper() == "NO_RELEVANT_CONTENT":
                return None

            # Copy the chunk and update its text
            chunk_copy = chunk.copy()
            chunk_copy.text = clean_res
            return chunk_copy

        except google.api_core.exceptions.ResourceExhausted as e:
            if attempt == max_attempts - 1:
                logger.exception("compression_rate_limit_failed", chunk_id=chunk.chunk_id, error=str(e))
                return None

            # Try to parse wait time from exception if provided
            err_msg = str(e)
            match = re.search(r"Please retry in ([0-9.]+)s", err_msg)
            wait_seconds = float(match.group(1)) + 1.5 if match else backoff
            backoff *= 2.0

            logger.warning(
                "compression_rate_limit_retry",
                chunk_id=chunk.chunk_id,
                attempt=attempt + 1,
                wait_seconds=round(wait_seconds, 2),
            )
            time.sleep(wait_seconds)

        except Exception as e:
            logger.exception("compression_failed", chunk_id=chunk.chunk_id, error=str(e))
            return None

    return None


def compress_chunks(query: str, chunks: List[Chunk]) -> List[Chunk]:
    """
    Compresses a list of chunks down by extracting only relevant sentences.
    Processes chunks in parallel, logging the resulting token count reduction.
    Falls back to the first 2 original chunks if fewer than 2 compressed chunks survive.
    """
    if not chunks:
        return []

    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.llm_model)

    original_tokens = sum(count_tokens(c.text) for c in chunks)

    max_workers = min(len(chunks), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(compress_single_chunk, query, chunk, model)
            for chunk in chunks
        ]
        results = [f.result() for f in futures]

    compressed_chunks = [c for c in results if c is not None]

    if len(compressed_chunks) >= 2:
        final_chunks = compressed_chunks[:5]
        fallback_used = False
    else:
        # Fall back to original parent chunks (minimum 2 chunks, or all if fewer exist)
        final_chunks = chunks[:2]
        fallback_used = True

    final_tokens = sum(count_tokens(c.text) for c in final_chunks)
    reduction_pct = (1.0 - (final_tokens / original_tokens)) * 100 if original_tokens > 0 else 0.0

    logger.info(
        "context_compression_completed",
        query=query,
        original_chunks=len(chunks),
        compressed_chunks=len(compressed_chunks),
        final_chunks=len(final_chunks),
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        reduction_percentage=round(reduction_pct, 2),
        fallback_used=fallback_used,
    )

    # Attach token stats helper info to the returned list for average tracking
    # We can attach attributes to list objects in python
    try:
        setattr(final_chunks, "_original_tokens", original_tokens)
        setattr(final_chunks, "_final_tokens", final_tokens)
    except Exception:
        pass

    return final_chunks
