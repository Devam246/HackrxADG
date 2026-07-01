from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
from typing import List, Optional
import groq
import structlog
import tiktoken

from config import get_settings
from models.domain import Chunk

logger = structlog.get_logger(__name__)
settings = get_settings()

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
        return len(text.split())


def compress_single_chunk(query: str, chunk: Chunk, client) -> Optional[Chunk]:
    """Compresses a single chunk using Groq with rate limiting and retry handling."""
    backoff = 2.0
    max_attempts = 3
    prompt = COMPRESSION_PROMPT.format(query=query, chunk_text=chunk.text)

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300,
            )

            if not response.choices or not response.choices[0].message.content:
                return None

            content = response.choices[0].message.content.strip()
            if "NO_RELEVANT_CONTENT" in content or content.upper() == "NO_RELEVANT_CONTENT":
                return None

            chunk_copy = chunk.copy()
            chunk_copy.text = content
            return chunk_copy

        except groq.RateLimitError as e:
            if attempt == max_attempts - 1:
                logger.exception("compression_rate_limit_failed", chunk_id=chunk.chunk_id, error=str(e))
                return None

            err_msg = str(e)
            wait_seconds = backoff
            match = re.search(r"(?:retry|try again) in ([0-9.]+)(m?s)", err_msg, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if unit == "ms":
                    wait_seconds = (value / 1000.0) + 1.5
                else:
                    wait_seconds = value + 1.5
            else:
                time_str_match = re.search(r"in\s+([0-9hm\.]+s)", err_msg, re.IGNORECASE)
                if time_str_match:
                    time_str = time_str_match.group(1).rstrip(".")
                    if "m" in time_str:
                        parts = time_str.split("m")
                        minutes = float(parts[0])
                        seconds = float(parts[1].replace("s", ""))
                        wait_seconds = minutes * 60.0 + seconds + 1.5
                    else:
                        seconds = float(time_str.replace("s", ""))
                        wait_seconds = seconds + 1.5
                else:
                    backoff *= 2.0

            logger.warning(
                "compression_rate_limit_retry",
                chunk_id=chunk.chunk_id,
                attempt=attempt + 1,
                wait_seconds=round(wait_seconds, 2),
            )
            time.sleep(wait_seconds)

        except Exception as e:
            if getattr(e, 'status_code', None) == 429:
                if attempt == max_attempts - 1:
                    logger.exception("compression_rate_limit_failed", chunk_id=chunk.chunk_id, error=str(e))
                    return None
                time.sleep(backoff)
                backoff *= 2.0
                continue

            logger.exception("compression_failed", chunk_id=chunk.chunk_id, error=str(e))
            return None

    return None


def compress_chunks_batched(query: str, chunks: List[Chunk], client) -> List[Optional[Chunk]]:
    """Compresses all chunks in a single batched API call to Groq."""
    backoff = 2.0
    max_attempts = 3

    # Format the prompt
    prompt = f'Compress the following policy excerpts to answer the query: "{query}"\n\n'
    prompt += 'For each excerpt, extract ONLY the sentences that directly answer the query. If none are relevant, return "NO_RELEVANT_CONTENT" for that excerpt.\n\n'
    prompt += 'You MUST output your response strictly as a JSON array of objects with the keys "index" and "compressed_text". Do not include any other markdown formatting outside of the JSON block.\n\n'
    prompt += 'Example format:\n'
    prompt += '[\n  {"index": 0, "compressed_text": "extracted text"},\n  {"index": 1, "compressed_text": "NO_RELEVANT_CONTENT"}\n]\n\n'
    prompt += 'Excerpts to compress:\n'
    
    for idx, chunk in enumerate(chunks):
        prompt += f'--- [Index {idx}] ---\n{chunk.text}\n\n'

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()
            # Clean JSON markdown if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()

            import json
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "results" in parsed:
                parsed = parsed["results"]
            elif isinstance(parsed, dict) and "excerpts" in parsed:
                parsed = parsed["excerpts"]
            elif isinstance(parsed, dict) and "answers" in parsed:
                parsed = parsed["answers"]
                
            if not isinstance(parsed, list):
                raise ValueError("Expected a list of compressed excerpts")

            # Match them back by index
            results = [None] * len(chunks)
            for item in parsed:
                idx = int(item.get("index", -1))
                comp_text = item.get("compressed_text", "").strip()
                if 0 <= idx < len(chunks) and comp_text:
                    if "NO_RELEVANT_CONTENT" in comp_text or comp_text.upper() == "NO_RELEVANT_CONTENT":
                        results[idx] = None
                    else:
                        chunk_copy = chunks[idx].copy()
                        chunk_copy.text = comp_text
                        results[idx] = chunk_copy

            logger.info("batched_compression_success", original_chunks=len(chunks), parsed_items=len(parsed))
            return results

        except groq.RateLimitError as e:
            if attempt == max_attempts - 1:
                logger.exception("batched_compression_rate_limit_failed", error=str(e))
                return [None] * len(chunks)

            err_msg = str(e)
            wait_seconds = backoff
            match = re.search(r"(?:retry|try again) in ([0-9.]+)(m?s)", err_msg, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if unit == "ms":
                    wait_seconds = (value / 1000.0) + 1.5
                else:
                    wait_seconds = value + 1.5
            else:
                time_str_match = re.search(r"in\s+([0-9hm\.]+s)", err_msg, re.IGNORECASE)
                if time_str_match:
                    time_str = time_str_match.group(1).rstrip(".")
                    if "m" in time_str:
                        parts = time_str.split("m")
                        minutes = float(parts[0])
                        seconds = float(parts[1].replace("s", ""))
                        wait_seconds = minutes * 60.0 + seconds + 1.5
                    else:
                        seconds = float(time_str.replace("s", ""))
                        wait_seconds = seconds + 1.5
                else:
                    backoff *= 2.0

            logger.warning(
                "compression_rate_limit_retry",
                chunk_id="BATCHED",
                attempt=attempt + 1,
                wait_seconds=round(wait_seconds, 2),
            )
            time.sleep(wait_seconds)

        except Exception as e:
            if getattr(e, 'status_code', None) == 429:
                if attempt == max_attempts - 1:
                    logger.exception("batched_compression_rate_limit_failed", error=str(e))
                    return [None] * len(chunks)
                time.sleep(backoff)
                backoff *= 2.0
                continue

            logger.exception("batched_compression_failed", error=str(e))
            return [None] * len(chunks)

    return [None] * len(chunks)


def compress_chunks(query: str, chunks: List[Chunk]) -> List[Chunk]:
    """
    Compresses a list of chunks down by extracting only relevant sentences.
    Attempts a single batched Groq call, falling back to individual calls if needed.
    """
    if not chunks:
        return []

    api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("groq_api_key_missing_compress_fallback")
        return chunks[:2]

    client = groq.Groq(api_key=api_key)
    original_tokens = sum(count_tokens(c.text) for c in chunks)

    # Attempt batched compression
    try:
        results = compress_chunks_batched(query, chunks, client)
        if all(r is None for r in results) and len(chunks) > 0:
            logger.warning("batched_compression_returned_all_none_trying_fallback")
            results = None
    except Exception as e:
        logger.warning("batched_compression_exception_trying_fallback", error=str(e))
        results = None

    # Fallback to individual chunk compression if batched failed or returned nothing
    if results is None:
        max_workers = min(len(chunks), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(compress_single_chunk, query, chunk, client)
                for chunk in chunks
            ]
            results = [f.result() for f in futures]

    compressed_chunks = [c for c in results if c is not None]

    if len(compressed_chunks) >= 2:
        final_chunks = compressed_chunks[:5]
        fallback_used = False
    else:
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

    try:
        setattr(final_chunks, "_original_tokens", original_tokens)
        setattr(final_chunks, "_final_tokens", final_tokens)
    except Exception:
        pass

    return final_chunks

