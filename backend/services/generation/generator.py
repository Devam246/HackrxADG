from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import re
from typing import Dict, List, Tuple
import google.generativeai as genai
import numpy as np
import structlog

from config import get_settings
from services.generation.postprocessor import (
    calculate_universal_confidence,
    extract_answers_fallback,
    parse_json_response,
)
from services.generation.prompts import build_universal_prompt
from services.retrieval.embedder import embed_text
from services.retrieval.vector_store import advanced_universal_retrieval

logger = structlog.get_logger(__name__)
settings = get_settings()

# Configure Gemini
genai.configure(api_key=settings.gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")


def batch_llm_answer(system: str, user: str, max_output_tokens: int = 2048) -> List[str]:
    """
    Enhanced Gemini API call with robust JSON parsing, fallback mechanisms,
    and exponential backoff retry logic for 429 rate limit exceptions.
    """
    import google.api_core.exceptions
    import time
    import re

    max_retries = 3
    backoff = 2.0

    for attempt in range(max_retries):
        try:
            full_prompt = f"{system}\n\n{user}"

            response = gemini_model.generate_content(
                contents=[full_prompt],
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": max_output_tokens,
                    "top_p": 0.8,
                    "top_k": 40,
                },
            )

            if not response.candidates:
                raise Exception("No response candidates from Gemini")

            content = response.candidates[0].content.parts[0].text.strip()
            print("\n" + "="*80)
            print("[RAW GEMINI RESPONSE] (attempt {}):\n{}".format(attempt + 1, content))
            print("="*80 + "\n")
            parsed_answers = parse_json_response(content)

            if parsed_answers:
                return parsed_answers
            else:
                logger.info(
                    "legacy_log",
                    message=f"⚠️ Attempt {attempt + 1}: Failed to parse JSON, retrying...",
                )
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
        except google.api_core.exceptions.ResourceExhausted as e:
            if attempt == max_retries - 1:
                logger.exception("gemini_generation_failed_exhausted", error=str(e))
                return extract_answers_fallback(content if "content" in locals() else "")

            err_msg = str(e)
            match = re.search(r"Please retry in ([0-9.]+)s", err_msg)
            if match:
                wait_seconds = float(match.group(1)) + 1.5
            else:
                wait_seconds = backoff
                backoff *= 2.0

            logger.warning(
                "gemini_generation_rate_limit_hit_retrying",
                attempt=attempt + 1,
                wait_seconds=round(wait_seconds, 2),
                error=err_msg,
            )
            time.sleep(wait_seconds)
        except Exception as e:
            logger.info("legacy_log", message=f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.info("legacy_log", message="🔴 All attempts failed, using fallback extraction")
                return extract_answers_fallback(content if "content" in locals() else "")

            time.sleep(backoff)
            backoff *= 2.0

    return ["Error: Could not generate response"] * 10



def universal_query_expansion(query: str, doc_type: str) -> str:
    """Expand queries based on document type and universal patterns"""
    universal_expansions = {
        "what is": "definition explanation meaning description",
        "how": "procedure method process steps way",
        "when": "time period duration date timeline",
        "where": "location place position section part",
        "why": "reason purpose cause explanation",
        "which": "specification type kind category",
        "who": "person responsible authority entity",
    }

    domain_expansions = {
        "legal": {
            "article": "article section clause provision rule",
            "constitution": "constitution fundamental law basic principle",
            "right": "right freedom liberty privilege entitlement",
            "duty": "duty obligation responsibility requirement",
            "amendment": "amendment modification change revision",
            "court": "court tribunal judiciary legal authority",
            "law": "law statute act regulation rule provision",
        },
        "insurance": {
            "premium": "premium payment cost fee amount",
            "coverage": "coverage benefit protection indemnity",
            "claim": "claim settlement reimbursement payout",
            "waiting": "waiting period time duration months",
            "exclusion": "exclusion limitation restriction exception",
        },
        "technical": {
            "part": "part component element piece section",
            "system": "system mechanism apparatus device",
            "procedure": "procedure process method operation steps",
            "specification": "specification requirement standard parameter",
            "manual": "manual guide instruction handbook",
        },
        "academic": {
            "theorem": "theorem principle law rule formula",
            "principle": "principle concept theory fundamental law",
            "equation": "equation formula expression relation",
            "chapter": "chapter section part topic subject",
            "research": "research study investigation analysis",
        },
    }

    expanded = query.lower()

    for term, expansion in universal_expansions.items():
        if term in expanded:
            expanded += " " + expansion

    if doc_type in domain_expansions:
        for term, expansion in domain_expansions[doc_type].items():
            if term in expanded:
                expanded += " " + expansion

    return expanded


def enhanced_query_preprocessing(query: str, doc_type: str) -> Dict[str, str]:
    original_query = query.strip()

    numbers = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:days?|months?|years?|%|percent|rs|rupees?)\b",
        query.lower(),
    )

    entities = {
        "numbers": numbers,
        "timeframes": re.findall(r"\b(?:waiting|grace)\s+period\b", query.lower()),
        "coverage": re.findall(r"\b(?:cover|coverage|benefit|claim|premium)\w*\b", query.lower()),
        "medical": re.findall(
            r"\b(?:maternity|surgery|treatment|hospital|medical)\w*\b",
            query.lower(),
        ),
        "sections": re.findall(r"\b(?:section|article|clause|part)\s+\w+\b", query.lower()),
    }

    expansions = []
    if entities["numbers"]:
        expansions.extend(["period", "duration", "time", "limit"])
    if entities["coverage"]:
        expansions.extend(["eligible", "applicable", "conditions", "requirements"])
    if entities["medical"]:
        expansions.extend(["hospital", "treatment", "medical", "healthcare"])

    expanded = f"{original_query} {' '.join(expansions[:5])}"

    return {
        "original": original_query,
        "expanded": expanded,
        "entities": entities,
        "intent": classify_query_intent(original_query),
    }


def classify_query_intent(query: str) -> str:
    query_lower = query.lower()

    if any(word in query_lower for word in ["what is", "define", "definition"]):
        return "definition"
    elif any(word in query_lower for word in ["how much", "amount", "cost", "premium"]):
        return "amount"
    elif any(word in query_lower for word in ["waiting period", "how long", "duration"]):
        return "timeframe"
    elif any(word in query_lower for word in ["cover", "include", "eligible"]):
        return "coverage"
    elif any(word in query_lower for word in ["procedure", "process", "how to"]):
        return "procedure"
    else:
        return "general"


def smart_candidate_filtering(
    query_processed: Dict, chunks: List[Dict], inv_index: Dict, max_candidates: int = 30
) -> List[int]:
    query = query_processed["original"]
    entities = query_processed["entities"]
    intent = query_processed["intent"]

    candidate_sets = []

    if entities["numbers"]:
        number_candidates = set()
        for num_phrase in entities["numbers"]:
            for chunk_idx, chunk in enumerate(chunks):
                if any(num in chunk["text"].lower() for num in num_phrase.split()):
                    number_candidates.add(chunk_idx)
        if number_candidates:
            candidate_sets.append(number_candidates)

    intent_keywords = get_intent_keywords(intent, query)
    intent_candidates = set()
    for keyword in intent_keywords:
        if keyword in inv_index:
            intent_candidates.update(inv_index[keyword])
    if intent_candidates:
        candidate_sets.append(intent_candidates)

    original_candidates = set()
    query_words = re.findall(r"\b\w{3,}\b", query.lower())
    for word in query_words:
        if word in inv_index:
            original_candidates.update(inv_index[word])
    if original_candidates:
        candidate_sets.append(original_candidates)

    if len(candidate_sets) > 1:
        intersected = set.intersection(*candidate_sets)
        if len(intersected) >= 5:
            final_candidates = intersected
        else:
            weighted_candidates = defaultdict(int)
            for i, cand_set in enumerate(candidate_sets):
                weight = [3, 2, 1][i] if i < 3 else 1
                for cand in cand_set:
                    weighted_candidates[cand] += weight

            sorted_candidates = sorted(weighted_candidates.items(), key=lambda x: x[1], reverse=True)
            final_candidates = set([cand for cand, _ in sorted_candidates[:max_candidates]])
    else:
        final_candidates = candidate_sets[0] if candidate_sets else set(range(min(len(chunks), max_candidates)))

    return list(final_candidates)


def get_intent_keywords(intent: str, query: str) -> List[str]:
    base_words = re.findall(r"\b\w{4,}\b", query.lower())

    intent_expansions = {
        "definition": ["meaning", "defined", "refers", "means"],
        "amount": ["amount", "cost", "price", "fee", "charges"],
        "timeframe": ["period", "duration", "time", "months", "days", "years"],
        "coverage": ["covered", "includes", "eligible", "applicable"],
        "procedure": ["process", "steps", "procedure", "method"],
    }

    expanded = base_words + intent_expansions.get(intent, [])
    return expanded[:8]


def _process_single_query(
    query: str,
    chunks: List[Dict],
    chunk_embeddings: np.ndarray,
    inv_index: Dict,
    doc_type: str,
    doc_id: str,
    top_k: int,
) -> Tuple[str, bool, int, int]:
    """
    Process a single query using the V9 LangGraph agentic RAG pipeline.
    Returns (answer, needs_human_review, original_tokens, final_tokens).
    """
    from agent.rag_graph import run_agentic_rag

    return run_agentic_rag(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        inv_index=inv_index,
        doc_type=doc_type,
        doc_id=doc_id,
        top_k=top_k,
    )


def handle_queries(
    queries: List[str],
    chunks: List[Dict],
    chunk_embeddings: np.ndarray,
    inv_index: Dict,
    doc_type: str,
    doc_id: str,
    top_k: int = settings.rerank_top_n,
) -> Tuple[List[str], List[bool]]:
    """
    Handles multiple queries in parallel using the V9 agentic RAG pipeline.
    Returns (answers, needs_human_review_flags).
    """
    max_workers = min(len(queries), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_single_query,
                query,
                chunks,
                chunk_embeddings,
                inv_index,
                doc_type,
                doc_id,
                top_k,
            )
            for query in queries
        ]
        results_tuples = [f.result() for f in futures]

    answers = [r[0] for r in results_tuples]
    needs_review_flags = [r[1] for r in results_tuples]

    # Accumulate token stats for reduction logging
    total_orig = sum(r[2] for r in results_tuples)
    total_final = sum(r[3] for r in results_tuples)
    avg_reduction_pct = (1.0 - (total_final / total_orig)) * 100 if total_orig > 0 else 0.0

    logger.info(
        "request_context_compression_summary",
        question_count=len(queries),
        total_original_tokens=total_orig,
        total_final_tokens=total_final,
        average_reduction_percentage=round(avg_reduction_pct, 2),
        human_review_triggered=sum(needs_review_flags),
    )

    return answers, needs_review_flags

