"""
agent/rag_graph.py — V9 Agentic RAG using LangGraph

State machine: Retrieve → Grade → (Rewrite → Retrieve)* → Generate
Max 2 rewrite cycles before forcing generation with needs_human_review=True.
"""

from typing import List, Optional, TypedDict

import google.api_core.exceptions
import google.generativeai as genai
import numpy as np
import re
import structlog
import time

from langgraph.graph import StateGraph, END

from config import get_settings
from models.domain import Chunk
from services.retrieval.compressor import compress_chunks
from services.retrieval.embedder import embed_text
from services.retrieval.hyde import generate_hypothetical_excerpt
from services.retrieval.vector_store import advanced_universal_retrieval
from services.generation.postprocessor import calculate_universal_confidence
from services.generation.prompts import build_universal_prompt

logger = structlog.get_logger(__name__)
settings = get_settings()

MAX_REWRITE_CYCLES = 2
HUMAN_REVIEW_CONFIDENCE_THRESHOLD = 0.4

# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Typed state dict passed between LangGraph nodes."""
    query: str                          # current (possibly rewritten) query
    original_query: str                 # original user query, never mutated
    chunks: List[Chunk]                 # all doc chunks available for search
    chunk_embeddings: np.ndarray        # corresponding dense embeddings
    inv_index: dict                     # inverted keyword index
    doc_type: str
    doc_id: str
    top_k: int                          # rerank_top_n setting
    retrieved_chunks: List[Chunk]       # result of current retrieval pass
    confidence: float
    enough_context: bool
    rewrite_count: int
    answer: str
    original_tokens: int
    final_tokens: int
    needs_human_review: bool


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini_text(prompt: str, temperature: float = 0.0,
                      max_output_tokens: int = 200) -> Optional[str]:
    """Low-level helper: call Gemini and return raw text, or None on failure."""
    backoff = 2.0
    for attempt in range(3):
        try:
            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel(settings.llm_model)
            response = model.generate_content(
                contents=[prompt],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            return None
        except google.api_core.exceptions.ResourceExhausted as e:
            if attempt == 2:
                logger.exception("gemini_rate_limit_exhausted", error=str(e))
                return None
            err_msg = str(e)
            match = re.search(r"Please retry in ([0-9.]+)s", err_msg)
            wait = float(match.group(1)) + 1.5 if match else backoff
            backoff *= 2.0
            time.sleep(wait)
        except Exception as e:
            logger.exception("gemini_call_failed", error=str(e))
            return None
    return None


def _batch_llm_answer(system: str, user: str) -> str:
    """Wrapper calling the generation stack; returns a single answer string."""
    from services.generation.generator import batch_llm_answer
    answers = batch_llm_answer(system, user, max_output_tokens=2000)
    return answers[0] if answers else "Error: Could not generate response"


# ─────────────────────────────────────────────────────────────────────────────
# Node: retrieve
# ─────────────────────────────────────────────────────────────────────────────

def node_retrieve(state: AgentState) -> AgentState:
    """HyDE expansion → dense embed → hybrid retrieval (BM25+Dense+RRF+Rerank)."""
    query = state["query"]
    doc_id = state["doc_id"]

    logger.info("agent_node_retrieve", query=query, doc_id=doc_id,
                rewrite_count=state["rewrite_count"])

    hyde_query = generate_hypothetical_excerpt(query)
    query_embedding = embed_text([hyde_query], task_type="retrieval_query")[0]

    retrieved = advanced_universal_retrieval(
        query_embedding,
        state["chunks"],
        state["chunk_embeddings"],
        state["inv_index"],
        query,
        state["doc_type"],
        doc_id,
        initial_k=20,
        final_k=state["top_k"],
    )

    confidence = calculate_universal_confidence(query, retrieved, state["doc_type"])

    return {
        **state,
        "retrieved_chunks": retrieved,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: grade_context
# ─────────────────────────────────────────────────────────────────────────────

GRADER_PROMPT = (
    "Does the following retrieved context contain enough information to "
    "answer this insurance question accurately?\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Respond with only: YES or NO"
)


def node_grade_context(state: AgentState) -> AgentState:
    """LLM-based grader: determines if retrieved context is sufficient."""
    confidence = state["confidence"]

    # Fast path: confidence too low → skip LLM call to save quota
    if confidence < 0.25:
        logger.info("agent_grade_skip_low_confidence", confidence=confidence)
        return {**state, "enough_context": False}

    context_text = "\n\n".join(c.text[:800] for c in state["retrieved_chunks"])
    prompt = GRADER_PROMPT.format(
        question=state["original_query"],
        context=context_text,
    )

    raw = _call_gemini_text(prompt, temperature=0.0, max_output_tokens=5)
    verdict = (raw or "NO").strip().upper()
    enough = verdict.startswith("YES")

    logger.info("agent_grade_context", confidence=confidence,
                verdict=verdict, enough_context=enough)

    return {**state, "enough_context": enough}


# ─────────────────────────────────────────────────────────────────────────────
# Node: rewrite
# ─────────────────────────────────────────────────────────────────────────────

REWRITE_PROMPT = (
    "You are an expert in insurance policy language.\n"
    "The following question did not retrieve sufficient context. "
    "Rewrite it using alternative insurance terminology "
    "(e.g. replace 'covered' with 'included', 'waiting period' with 'deferment period', "
    "'claim' with 'reimbursement request', etc.) to improve retrieval.\n"
    "Return ONLY the rewritten question, no explanation.\n\n"
    "Original question: {question}"
)


def node_rewrite(state: AgentState) -> AgentState:
    """Reformulate the query using alternative insurance terminology."""
    rewrite_count = state["rewrite_count"] + 1
    logger.info("agent_node_rewrite", original_query=state["original_query"],
                rewrite_count=rewrite_count)

    prompt = REWRITE_PROMPT.format(question=state["query"])
    rewritten = _call_gemini_text(prompt, temperature=0.3, max_output_tokens=150)

    new_query = rewritten.strip() if rewritten else state["query"]
    logger.info("agent_rewrite_result", new_query=new_query)

    return {
        **state,
        "query": new_query,
        "rewrite_count": rewrite_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: generate
# ─────────────────────────────────────────────────────────────────────────────

def node_generate(state: AgentState) -> AgentState:
    """Compress context and call LLM to produce the final answer."""
    confidence = state["confidence"]
    enough_context = state["enough_context"]
    rewrite_count = state["rewrite_count"]

    # Trigger human review if confidence is too low or retries exhausted without
    # sufficient context
    needs_review = (confidence < HUMAN_REVIEW_CONFIDENCE_THRESHOLD) or \
                   (not enough_context and rewrite_count >= MAX_REWRITE_CYCLES)

    # Fall back immediately when confidence is critically low
    if confidence < 0.25:
        return {
            **state,
            "answer": "Insufficient information to answer this question.",
            "needs_human_review": True,
            "original_tokens": 0,
            "final_tokens": 0,
        }

    # Context compression: top 8 reranked → compressed → top 5
    final_chunks = compress_chunks(state["original_query"], state["retrieved_chunks"])
    orig_tokens = getattr(final_chunks, "_original_tokens", 0)
    final_tokens_count = getattr(final_chunks, "_final_tokens", 0)

    system, user_prompt = build_universal_prompt(
        [state["original_query"]],
        [final_chunks],
        [confidence],
        state["doc_type"],
        snippet_len=2000,
    )
    answer = _batch_llm_answer(system, user_prompt)

    logger.info("agent_node_generate", confidence=confidence,
                needs_human_review=needs_review, rewrite_count=rewrite_count)

    return {
        **state,
        "answer": answer,
        "needs_human_review": needs_review,
        "original_tokens": orig_tokens,
        "final_tokens": final_tokens_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def route_after_grade(state: AgentState) -> str:
    """Return next node name based on grading result and rewrite budget."""
    if state["enough_context"]:
        return "generate"
    if state["rewrite_count"] < MAX_REWRITE_CYCLES:
        return "rewrite"
    # Exhausted rewrite budget — go to generate with needs_human_review flag
    return "generate"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)

    g.add_node("retrieve", node_retrieve)
    g.add_node("grade_context", node_grade_context)
    g.add_node("rewrite", node_rewrite)
    g.add_node("generate", node_generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_context")
    g.add_conditional_edges(
        "grade_context",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)

    return g.compile()


_graph = _build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_agentic_rag(
    query: str,
    chunks: List[Chunk],
    chunk_embeddings: np.ndarray,
    inv_index: dict,
    doc_type: str,
    doc_id: str,
    top_k: int = settings.rerank_top_n,
) -> tuple[str, bool, int, int]:
    """
    Run the full V9 agentic RAG pipeline for a single query.

    Returns:
        (answer, needs_human_review, original_tokens, final_tokens)
    """
    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "inv_index": inv_index,
        "doc_type": doc_type,
        "doc_id": doc_id,
        "top_k": top_k,
        "retrieved_chunks": [],
        "confidence": 0.0,
        "enough_context": False,
        "rewrite_count": 0,
        "answer": "",
        "original_tokens": 0,
        "final_tokens": 0,
        "needs_human_review": False,
    }

    final_state = _graph.invoke(initial_state)

    return (
        final_state["answer"],
        final_state["needs_human_review"],
        final_state["original_tokens"],
        final_state["final_tokens"],
    )
