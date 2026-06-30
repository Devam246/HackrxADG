"""
Unit tests for the V9 agentic RAG state machine (agent/rag_graph.py).

All external I/O (Gemini, retrieval) is mocked so the graph logic is tested
in isolation without network calls.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.domain import Chunk


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_chunk(chunk_id: str, text: str = "Policy text about coverage and benefits.") -> Chunk:
    return Chunk(
        doc_id="doc1",
        text=text,
        section_id="1.1",
        section_title="Coverage",
        chunk_index=0,
        keywords=["coverage"],
        raw_text=text,
        section="1.1",
        page=1,
        parent_id="p1",
        chunk_id=chunk_id,
        is_parent=False,
        token_count=20,
        rerank_score=0.9,
    )


@pytest.fixture
def sample_state():
    """Minimal AgentState dict to bootstrap graph invocations."""
    chunks = [make_chunk("c1"), make_chunk("c2")]
    return dict(
        query="What is the waiting period?",
        original_query="What is the waiting period?",
        chunks=chunks,
        chunk_embeddings=np.zeros((2, 3072), dtype="float32"),
        inv_index={"waiting": {0}},
        doc_type="insurance",
        doc_id="doc1",
        top_k=8,
        retrieved_chunks=[],
        confidence=0.0,
        enough_context=False,
        rewrite_count=0,
        answer="",
        original_tokens=0,
        final_tokens=0,
        needs_human_review=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Direct path — retrieve → grade YES → generate
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_direct_path_sufficient_context(mocker, sample_state):
    """When grader says YES, the graph goes directly to generate without rewrite."""
    chunks = sample_state["chunks"]
    retrieved = [make_chunk("c1")]

    mocker.patch(
        "agent.rag_graph.generate_hypothetical_excerpt",
        return_value="Waiting period hypothetical excerpt",
    )
    mocker.patch(
        "agent.rag_graph.embed_text",
        return_value=[np.zeros(3072)],
    )
    mocker.patch(
        "agent.rag_graph.advanced_universal_retrieval",
        return_value=retrieved,
    )
    mocker.patch(
        "agent.rag_graph.calculate_universal_confidence",
        return_value=0.75,
    )
    # Grader returns YES
    mocker.patch(
        "agent.rag_graph._call_gemini_text",
        return_value="YES",
    )
    compressed = [make_chunk("c1", text="Waiting period is 36 months.")]
    mocker.patch(
        "agent.rag_graph.compress_chunks",
        return_value=compressed,
    )
    mocker.patch(
        "agent.rag_graph.build_universal_prompt",
        return_value=("system", "user"),
    )
    mocker.patch(
        "agent.rag_graph._batch_llm_answer",
        return_value="The waiting period is 36 months.",
    )

    from agent.rag_graph import run_agentic_rag

    answer, needs_review, _, _ = run_agentic_rag(**{
        k: sample_state[k]
        for k in ("query", "chunks", "chunk_embeddings", "inv_index", "doc_type", "doc_id", "top_k")
    })

    assert "36 months" in answer
    assert needs_review is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Rewrite path — retrieve → grade NO → rewrite → retrieve → grade YES → generate
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_rewrite_cycle_then_success(mocker, sample_state):
    """
    First retrieval returns insufficient context (grade NO),
    rewrite produces a new query, second retrieval is graded YES.
    """
    retrieved = [make_chunk("c1")]

    mocker.patch(
        "agent.rag_graph.generate_hypothetical_excerpt",
        return_value="hypothetical excerpt",
    )
    mocker.patch(
        "agent.rag_graph.embed_text",
        return_value=[np.zeros(3072)],
    )
    mocker.patch(
        "agent.rag_graph.advanced_universal_retrieval",
        return_value=retrieved,
    )
    mocker.patch(
        "agent.rag_graph.calculate_universal_confidence",
        return_value=0.55,
    )

    # Grader: first call → NO, second call (after rewrite) → YES
    # Rewriter call is a separate call to _call_gemini_text; distinguish by call count
    call_count = {"n": 0}
    def _gemini_side_effect(prompt, **kwargs):
        if "Respond with only: YES or NO" in prompt:
            call_count["n"] += 1
            return "YES" if call_count["n"] > 1 else "NO"
        # rewrite prompt
        return "What is the deferment period for coverage?"

    mocker.patch("agent.rag_graph._call_gemini_text", side_effect=_gemini_side_effect)

    compressed = [make_chunk("c1", text="Deferment period is 2 years.")]
    mocker.patch("agent.rag_graph.compress_chunks", return_value=compressed)
    mocker.patch("agent.rag_graph.build_universal_prompt", return_value=("system", "user"))
    mocker.patch("agent.rag_graph._batch_llm_answer", return_value="The deferment period is 2 years.")

    from agent.rag_graph import run_agentic_rag

    answer, needs_review, _, _ = run_agentic_rag(**{
        k: sample_state[k]
        for k in ("query", "chunks", "chunk_embeddings", "inv_index", "doc_type", "doc_id", "top_k")
    })

    assert "deferment" in answer.lower() or "period" in answer.lower()
    # Confidence was >= 0.4 so no human review needed
    assert needs_review is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Exhausted retries → needs_human_review = True
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_exhausted_rewrites_triggers_human_review(mocker, sample_state):
    """
    Grader always says NO and rewrite budget is exhausted (2 rewrites).
    The graph must eventually call generate with needs_human_review=True.
    """
    retrieved = [make_chunk("c1"), make_chunk("c2")]

    mocker.patch(
        "agent.rag_graph.generate_hypothetical_excerpt",
        return_value="hypothetical excerpt",
    )
    mocker.patch(
        "agent.rag_graph.embed_text",
        return_value=[np.zeros(3072)],
    )
    mocker.patch(
        "agent.rag_graph.advanced_universal_retrieval",
        return_value=retrieved,
    )
    # Confidence is above 0.25 (so we don't short-circuit) but below 0.4 (triggers review)
    mocker.patch(
        "agent.rag_graph.calculate_universal_confidence",
        return_value=0.30,
    )

    def _gemini_side_effect(prompt, **kwargs):
        if "Respond with only: YES or NO" in prompt:
            return "NO"   # grader always says NO
        return "Rewritten query for insurance terminology"

    mocker.patch("agent.rag_graph._call_gemini_text", side_effect=_gemini_side_effect)

    compressed = [make_chunk("c1", text="Best available answer."), make_chunk("c2")]
    mocker.patch("agent.rag_graph.compress_chunks", return_value=compressed)
    mocker.patch("agent.rag_graph.build_universal_prompt", return_value=("system", "user"))
    mocker.patch("agent.rag_graph._batch_llm_answer", return_value="Best effort answer.")

    from agent.rag_graph import run_agentic_rag

    answer, needs_review, _, _ = run_agentic_rag(**{
        k: sample_state[k]
        for k in ("query", "chunks", "chunk_embeddings", "inv_index", "doc_type", "doc_id", "top_k")
    })

    assert needs_review is True
