# RAG v9 — Agentic RAG (LangGraph)

**Date:** 2026-06-30
**Scope:** Add an intelligent multi-hop agentic retrieval layer using LangGraph. Instead of a fixed pipeline, the system can now recognize insufficient context, reformulate the query, and retry — returning a `needs_human_review` flag when retries are exhausted or confidence remains low.

---

## Goal

- Implement a LangGraph state machine with four nodes: Retrieve, Grade, Rewrite, Generate.
- Enable automatic query reformulation (max 2 cycles) when the retrieved context is insufficient.
- Surface `needs_human_review: true` on the API response for queries where confidence < 0.4 or all retries were exhausted without sufficient context.
- Keep all V8 improvements (cross-encoder reranking, context compression) active inside the agent.

---

## Previous Limitation (why this was needed)

The V8 pipeline was a fixed single-pass: retrieve → compress → generate. It could not detect a weak retrieval result and try again with different terminology. Vocabulary mismatch (e.g. "knee replacement" vs "orthopaedic surgery") would produce a low-confidence answer with no opportunity for self-correction.

---

## Concepts Introduced

1. **LangGraph StateGraph**: A directed graph over a typed `AgentState` dict. Each node transforms the state and returns it to the graph runner.
2. **Context Grader**: A deterministic LLM call (`temperature=0.0`) that checks whether retrieved context is sufficient for the query. Fast-path skips the LLM if `confidence < 0.25`.
3. **Query Rewriter**: A creativity-budget call (`temperature=0.3`) that reformulates the question using alternative insurance vocabulary.
4. **Human Review Flag**: `needs_human_review` is set to `True` in the API response when the system could not gather sufficient context even after maximum rewrite cycles, or when confidence is below 0.4.

---

## LangGraph Workflow

```
retrieve
    │
    ▼
grade_context
    │
    ├── enough_context = YES ──────────────────────────────────────────────────► generate
    │
    └── enough_context = NO ──┬── rewrite_count < 2 ── rewrite ── retrieve ──┘
                               │
                               └── rewrite_count >= 2 ──────────────────────────► generate (needs_human_review=True)
```

---

## Changed Files

| File | Changes |
|---|---|
| `backend/agent/__init__.py` | New agent package (empty init). |
| `backend/agent/rag_graph.py` | Full LangGraph state machine with `AgentState`, four node functions, conditional routing, and `run_agentic_rag` public API. |
| `backend/services/generation/generator.py` | `_process_single_query` delegates to `run_agentic_rag`; `handle_queries` returns `(answers, needs_review_flags)`. |
| `backend/api/v1/routes_query.py` | Unpacks `(answers, needs_human_review)` tuple from `handle_queries` and includes it in `HackathonResponse`. |
| `backend/models/schemas.py` | Added `needs_human_review: Optional[List[bool]] = None` to `HackathonResponse`. |
| `backend/tests/test_smoke.py` | Updated mock of `handle_queries` to return the new `(answers, flags)` tuple. |
| `backend/tests/unit/test_agent.py` | Three unit tests covering the direct path, one rewrite cycle, and exhausted-retry human review fallback. |
| `backend/requirements.txt` | Added `langgraph==1.2.7`. |
| `docs/structure.md` | Updated to V9 with `agent/` package and `test_agent.py` inventoried. |

---

## Testing

- 17 tests collected and all passed.
- Verified using `pytest backend/tests -v`.
- Result: `17 passed in 14.11s`.
