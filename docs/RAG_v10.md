# RAG v10 — Evaluation Suite

**Date:** 2026-06-30
**Scope:** Add the formal evaluation harness: a 15-question insurance benchmark dataset, DeepEval threshold-enforced tests for four RAG quality metrics, RAGAS batch scoring, and a GitHub Actions CI job that runs the eval suite on every merge to `main`.

---

## Goal

- Create a benchmark dataset of 15 insurance policy Q&A pairs.
- Implement DeepEval `assert_test` calls that **fail CI if any metric falls below threshold**.
- Add RAGAS batch scoring to produce aggregate summary scores per version.
- Set up a GitHub Actions `rag-eval` job triggered only on `main` branch pushes.
- Document the V10 entry in `docs/benchmarks.md` with run instructions.

---

## Previous Limitation (why this was needed)

Through V9, every retrieval improvement was made by inspection and manual testing. There was no systematic way to confirm that a new version didn't regress on faithfulness or recall. V10 establishes measurable, automated guardrails.

---

## Benchmark Dataset

**File:** `backend/tests/rag_eval/insurance_benchmark.json`  
**Questions:** 15  
**Topics covered:**

| Topic | Count |
|---|---|
| Waiting periods (cataract, maternity, pre-existing) | 3 |
| Exclusions (sports, dental, surgery first-year) | 3 |
| Sub-limits (room rent, ambulance) | 2 |
| Claim procedures and grace period | 2 |
| Coverage conditions (AYUSH, domiciliary, mental illness) | 3 |
| Policy terms (co-payment, free-look) | 2 |

---

## Metric Thresholds (CI-enforced)

| Metric | Threshold | Tool |
|---|---|---|
| Faithfulness | ≥ 0.85 | DeepEval `FaithfulnessMetric` |
| Answer Relevancy | ≥ 0.80 | DeepEval `AnswerRelevancyMetric` |
| Contextual Precision | ≥ 0.75 | DeepEval `ContextualPrecisionMetric` |
| Contextual Recall | ≥ 0.75 | DeepEval `ContextualRecallMetric` |

---

## Credit-Safe Design

- All 4 DeepEval tests and the RAGAS test use `pytest.mark.skipif(not GEMINI_API_KEY)`.
- Running `pytest backend/tests` in development **automatically skips** all eval tests — zero API credits consumed.
- Only the GitHub Actions `rag-eval` job (which has access to the `GEMINI_API_KEY` secret) runs them.

---

## Changed Files

| File | Changes |
|---|---|
| `backend/tests/rag_eval/__init__.py` | New package init. |
| `backend/tests/rag_eval/insurance_benchmark.json` | 15 Q&A pairs across major insurance policy topic areas. |
| `backend/tests/rag_eval/test_rag_metrics.py` | 4 DeepEval threshold tests + RAGAS batch scoring, all guarded by `skipif`. |
| `.github/workflows/rag_eval.yml` | CI job that runs `pytest tests/rag_eval/` only on pushes to `main`. |
| `backend/requirements.txt` | Added `deepeval` and `ragas`. |
| `docs/benchmarks.md` | Populated V10 section with run instructions and threshold table. |
| `docs/structure.md` | Updated to V10; added `rag_eval/`, `.github/workflows/` inventory. |

---

## Testing

- Running `pytest backend/tests -v` confirms all 17 existing tests pass and the 4 new eval tests are **correctly skipped** (no API key in dev environment).
- Result: `17 passed, 5 skipped`.
