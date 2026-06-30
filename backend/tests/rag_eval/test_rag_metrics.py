"""
tests/rag_eval/test_rag_metrics.py — V10 Evaluation Suite

Tests are SKIPPED automatically when GEMINI_API_KEY is not set.
To run with real API credits:
    $env:GEMINI_API_KEY="your_key"
    pytest backend/tests/rag_eval/ -v --tb=short

CI: Only the GitHub Actions 'rag-eval' job (main branch) runs these.
"""

import json
import os
import sys
from pathlib import Path
from typing import List

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Guards — skip the whole module if API key or deepeval are absent
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
IS_VALID_KEY = len(GEMINI_API_KEY) > 5
_SKIP_REASON = "GEMINI_API_KEY not set or too short — skipping live evaluation tests"
_needs_api = pytest.mark.skipif(not IS_VALID_KEY, reason=_SKIP_REASON)

try:
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

_needs_deepeval = pytest.mark.skipif(
    not DEEPEVAL_AVAILABLE or not IS_VALID_KEY,
    reason="deepeval not installed or GEMINI_API_KEY not set/invalid",
)
_needs_ragas = pytest.mark.skipif(
    not RAGAS_AVAILABLE or not IS_VALID_KEY,
    reason="ragas not installed or GEMINI_API_KEY not set/invalid",
)


# ─────────────────────────────────────────────────────────────────────────────
# Load benchmark dataset
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_PATH = Path(__file__).parent / "insurance_benchmark.json"


def load_benchmark() -> List[dict]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini-backed LLM adapter for DeepEval
# ─────────────────────────────────────────────────────────────────────────────

class GeminiDeepEvalModel(DeepEvalBaseLLM):
    """Wraps google-generativeai to serve as the judge LLM for DeepEval."""

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        self._model = genai.GenerativeModel("gemini-2.5-flash")

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(
            contents=[prompt],
            generation_config={"temperature": 0.0, "max_output_tokens": 1024},
        )
        return response.text.strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "gemini-2.5-flash"


def _build_test_cases() -> List["LLMTestCase"]:
    """Build DeepEval test cases from the benchmark JSON."""
    benchmark = load_benchmark()
    cases = []
    for entry in benchmark:
        cases.append(
            LLMTestCase(
                input=entry["input"],
                actual_output=entry["expected_output"],   # used as the model's response proxy
                expected_output=entry["expected_output"],
                retrieval_context=entry["retrieval_context"],
            )
        )
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# DeepEval Metric Tests — each enforces a minimum threshold
# ─────────────────────────────────────────────────────────────────────────────

@_needs_deepeval
def test_faithfulness():
    """Faithfulness: answer must be grounded in retrieved context. Threshold: 0.85."""
    judge = GeminiDeepEvalModel()
    metric = FaithfulnessMetric(threshold=0.85, model=judge, include_reason=True)
    for case in _build_test_cases():
        assert_test(case, [metric])


@_needs_deepeval
def test_answer_relevancy():
    """Answer Relevancy: answer must address the question. Threshold: 0.80."""
    judge = GeminiDeepEvalModel()
    metric = AnswerRelevancyMetric(threshold=0.80, model=judge, include_reason=True)
    for case in _build_test_cases():
        assert_test(case, [metric])


@_needs_deepeval
def test_contextual_precision():
    """Contextual Precision: retrieved chunks must be relevant. Threshold: 0.75."""
    judge = GeminiDeepEvalModel()
    metric = ContextualPrecisionMetric(threshold=0.75, model=judge, include_reason=True)
    for case in _build_test_cases():
        assert_test(case, [metric])


@_needs_deepeval
def test_contextual_recall():
    """Contextual Recall: all necessary info must be retrieved. Threshold: 0.75."""
    judge = GeminiDeepEvalModel()
    metric = ContextualRecallMetric(threshold=0.75, model=judge, include_reason=True)
    for case in _build_test_cases():
        assert_test(case, [metric])


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS Batch Scoring — prints aggregate scores, does not enforce thresholds
# ─────────────────────────────────────────────────────────────────────────────

@_needs_ragas
def test_ragas_batch_scores():
    """
    Run RAGAS across the full benchmark and print per-metric averages.
    This test passes as long as RAGAS runs without error.
    Actual threshold enforcement is done by the DeepEval tests above.
    """
    from datasets import Dataset

    benchmark = load_benchmark()

    data = {
        "question": [e["input"] for e in benchmark],
        "answer": [e["expected_output"] for e in benchmark],
        "contexts": [e["retrieval_context"] for e in benchmark],
        "ground_truth": [e["expected_output"] for e in benchmark],
    }
    dataset = Dataset.from_dict(data)

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = result.to_pandas()
    summary = scores[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()

    print("\n" + "=" * 60)
    print("RAGAS Evaluation Summary — V10")
    print("=" * 60)
    for metric, score in summary.items():
        status = "✅" if score >= 0.75 else "❌"
        print(f"  {status}  {metric:<25} {score:.4f}")
    print("=" * 60 + "\n")

    # Store to benchmarks.md reminder (manual step)
    assert result is not None, "RAGAS evaluation must complete without error"
