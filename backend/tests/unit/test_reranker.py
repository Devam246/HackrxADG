import sys
from unittest.mock import MagicMock

# Ensure sentence_transformers and FlagEmbedding are mockable
if "torch" not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    sys.modules["torch"] = mock_torch
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()
if "FlagEmbedding" not in sys.modules:
    sys.modules["FlagEmbedding"] = MagicMock()

import pytest
from services.retrieval.reranker import CrossEncoderReranker, get_reranker_model
from models.domain import Chunk


def test_reranker_success_sentence_transformers(mocker):
    # Clear the lru cache to ensure new mock is registered
    get_reranker_model.cache_clear()

    # Mock sentence_transformers CrossEncoder loading
    mock_cross_encoder = mocker.patch("sentence_transformers.CrossEncoder")
    mock_model_instance = mocker.MagicMock()
    mock_model_instance.predict.return_value = [0.85, 0.12, 0.99]
    mock_cross_encoder.return_value = mock_model_instance

    reranker = CrossEncoderReranker()
    chunks = [
        Chunk(
            doc_id="d1",
            text="text1",
            section_id="1",
            section_title="T1",
            chunk_index=0,
            keywords=[],
            raw_text="text1",
            section="1",
            page=1,
            parent_id=None,
            chunk_id="c1",
            is_parent=False,
            token_count=10,
        ),
        Chunk(
            doc_id="d1",
            text="text2",
            section_id="2",
            section_title="T2",
            chunk_index=1,
            keywords=[],
            raw_text="text2",
            section="2",
            page=1,
            parent_id=None,
            chunk_id="c2",
            is_parent=False,
            token_count=10,
        ),
        Chunk(
            doc_id="d1",
            text="text3",
            section_id="3",
            section_title="T3",
            chunk_index=2,
            keywords=[],
            raw_text="text3",
            section="3",
            page=1,
            parent_id=None,
            chunk_id="c3",
            is_parent=False,
            token_count=10,
        ),
    ]

    result = reranker.rerank("query text", chunks, top_n=2)

    assert len(result) == 2
    # Sorted by score descending: c3 (0.99) -> c1 (0.85) -> c2 (0.12, cut off by top_n=2)
    assert result[0].chunk_id == "c3"
    assert result[0].rerank_score == 0.99
    assert result[1].chunk_id == "c1"
    assert result[1].rerank_score == 0.85


def test_reranker_load_failure_fallback(mocker):
    get_reranker_model.cache_clear()

    # Force import or initialization failure
    mocker.patch("sentence_transformers.CrossEncoder", side_effect=Exception("Load failed"))
    mocker.patch("FlagEmbedding.FlagReranker", side_effect=Exception("Load failed"))

    reranker = CrossEncoderReranker()
    chunks = [
        Chunk(
            doc_id="d1",
            text="text1",
            section_id="1",
            section_title="T1",
            chunk_index=0,
            keywords=[],
            raw_text="text1",
            section="1",
            page=1,
            parent_id=None,
            chunk_id="c1",
            is_parent=False,
            token_count=10,
        ),
        Chunk(
            doc_id="d1",
            text="text2",
            section_id="2",
            section_title="T2",
            chunk_index=1,
            keywords=[],
            raw_text="text2",
            section="2",
            page=1,
            parent_id=None,
            chunk_id="c2",
            is_parent=False,
            token_count=10,
        ),
    ]

    result = reranker.rerank("query text", chunks, top_n=2)

    # Verify fallback to original ordering
    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[1].chunk_id == "c2"
    assert result[0].rerank_score is None


def test_reranker_inference_failure_fallback(mocker):
    get_reranker_model.cache_clear()

    mock_cross_encoder = mocker.patch("sentence_transformers.CrossEncoder")
    mock_model_instance = mocker.MagicMock()
    mock_model_instance.predict.side_effect = Exception("Inference failed")
    mock_cross_encoder.return_value = mock_model_instance

    reranker = CrossEncoderReranker()
    chunks = [
        Chunk(
            doc_id="d1",
            text="text1",
            section_id="1",
            section_title="T1",
            chunk_index=0,
            keywords=[],
            raw_text="text1",
            section="1",
            page=1,
            parent_id=None,
            chunk_id="c1",
            is_parent=False,
            token_count=10,
        ),
        Chunk(
            doc_id="d1",
            text="text2",
            section_id="2",
            section_title="T2",
            chunk_index=1,
            keywords=[],
            raw_text="text2",
            section="2",
            page=1,
            parent_id=None,
            chunk_id="c2",
            is_parent=False,
            token_count=10,
        ),
    ]

    result = reranker.rerank("query text", chunks, top_n=2)

    # Verify fallback to original ordering on inference error
    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[1].chunk_id == "c2"
    assert result[0].rerank_score is None
