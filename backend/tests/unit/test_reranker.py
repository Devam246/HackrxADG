import os
import sys
from unittest.mock import MagicMock, patch
import pytest
from services.retrieval.reranker import CrossEncoderReranker
from models.domain import Chunk

def test_reranker_success_cohere(mocker):
    # Mock Cohere ClientV2 and its rerank response
    mock_client = mocker.patch("cohere.ClientV2")
    mock_instance = mocker.MagicMock()
    
    mock_result_0 = mocker.MagicMock()
    mock_result_0.index = 2
    mock_result_0.relevance_score = 0.99
    
    mock_result_1 = mocker.MagicMock()
    mock_result_1.index = 0
    mock_result_1.relevance_score = 0.85
    
    mock_response = mocker.MagicMock()
    mock_response.results = [mock_result_0, mock_result_1]
    
    mock_instance.rerank.return_value = mock_response
    mock_client.return_value = mock_instance

    mocker.patch("services.retrieval.reranker.settings.cohere_api_key", "test_key")

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
    assert result[0].chunk_id == "c3"
    assert result[0].rerank_score == 0.99
    assert result[1].chunk_id == "c1"
    assert result[1].rerank_score == 0.85


def test_reranker_key_missing_fallback(mocker):
    mocker.patch("services.retrieval.reranker.settings.cohere_api_key", "")
    mocker.patch.dict("os.environ", {}, clear=True)

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

    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[1].chunk_id == "c2"
    assert result[0].rerank_score is None


def test_reranker_inference_failure_fallback(mocker):
    mock_client = mocker.patch("cohere.ClientV2")
    mock_instance = mocker.MagicMock()
    mock_instance.rerank.side_effect = Exception("API call failed")
    mock_client.return_value = mock_instance

    mocker.patch("services.retrieval.reranker.settings.cohere_api_key", "test_key")

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

    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[1].chunk_id == "c2"
    assert result[0].rerank_score is None
