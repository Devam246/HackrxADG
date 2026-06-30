import pytest
from unittest.mock import MagicMock
from models.domain import Chunk
from services.retrieval.compressor import compress_chunks


def make_mock_response(text_content: str):
    mock_part = MagicMock()
    mock_part.text = text_content
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    return mock_response


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            doc_id="doc1",
            text="This is a long sentence about policy premium payments due on the first day of every single month without fail.",
            section_id="1.1",
            section_title="Premium Details",
            chunk_index=0,
            keywords=["premium", "payment"],
            raw_text="This is a long sentence about policy premium payments due on the first day of every single month without fail.",
            section="1.1",
            page=1,
            parent_id="p1",
            chunk_id="c1",
            is_parent=False,
            token_count=20,
        ),
        Chunk(
            doc_id="doc1",
            text="This is another very long paragraph detailing grace period rules. If you do not pay premium, you get 30 days of grace period before termination.",
            section_id="1.2",
            section_title="Grace Period",
            chunk_index=1,
            keywords=["grace", "period"],
            raw_text="This is another very long paragraph detailing grace period rules. If you do not pay premium, you get 30 days of grace period before termination.",
            section="1.2",
            page=1,
            parent_id="p2",
            chunk_id="c2",
            is_parent=False,
            token_count=25,
        ),
        Chunk(
            doc_id="doc1",
            text="Exclusions include self-inflicted injuries, adventure sports accidents, and cosmetic surgeries.",
            section_id="1.3",
            section_title="Exclusions",
            chunk_index=2,
            keywords=["exclusions"],
            raw_text="Exclusions include self-inflicted injuries, adventure sports accidents, and cosmetic surgeries.",
            section="1.3",
            page=1,
            parent_id="p3",
            chunk_id="c3",
            is_parent=False,
            token_count=15,
        ),
    ]


def test_compress_chunks_success(mocker, sample_chunks):
    # Mock Google GenAI model initialization and generate_content
    mock_model = mocker.patch("google.generativeai.GenerativeModel")
    
    # Setup model to return success for all chunks
    mock_model.return_value.generate_content.side_effect = [
        make_mock_response("Premium is due monthly."),
        make_mock_response("Grace period is 30 days."),
        make_mock_response("Exclusions are sports and self-injury."),
    ]

    result = compress_chunks("What is premium and grace period?", sample_chunks)

    # 3 chunks compressed successfully, len >= 2 so no fallback
    assert len(result) == 3
    assert result[0].chunk_id == "c1"
    assert result[0].text == "Premium is due monthly."
    assert result[1].chunk_id == "c2"
    assert result[1].text == "Grace period is 30 days."
    assert result[2].chunk_id == "c3"
    assert result[2].text == "Exclusions are sports and self-injury."


def test_compress_chunks_filtering_no_relevant(mocker, sample_chunks):
    mock_model = mocker.patch("google.generativeai.GenerativeModel")
    
    # First two chunks return valid text, third returns NO_RELEVANT_CONTENT
    mock_model.return_value.generate_content.side_effect = [
        make_mock_response("Premium is due monthly."),
        make_mock_response("Grace period is 30 days."),
        make_mock_response("NO_RELEVANT_CONTENT"),
    ]

    result = compress_chunks("What is premium and grace period?", sample_chunks)

    # c3 is filtered out. Remaining are c1 and c2. Since len([c1, c2]) == 2 (>= 2), we keep them and don't fallback.
    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].text == "Premium is due monthly."
    assert result[1].chunk_id == "c2"
    assert result[1].text == "Grace period is 30 days."


def test_compress_chunks_fallback_fewer_than_two(mocker, sample_chunks):
    mock_model = mocker.patch("google.generativeai.GenerativeModel")
    
    # Only one chunk survives, other two return NO_RELEVANT_CONTENT
    mock_model.return_value.generate_content.side_effect = [
        make_mock_response("Premium is due monthly."),
        make_mock_response("NO_RELEVANT_CONTENT"),
        make_mock_response("NO_RELEVANT_CONTENT"),
    ]

    result = compress_chunks("What is premium?", sample_chunks)

    # Since only 1 survived (c1), which is < 2, we fallback to the first 2 original chunks (uncompressed)
    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].text == sample_chunks[0].text  # uncompressed original text
    assert result[1].chunk_id == "c2"
    assert result[1].text == sample_chunks[1].text  # uncompressed original text


def test_compress_chunks_api_failure_fallback(mocker, sample_chunks):
    mock_model = mocker.patch("google.generativeai.GenerativeModel")
    
    # Raise exception for all calls
    mock_model.return_value.generate_content.side_effect = Exception("Gemini API Overloaded")

    result = compress_chunks("What is premium?", sample_chunks)

    # Since all failed, 0 survived, which is < 2. Falls back to first 2 original chunks.
    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].text == sample_chunks[0].text
    assert result[1].chunk_id == "c2"
    assert result[1].text == sample_chunks[1].text
