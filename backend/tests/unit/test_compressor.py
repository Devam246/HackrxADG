import pytest
from unittest.mock import MagicMock
from models.domain import Chunk
from services.retrieval.compressor import compress_chunks

def make_mock_choice(text_content: str):
    mock_choice = MagicMock()
    mock_choice.message.content = text_content
    return mock_choice

def make_mock_response(text_content: str):
    mock_response = MagicMock()
    mock_response.choices = [make_mock_choice(text_content)]
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
    mock_client = mocker.patch("groq.Groq")
    mock_instance = mocker.MagicMock()
    
    response_json = """[
        {"index": 0, "compressed_text": "Premium is due monthly."},
        {"index": 1, "compressed_text": "Grace period is 30 days."},
        {"index": 2, "compressed_text": "Exclusions are sports and self-injury."}
    ]"""
    
    mock_instance.chat.completions.create.return_value = make_mock_response(response_json)
    mock_client.return_value = mock_instance

    mocker.patch("services.retrieval.compressor.settings.groq_api_key", "test_key")

    result = compress_chunks("What is premium and grace period?", sample_chunks)

    assert len(result) == 3
    assert result[0].chunk_id == "c1"
    assert result[0].text == "Premium is due monthly."
    assert result[1].chunk_id == "c2"
    assert result[1].text == "Grace period is 30 days."
    assert result[2].chunk_id == "c3"
    assert result[2].text == "Exclusions are sports and self-injury."


def test_compress_chunks_filtering_no_relevant(mocker, sample_chunks):
    mock_client = mocker.patch("groq.Groq")
    mock_instance = mocker.MagicMock()
    
    response_json = """[
        {"index": 0, "compressed_text": "Premium is due monthly."},
        {"index": 1, "compressed_text": "Grace period is 30 days."},
        {"index": 2, "compressed_text": "NO_RELEVANT_CONTENT"}
    ]"""
    
    mock_instance.chat.completions.create.return_value = make_mock_response(response_json)
    mock_client.return_value = mock_instance

    mocker.patch("services.retrieval.compressor.settings.groq_api_key", "test_key")

    result = compress_chunks("What is premium and grace period?", sample_chunks)

    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].text == "Premium is due monthly."
    assert result[1].chunk_id == "c2"
    assert result[1].text == "Grace period is 30 days."


def test_compress_chunks_fallback_fewer_than_two(mocker, sample_chunks):
    mock_client = mocker.patch("groq.Groq")
    mock_instance = mocker.MagicMock()
    
    response_json = """[
        {"index": 0, "compressed_text": "Premium is due monthly."},
        {"index": 1, "compressed_text": "NO_RELEVANT_CONTENT"},
        {"index": 2, "compressed_text": "NO_RELEVANT_CONTENT"}
    ]"""
    
    mock_instance.chat.completions.create.return_value = make_mock_response(response_json)
    mock_client.return_value = mock_instance

    mocker.patch("services.retrieval.compressor.settings.groq_api_key", "test_key")

    result = compress_chunks("What is premium?", sample_chunks)

    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].text == sample_chunks[0].text
    assert result[1].chunk_id == "c2"
    assert result[1].text == sample_chunks[1].text


def test_compress_chunks_api_failure_fallback(mocker, sample_chunks):
    mock_client = mocker.patch("groq.Groq")
    mock_instance = mocker.MagicMock()
    mock_instance.chat.completions.create.side_effect = Exception("Groq failure")
    mock_client.return_value = mock_instance

    mocker.patch("services.retrieval.compressor.settings.groq_api_key", "test_key")

    result = compress_chunks("What is premium?", sample_chunks)

    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].text == sample_chunks[0].text
    assert result[1].chunk_id == "c2"
    assert result[1].text == sample_chunks[1].text
