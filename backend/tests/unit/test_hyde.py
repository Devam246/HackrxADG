import pytest
from services.retrieval.hyde import generate_hypothetical_excerpt


def test_generate_hypothetical_excerpt_success(mocker):
    # Mock Google GenAI model initialization and generate_content
    mock_model = mocker.patch("google.generativeai.GenerativeModel")
    mock_response = mocker.MagicMock()
    mock_response.text = "This is a hypothetical policy excerpt describing orthopaedic surgery coverage."
    mock_model.return_value.generate_content.return_value = mock_response

    result = generate_hypothetical_excerpt("Will my knee replacement be covered?")
    assert result == "This is a hypothetical policy excerpt describing orthopaedic surgery coverage."
    mock_model.return_value.generate_content.assert_called_once()


def test_generate_hypothetical_excerpt_failure_fallback(mocker):
    # Mock Google GenAI model to raise an exception on generation
    mock_model = mocker.patch("google.generativeai.GenerativeModel")
    mock_model.return_value.generate_content.side_effect = Exception("Gemini API Error")

    result = generate_hypothetical_excerpt("Will my knee replacement be covered?")
    # Must silently fall back to the original question text
    assert result == "Will my knee replacement be covered?"
