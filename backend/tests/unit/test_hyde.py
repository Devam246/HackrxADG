import pytest
from services.retrieval.hyde import generate_hypothetical_excerpt

def test_generate_hypothetical_excerpt_success(mocker):
    # Mock Groq client initialization and chat completions create
    mock_client = mocker.patch("groq.Groq")
    mock_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = "This is a hypothetical policy excerpt describing orthopaedic surgery coverage."
    mock_response.choices = [mock_choice]
    mock_instance.chat.completions.create.return_value = mock_response
    mock_client.return_value = mock_instance

    mocker.patch("services.retrieval.hyde.settings.groq_api_key", "test_key")

    result = generate_hypothetical_excerpt("Will my knee replacement be covered?")
    assert result == "This is a hypothetical policy excerpt describing orthopaedic surgery coverage."
    mock_instance.chat.completions.create.assert_called_once()


def test_generate_hypothetical_excerpt_failure_fallback(mocker):
    # Mock Groq to raise an exception on generation
    mock_client = mocker.patch("groq.Groq")
    mock_instance = mocker.MagicMock()
    mock_instance.chat.completions.create.side_effect = Exception("Groq API Error")
    mock_client.return_value = mock_instance

    mocker.patch("services.retrieval.hyde.settings.groq_api_key", "test_key")

    result = generate_hypothetical_excerpt("Will my knee replacement be covered?")
    assert result == "Will my knee replacement be covered?"
