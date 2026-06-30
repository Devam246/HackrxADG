from fastapi.testclient import TestClient

import main


def test_query_smoke_returns_200(monkeypatch):
    def fake_load_or_create_chunks(document_url: str):
        chunk = {
            "doc_id": "doc123",
            "text": "Section 1\nGrace period is 30 days.",
            "raw_text": "Grace period is 30 days.",
            "section_id": "1",
            "section_title": "Grace Period",
            "keywords": ["grace", "period"],
        }
        return [chunk], [[0.1, 0.2]], "doc123", "insurance"

    monkeypatch.setattr(main, "load_or_create_chunks", fake_load_or_create_chunks)
    monkeypatch.setattr(main, "build_inverted_index", lambda chunks: {"grace": {0}})
    monkeypatch.setattr(main, "handle_queries", lambda *args, **kwargs: ["The grace period is 30 days."])
    monkeypatch.setattr(main, "EXPECTED_BEARER_TOKEN", "test-token")
    main.document_cache.clear()

    client = TestClient(main.app)
    response = client.post(
        "/hackrx/run",
        headers={"Authorization": "Bearer test-token"},
        json={
            "documents": "https://example.com/policy.pdf",
            "questions": ["What is the grace period?"],
        },
    )

    assert response.status_code == 200
    assert response.json()["answers"] == ["The grace period is 30 days."]
