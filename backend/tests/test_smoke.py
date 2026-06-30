from fastapi.testclient import TestClient

import main
import api.v1.routes_query
import api.v1.deps
from utils.cache import document_cache


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

    monkeypatch.setattr(api.v1.routes_query, "load_or_create_chunks", fake_load_or_create_chunks)
    monkeypatch.setattr(api.v1.routes_query, "build_inverted_index", lambda chunks: {"grace": {0}})
    monkeypatch.setattr(
        api.v1.routes_query,
        "handle_queries",
        lambda *args, **kwargs: ["The grace period is 30 days."],
    )
    monkeypatch.setattr(api.v1.deps, "EXPECTED_BEARER_TOKEN", "test-token")
    document_cache.clear()

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
