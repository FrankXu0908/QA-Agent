"""Smoke test simulating a QA call with mocked dependencies."""

from unittest.mock import MagicMock

import pytest

try:
    from clients.vectorstore_client import VectorStoreClient
    from core.models import SearchHit, SearchResponse, VectorMetadata
    from services.rag_api.app import QARequest, qa
except Exception as exc:  # pragma: no cover - allow optional deps
    pytest.skip(f"Skipping smoke tests because dependencies are missing: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def patch_clients(monkeypatch):
    # Mock vector store search
    mock_client = MagicMock(spec=VectorStoreClient)
    hit = SearchHit(
        id="chunk-1",
        score=0.95,
        metadata=VectorMetadata(doc_id="doc-1", source="test", text="测试内容"),
    )
    mock_client.search.return_value = SearchResponse(hits=[hit])
    monkeypatch.setattr("services.rag_api.app.vector_client", mock_client)

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [[0.1, 0.2, 0.3]]
    monkeypatch.setattr("services.rag_api.app.embedder", mock_embedder)

    # Mock requests.post for LLM
    mock_post = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"text": "示例回答"}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp
    monkeypatch.setattr("services.rag_api.app.requests.post", mock_post)

    yield mock_client, mock_embedder, mock_post


def test_qa_smoke(patch_clients):
    req = QARequest(question="测试问题", topk=1)
    resp = qa(req)

    assert resp.answer == "示例回答"
    assert resp.sources

if __name__ == "__main__":
    pytest.main([__file__])
