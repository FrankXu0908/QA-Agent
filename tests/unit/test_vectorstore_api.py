"""Tests for the vector store FastAPI service."""

from fastapi.testclient import TestClient

import services.vectorstore_api.app as vector_app


class DummyStore:
    def __init__(self):
        self.namespaces = {}
        self.records = {}

    def create_namespace(self, namespace: str, dim: int, *, overwrite: bool = False):
        if namespace in self.namespaces and not overwrite:
            raise FileExistsError("exists")
        self.namespaces[namespace] = dim
        self.records.setdefault(namespace, {})

    def upsert(self, records):
        for record in records:
            ns = record.namespace
            self.records.setdefault(ns, {})
            self.records[ns][record.id] = record

    def delete(self, namespace: str, ids):
        for rid in ids:
            self.records.get(namespace, {}).pop(rid, None)

    def search(self, query):
        return vector_app.SearchResponse(hits=[])


def test_create_namespace(monkeypatch):
    dummy = DummyStore()
    monkeypatch.setattr(vector_app, "store", dummy)
    client = TestClient(vector_app.app)

    resp = client.post("/namespaces", json={"namespace": "kb", "dim": 384})
    assert resp.status_code == 201
    assert dummy.namespaces["kb"] == 384


def test_upsert(monkeypatch):
    dummy = DummyStore()
    monkeypatch.setattr(vector_app, "store", dummy)
    client = TestClient(vector_app.app)

    payload = {
        "records": [
            {
                "id": "chunk1",
                "namespace": "kb",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "doc_id": "doc1",
                    "schema_version": "v1",
                },
            }
        ]
    }

    resp = client.post("/records:upsert", json=payload)
    assert resp.status_code == 200
    assert dummy.records["kb"]["chunk1"].metadata.doc_id == "doc1"


def test_delete(monkeypatch):
    dummy = DummyStore()
    dummy.records["kb"] = {"chunk1": "record"}
    monkeypatch.setattr(vector_app, "store", dummy)
    client = TestClient(vector_app.app)

    resp = client.post(
        "/records:delete",
        json={"namespace": "kb", "ids": ["chunk1"]},
    )
    assert resp.status_code == 200
    assert "chunk1" not in dummy.records["kb"]


def test_search(monkeypatch):
    class SearchDummy(DummyStore):
        def search(self, query):
            metadata = vector_app.VectorMetadata(doc_id="doc", schema_version="v1")
            hit = vector_app.SearchHit(id="chunk1", score=0.9, metadata=metadata)
            return vector_app.SearchResponse(hits=[hit])

    dummy = SearchDummy()
    monkeypatch.setattr(vector_app, "store", dummy)
    client = TestClient(vector_app.app)

    resp = client.post(
        "/search",
        json={
            "namespace": "kb",
            "query_embedding": [0.1, 0.2, 0.3],
            "top_k": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["hits"][0]["id"] == "chunk1"
