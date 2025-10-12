# vectorstore/query_index.py
"""Convenience helper for running ad-hoc searches against the vector store service."""

from clients.vectorstore_client import VectorStoreClient
from core.config import get_settings
from core.models import SearchQuery
from ingestion.embedder import Embedder

embedder = Embedder()
vector_client = VectorStoreClient()
settings = get_settings()


def retrieve(text: str, topk: int = 5):
    embedding = embedder.encode([text])[0].tolist()
    query = SearchQuery(
        namespace=settings.vectorstore_default_namespace,
        query_embedding=embedding,
        top_k=topk,
    )
    resp = vector_client.search(query)
    return [hit.model_dump() for hit in resp.hits]


def test():
    query = "项目领料流程是怎样的？"
    results = retrieve(query, topk=3)
    for r in results:
        metadata = r.get("metadata", {})
        print(
            f"Score: {r.get('score', 0):.4f}, Source: {metadata.get('source')}, "
            f"Text: {metadata.get('text', '')[:50]}..."
        )


if __name__ == "__main__":
    test()
