"""Ingestion helper to persist embeddings to the vector store service."""

from __future__ import annotations

from typing import Iterable, List

from clients.vectorstore_client import VectorStoreClient
from core.config import get_settings
from core.models import VectorMetadata, VectorRecord

settings = get_settings()
vector_client = VectorStoreClient()


def build_record(
    record_id: str,
    namespace: str,
    embedding: List[float],
    metadata: dict,
) -> VectorRecord:
    """Construct a VectorRecord from raw ingestion outputs."""
    meta = VectorMetadata(**metadata)
    return VectorRecord(id=record_id, namespace=namespace, embedding=embedding, metadata=meta)


def write_upserts(records: Iterable[VectorRecord]) -> None:
    """Send upsert request to the vector store."""
    vector_client.upsert(records)


def write_deletes(namespace: str, ids: List[str]) -> None:
    """Delete records from the vector store."""
    vector_client.delete(namespace, ids)
