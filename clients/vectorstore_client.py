"""Direct Milvus client for vector store operations."""

from __future__ import annotations

from typing import Iterable, List, Optional

from core.config import get_settings
from core.models import SearchQuery, SearchResponse, VectorRecord

class VectorStoreClient:
    """Vector store client backed by Milvus Python SDK."""

    def __init__(self, *, connect: bool = True):
        settings = get_settings()
        self._default_namespace = settings.vectorstore_default_namespace
        self.store = None
        if connect:
            self._get_store()

    def _get_store(self):
        if self.store is None:
            from core.vectorstore.backends.milvus import MilvusStore

            self.store = MilvusStore()
        return self.store

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ensure_namespace(self, namespace: Optional[str], dim: int) -> None:
        namespace = namespace or self._default_namespace
        try:
            self.store.create_namespace(namespace, dim, overwrite=False)
        except FileExistsError:
            # already exists, so nothing to do
            pass

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        store = self._get_store()
        recs = list(records)
        if not recs:
            return
        store.upsert(recs)

    def delete(self, namespace: str, ids: List[str]) -> None:
        store = self._get_store()
        if not ids:
            return
        store.delete(namespace, ids)

    def search(self, query: SearchQuery) -> SearchResponse:
        store = self._get_store()
        return store.search(query)
