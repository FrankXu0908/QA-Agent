"""Abstract vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from core.models import SearchQuery, SearchResponse, VectorRecord


class VectorStore(ABC):
    """Interface that concrete vector store backends must implement."""

    @abstractmethod
    def create_namespace(self, namespace: str, dim: int, *, overwrite: bool = False) -> None:
        """Create a namespace (collection) with the provided embedding dimension."""

    @abstractmethod
    def upsert(self, records: Iterable[VectorRecord]) -> None:
        """Insert or update records."""

    @abstractmethod
    def delete(self, namespace: str, ids: List[str]) -> None:
        """Delete records for the namespace."""

    @abstractmethod
    def search(self, query: SearchQuery) -> SearchResponse:
        """Execute a similarity search."""
