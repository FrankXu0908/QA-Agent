"""HTTP client for the vector store service."""

from __future__ import annotations

from typing import Iterable, List, Optional

import requests
from requests import HTTPError, Response

from core.config import get_settings
from core.models import SearchQuery, SearchResponse, VectorRecord


class VectorStoreClient:
    """Simple REST client targeting the vector store API."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        settings = get_settings()
        self.base_url = base_url or settings.vectorstore_api_url
        self.timeout = timeout
        self._session = requests.Session()
        self._default_namespace = settings.vectorstore_default_namespace

    # ------------------------------------------------------------------ #
    # REST helpers
    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, *, json_body: Optional[dict] = None) -> Response:
        url = f"{self.base_url.rstrip('/')}{path}"
        resp = self._session.request(method, url, json=json_body, timeout=self.timeout)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ensure_namespace(self, namespace: Optional[str], dim: int) -> None:
        namespace = namespace or self._default_namespace
        try:
            self._request(
                "POST",
                "/namespaces",
                json_body={"namespace": namespace, "dim": dim},
            )
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 409:
                return
            raise

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        records = [r.model_dump(mode="json") for r in records]
        if not records:
            return
        self._request("POST", "/records:upsert", json_body={"records": records})

    def delete(self, namespace: str, ids: List[str]) -> None:
        if not ids:
            return
        self._request(
            "POST",
            "/records:delete",
            json_body={"namespace": namespace, "ids": ids},
        )

    def search(self, query: SearchQuery) -> SearchResponse:
        resp = self._request("POST", "/search", json_body=query.model_dump(mode="json"))
        data = resp.json()
        return SearchResponse(**data)
