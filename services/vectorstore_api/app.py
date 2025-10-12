"""FastAPI application exposing vector store operations."""

from __future__ import annotations

from typing import Iterable, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.config import get_settings
from core.models import SearchQuery, SearchResponse, VectorRecord
from core.vectorstore.backends import MilvusStore

settings = get_settings()
store = MilvusStore()

app = FastAPI(title="QA-Agent Vector Store", version="1.0.0")


class NamespaceCreateBody(BaseModel):
    namespace: str = Field(default_factory=lambda: settings.vectorstore_default_namespace)
    dim: int = Field(..., gt=0)
    overwrite: bool = False


class UpsertBody(BaseModel):
    records: List[VectorRecord]


class DeleteBody(BaseModel):
    namespace: str
    ids: List[str]


def _iter_namespaces(records: Iterable[VectorRecord]) -> List[str]:
    namespaces = {r.namespace for r in records}
    return list(namespaces)


@app.post("/namespaces", status_code=201)
def create_namespace(body: NamespaceCreateBody):
    try:
        store.create_namespace(body.namespace, body.dim, overwrite=body.overwrite)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"namespace": body.namespace, "dim": body.dim}


@app.post("/records:upsert")
def upsert_records(body: UpsertBody):
    if not body.records:
        return {"updated_namespaces": []}

    namespaces = _iter_namespaces(body.records)
    if len(namespaces) != 1:
        raise HTTPException(status_code=400, detail="All records must use a single namespace.")

    store.upsert(body.records)
    return {"updated_namespaces": namespaces}


@app.post("/records:delete")
def delete_records(body: DeleteBody):
    store.delete(body.namespace, body.ids)
    return {"namespace": body.namespace, "deleted": len(body.ids)}


@app.post("/search", response_model=SearchResponse)
def search(body: SearchQuery):
    return store.search(body)
