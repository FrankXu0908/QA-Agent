"""Milvus-backed vector store implementation."""

from __future__ import annotations

import json
from typing import Iterable, List
from pymilvus import (  # type: ignore[import]
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from core.config import get_settings
from core.models import SearchHit, SearchQuery, SearchResponse, VectorMetadata, VectorRecord
from core.vectorstore.interface import VectorStore


class MilvusStore(VectorStore):
    """Vector store powered by Milvus."""

    def __init__(self, alias: str = "default"):
        settings = get_settings()
        self._alias = alias
        connections.connect(
            alias=alias,
            host=settings.milvus_host,
            port=str(settings.milvus_port),
            user=settings.milvus_user or None,
            password=settings.milvus_password or None,
            secure=settings.milvus_secure,
        )
        self._metric_type = "COSINE"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collection(self, namespace: str) -> Collection:
        return Collection(namespace, using=self._alias)

    def _ensure_loaded(self, collection: Collection) -> None:
        if not collection.has_index():
            collection.create_index(
                field_name="embedding",
                index_params={"index_type": "AUTOINDEX", "metric_type": self._metric_type},
            )
        collection.load()

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------
    def create_namespace(self, namespace: str, dim: int, *, overwrite: bool = False) -> None:
        if utility.has_collection(namespace, using=self._alias):
            if not overwrite:
                raise FileExistsError(
                    f"Namespace '{namespace}' already exists. Use overwrite=True to recreate it."
                )
            utility.drop_collection(namespace, using=self._alias)

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=128,
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields, description=f"QA-Agent namespace {namespace}")
        collection = Collection(namespace, schema, using=self._alias)
        self._ensure_loaded(collection)

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        records = list(records)
        if not records:
            return
        namespace = records[0].namespace
        if any(r.namespace != namespace for r in records):
            raise ValueError("All records must belong to the same namespace.")

        collection = self._collection(namespace)
        self._ensure_loaded(collection)

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadata_json: List[str] = []
        for record in records:
            ids.append(record.id)
            embeddings.append(record.embedding)
            metadata_json.append(json.dumps(record.metadata.model_dump(mode="json", exclude_none=True), ensure_ascii=False))

        try:
            collection.upsert([ids, embeddings, metadata_json])
        except AttributeError:
            collection.insert([ids, embeddings, metadata_json])
        collection.flush()

    def delete(self, namespace: str, ids: List[str]) -> None:
        if not ids:
            return
        collection = self._collection(namespace)
        expr_ids = ",".join(f'"{_id}"' for _id in ids)
        collection.delete(expr=f"id in [{expr_ids}]")
        collection.flush()

    def search(self, query: SearchQuery) -> SearchResponse:
        collection = self._collection(query.namespace)
        self._ensure_loaded(collection)

        results = collection.search(
            data=[query.query_embedding],
            anns_field="embedding",
            param={"metric_type": self._metric_type},
            limit=query.top_k,
            output_fields=["metadata"],
        )

        hits: List[SearchHit] = []
        filter_items = query.filters.items()

        for hit in results[0]:
            metadata_raw = hit.entity.get("metadata") or "{}"
            metadata_dict = json.loads(metadata_raw)
            if filter_items and any(metadata_dict.get(k) != v for k, v in filter_items):
                continue
            metadata = VectorMetadata(**metadata_dict)
            distance = float(hit.distance)
            score = 1.0 - distance  # convert cosine distance to similarity-style score
            hits.append(
                SearchHit(
                    id=str(hit.id),
                    score=score,
                    metadata=metadata,
                )
            )

        return SearchResponse(hits=hits)
