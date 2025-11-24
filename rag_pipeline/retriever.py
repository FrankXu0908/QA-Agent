# rag_pipeline/retriever.py
"""Convenience helper for running ad-hoc searches against the vector store service."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency for image queries
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

from clients.vectorstore_client import VectorStoreClient
from core.config import get_settings
from core.models import SearchHit, SearchQuery
from ingestion.embedders.siglip import encode_images, encode_text

vector_client = VectorStoreClient()
settings = get_settings()


def _load_image(data: str) -> Image.Image:
    if Image is None:  # pragma: no cover - defensive
        raise RuntimeError("Pillow must be installed for image queries.")
    _, _, b64_data = data.partition(",")
    payload = b64_data or data
    image_bytes = base64.b64decode(payload)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return img


def _deduplicate_hits(hits: Iterable[SearchHit]) -> List[SearchHit]:
    best: Dict[str, SearchHit] = {}
    for hit in hits:
        current = best.get(hit.id)
        if current is None or hit.score > current.score:
            best[hit.id] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def retrieve(
    text: Optional[str],
    *,
    images: Optional[List[str]] = None,
    topk: int = 5,
    namespace: Optional[str] = None,
) -> List[dict]:
    """Return merged results for the supplied text and/or image inputs."""
    queries: List[Tuple[List[float], Optional[dict]]] = []
    if text and text.strip():
        text_vec = encode_text([text])[0].tolist()
        queries.append((text_vec, None))
    if images:
        loaded_images = [_load_image(raw) for raw in images if raw]
        if loaded_images:
            for vec in encode_images(loaded_images):
                queries.append((vec.tolist(), None))

    if not queries:
        raise ValueError("At least one text question or image must be provided.")

    namespace = namespace or settings.vectorstore_default_namespace
    collected: List[SearchHit] = []
    for embedding, filters in queries:
        query = SearchQuery(
            namespace=namespace,
            query_embedding=embedding,
            top_k=topk,
            filters=filters or {},
        )
        resp = vector_client.search(query)
        collected.extend(resp.hits)

    merged = _deduplicate_hits(collected)[:topk]
    return [hit.model_dump() for hit in merged]


def test():
    query = "项目领料流程是怎样的？"
    results = retrieve(query, topk=3)
    for r in results:
        metadata = r.get("metadata", {})
        print(
            f"Score: {r.get('score', 0):.4f}, Source: {metadata.get('source')}, "
            f"Modality: {metadata.get('extra', {}).get('modality')}, "
            f"Snippet: {metadata.get('text', '')[:50]}..."
        )


if __name__ == "__main__":
    test()
