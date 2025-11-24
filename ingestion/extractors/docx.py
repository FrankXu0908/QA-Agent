"""DOCX multimodal extraction (text + embedded images)."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ingestion.chunker import split_by_size
from ingestion.chunk_types import RawChunk
from multimodal.base import Modality


def _load_docx():
    try:
        from docx import Document  # type: ignore
        from docx.opc.constants import RELATIONSHIP_TYPE as RT  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("python-docx is required for DOCX extraction") from exc
    return Document, RT


def extract_docx_chunks(path: Path, *, chunk_size: int = 800) -> List[RawChunk]:
    Document, RT = _load_docx()
    doc = Document(path)
    chunks: List[RawChunk] = []

    # Text paragraphs
    text = "\n".join(p.text for p in doc.paragraphs if p.text)
    if text:
        for idx, chunk_text in enumerate(split_by_size(text, max_chars=chunk_size)):
            chunk_id = f"{path.stem}-p{idx}"
            chunks.append(
                RawChunk(
                    chunk_id=chunk_id,
                    modality=Modality.TEXT,
                    content=chunk_text,
                    metadata={
                        "modality": Modality.TEXT.value,
                        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    },
                )
            )

    # Embedded images
    for rel_id, rel in doc.part.related_parts.items():
        if getattr(rel, "reltype", None) != RT.IMAGE:
            continue
        blob = rel.blob
        if not blob:
            continue
        chunk_id = f"{path.stem}-img-{rel_id}"
        chunks.append(
            RawChunk(
                chunk_id=chunk_id,
                modality=Modality.IMAGE,
                content=blob,
                metadata={
                    "modality": Modality.IMAGE.value,
                    "image_rel_id": rel_id,
                    "mime": rel.content_type if hasattr(rel, "content_type") else "image/png",
                },
            )
        )

    return chunks


__all__ = ["extract_docx_chunks"]
