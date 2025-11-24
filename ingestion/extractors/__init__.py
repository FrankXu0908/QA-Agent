"""Extractor helpers for different multimodal sources."""

from __future__ import annotations

from pathlib import Path
from typing import List, Protocol

from ingestion.chunk_types import RawChunk


class ChunkExtractor(Protocol):
    """Protocol for extractor callables."""

    def __call__(self, path: Path, *, chunk_size: int) -> List[RawChunk]:
        ...


from .pdf import extract_pdf_chunks  # noqa: E402
from .docx import extract_docx_chunks  # noqa: E402
from .pptx import extract_pptx_chunks  # noqa: E402
from .tabular import extract_tabular_chunks  # noqa: E402

__all__ = [
    "ChunkExtractor",
    "extract_pdf_chunks",
    "extract_docx_chunks",
    "extract_pptx_chunks",
    "extract_tabular_chunks",
]
