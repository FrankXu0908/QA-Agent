"""PDF-specific multimodal extraction."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import List

from ingestion.chunker import split_by_size
from ingestion.chunk_types import RawChunk
from multimodal.base import Modality


def _load_pymupdf():
    try:
        return importlib.import_module("pymupdf")
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pymupdf is required for PDF multimodal extraction") from exc


def _image_to_bytes(pymupdf_module, page, xref: int) -> bytes:
    pix = pymupdf_module.Pixmap(page.parent, xref)
    try:
        if pix.n > 4:  # convert CMYK/etc. to RGB
            pix = pymupdf_module.Pixmap(pymupdf_module.csRGB, pix)
        data = pix.tobytes("png")
    finally:
        pix = None
    return data


def extract_pdf_chunks(path: Path, *, chunk_size: int = 800) -> List[RawChunk]:
    """Return text and image chunks extracted from the PDF."""
    pymupdf_module = _load_pymupdf()
    doc = pymupdf_module.open(path)
    chunks: List[RawChunk] = []
    try:
        for page_index, page in enumerate(doc):
            try:
                text = page.get_text("text")
            except Exception:
                text = ""
            if text:
                for idx, chunk_text in enumerate(split_by_size(text, max_chars=chunk_size)):
                    chunk_id = f"{path.stem}-p{page_index + 1}-t{idx}"
                    chunks.append(
                        RawChunk(
                            chunk_id=chunk_id,
                            modality=Modality.TEXT,
                            content=chunk_text,
                            metadata={
                                "page": page_index + 1,
                                "modality": Modality.TEXT.value,
                                "mime": "text/plain",
                            },
                        )
                    )
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                chunk_id = f"{path.stem}-p{page_index + 1}-img{img_idx}"
                data_bytes = _image_to_bytes(pymupdf_module, page, xref)
                chunks.append(
                    RawChunk(
                        chunk_id=chunk_id,
                        modality=Modality.IMAGE,
                        content=data_bytes,
                        metadata={
                            "page": page_index + 1,
                            "xref": xref,
                            "width": img[2],
                            "height": img[3],
                            "cs": img[4],
                            "bpc": img[5],
                            "modality": Modality.IMAGE.value,
                            "mime": "image/png",
                        },
                    )
                )
    finally:
        doc.close()
    return chunks


__all__ = ["extract_pdf_chunks"]
