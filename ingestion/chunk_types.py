"""Dataclasses representing intermediate ingestion outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from multimodal.base import Modality


@dataclass
class RawChunk:
    """Intermediate chunk prior to cleaning/embedding."""

    chunk_id: str
    modality: Modality
    content: bytes | str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        if isinstance(self.content, bytes):
            raise TypeError("Chunk content is binary; cannot represent as text.")
        return self.content

    def as_bytes(self) -> bytes:
        if isinstance(self.content, str):
            return self.content.encode("utf-8")
        return self.content


__all__ = ["RawChunk"]
