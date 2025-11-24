"""Common interfaces for modality-specific embedders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from multimodal.base import Modality
from ingestion.chunk_types import RawChunk


class BaseEmbedder(ABC):
    """Shared contract for all embedder implementations."""

    modality: Modality
    model_name: str

    @abstractmethod
    def encode(self, chunks: Sequence[RawChunk], payloads: Sequence[Any] | None = None) -> np.ndarray:
        """Return embeddings for the provided chunks."""


__all__ = ["BaseEmbedder"]
