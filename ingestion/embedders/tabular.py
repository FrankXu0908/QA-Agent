"""Tabular embedder that flattens tables and reuses the text encoder."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ingestion.embedders.base import BaseEmbedder
from ingestion.embedders.siglip import encode_text, DEFAULT_SIGLIP_MODEL
from ingestion.chunk_types import RawChunk
from multimodal.base import Modality


class TabularEmbedder(BaseEmbedder):
    """Embed tabular chunks by delegating to the text embedder."""

    modality = Modality.TABULAR

    def __init__(self, text_embedder: BaseEmbedder | None = None, text_model_name: str | None = None):
        self.model_name = f"{text_model_name or DEFAULT_SIGLIP_MODEL}-tabular"

    def encode(self, chunks: Sequence[RawChunk], payloads: Sequence[str] | None = None) -> np.ndarray:  # type: ignore[override]
        texts = payloads or [chunk.as_text() for chunk in chunks]
        embeddings = encode_text(list(texts))
        return np.asarray(embeddings)


__all__ = ["TabularEmbedder"]
