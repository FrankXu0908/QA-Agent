"""SentenceTransformer-based text embedder."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ingestion.embedders.base import BaseEmbedder
from ingestion.embedders.siglip import encode_text, DEFAULT_SIGLIP_MODEL
from ingestion.chunk_types import RawChunk
from multimodal.base import Modality


class TextEmbedder(BaseEmbedder):
    """Default text embedder using SigLIP."""

    modality = Modality.TEXT

    def __init__(self, model_name: str = DEFAULT_SIGLIP_MODEL):
        self.model_name = model_name

    def encode(self, chunks: Sequence[RawChunk], payloads: Sequence[str] | None = None) -> np.ndarray:
        texts: Iterable[str]
        if payloads is not None:
            texts = payloads
        else:
            texts = [chunk.as_text() for chunk in chunks]
        embeddings = encode_text(list(texts))
        return np.asarray(embeddings)


__all__ = ["TextEmbedder", "DEFAULT_SIGLIP_MODEL"]
