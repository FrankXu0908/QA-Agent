"""Vision embedder powered by SigLIP models."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Sequence

import numpy as np

try:  # pragma: no cover - optional dependency guard
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

from ingestion.embedders.base import BaseEmbedder
from ingestion.embedders.siglip import DEFAULT_SIGLIP_MODEL, encode_images
from ingestion.chunk_types import RawChunk
from multimodal.base import Modality
DEFAULT_VISION_MODEL = DEFAULT_SIGLIP_MODEL


class VisionEmbedder(BaseEmbedder):
    """Embeds images using CLIP models from sentence-transformers."""

    modality = Modality.IMAGE

    def __init__(self, model_name: str = DEFAULT_VISION_MODEL):
        if Image is None:
            raise RuntimeError("Pillow is required for VisionEmbedder.")
        self.model_name = model_name

    def encode(self, chunks: Sequence[RawChunk], payloads: Sequence[bytes | str] | None = None) -> np.ndarray:
        images = []
        data_iter = payloads if payloads is not None else [chunk.content for chunk in chunks]
        for chunk, data in zip(chunks, data_iter):
            if isinstance(data, str):
                # Assume base64-encoded payload
                image_bytes = base64.b64decode(data)
            else:
                image_bytes = data
            buffer = BytesIO(image_bytes)
            img = Image.open(buffer).convert("RGB")
            images.append(img)
        embeddings = encode_images(images)
        return np.asarray(embeddings)


__all__ = ["VisionEmbedder", "DEFAULT_VISION_MODEL"]
