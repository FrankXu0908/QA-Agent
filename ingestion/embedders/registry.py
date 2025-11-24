"""Registry for modality-specific embedders."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

from ingestion.embedders.base import BaseEmbedder
from ingestion.embedders.tabular import TabularEmbedder
from ingestion.embedders.text import TextEmbedder
from ingestion.embedders.vision import VisionEmbedder
from multimodal.base import Modality

logger = logging.getLogger(__name__)


class EmbedderRegistry:
    """Simple registry that maps modalities to embedder instances."""

    def __init__(self, embedders: Optional[Iterable[BaseEmbedder]] = None):
        self._registry: Dict[Modality, BaseEmbedder] = {}
        if embedders:
            for emb in embedders:
                self.register(emb)

    def register(self, embedder: BaseEmbedder) -> None:
        current = self._registry.get(embedder.modality)
        if current:
            logger.warning(
                "Replacing existing embedder for modality %s (%s -> %s)",
                embedder.modality,
                current.model_name,
                embedder.model_name,
            )
        self._registry[embedder.modality] = embedder

    def get(self, modality: Modality) -> Optional[BaseEmbedder]:
        return self._registry.get(modality)


def get_default_registry(
    *,
    text_embedder: Optional[BaseEmbedder] = None,
    vision_embedder: Optional[BaseEmbedder] = None,
    tabular_embedder: Optional[BaseEmbedder] = None,
) -> EmbedderRegistry:
    embedders = []
    if text_embedder is None:
        try:
            text_embedder = TextEmbedder()
        except Exception as exc:  # pragma: no cover - env specific
            raise RuntimeError(f"Failed to initialize text embedder: {exc}") from exc
    embedders.append(text_embedder)

    if vision_embedder is None:
        try:
            vision_embedder = VisionEmbedder()
        except Exception as exc:  # pragma: no cover
            logger.info("Vision embedder unavailable: %s", exc)
            vision_embedder = None
    if vision_embedder is not None:
        embedders.append(vision_embedder)

    if tabular_embedder is None:
        try:
            tabular_embedder = TabularEmbedder()
        except Exception as exc:  # pragma: no cover
            logger.info("Tabular embedder unavailable: %s", exc)
            tabular_embedder = None
    if tabular_embedder is not None:
        embedders.append(tabular_embedder)

    return EmbedderRegistry(embedders)


__all__ = ["EmbedderRegistry", "get_default_registry"]
