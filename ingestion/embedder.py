"""Backwards-compatible exports for legacy imports."""

from ingestion.embedders.text import TextEmbedder, DEFAULT_SIGLIP_MODEL
from ingestion.embedders.registry import EmbedderRegistry, get_default_registry
from ingestion.embedders.vision import VisionEmbedder
from ingestion.embedders.tabular import TabularEmbedder

Embedder = TextEmbedder

__all__ = [
    "Embedder",
    "TextEmbedder",
    "VisionEmbedder",
    "TabularEmbedder",
    "EmbedderRegistry",
    "get_default_registry",
    "DEFAULT_SIGLIP_MODEL",
]
