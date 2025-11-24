"""Convenience exports for the ingestion toolkit."""

from .pipeline import FileIngestionResult, IngestionReport, MultimodalIngestionPipeline
from .modality import detect_modality
from .embedders import (
    BaseEmbedder,
    EmbedderRegistry,
    TextEmbedder,
    VisionEmbedder,
    TabularEmbedder,
    get_default_registry,
)

__all__ = [
    "MultimodalIngestionPipeline",
    "IngestionReport",
    "FileIngestionResult",
    "detect_modality",
    "BaseEmbedder",
    "EmbedderRegistry",
    "TextEmbedder",
    "VisionEmbedder",
    "TabularEmbedder",
    "get_default_registry",
]
