"""Embedders for different modalities."""

from .base import BaseEmbedder
from .registry import EmbedderRegistry, get_default_registry
from .text import TextEmbedder
from .vision import VisionEmbedder
from .tabular import TabularEmbedder

__all__ = [
    "BaseEmbedder",
    "EmbedderRegistry",
    "get_default_registry",
    "TextEmbedder",
    "VisionEmbedder",
    "TabularEmbedder",
]
