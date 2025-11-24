"""Abstract representations for future multimodal workflows."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"


class MultimodalRequest(BaseModel):
    """Placeholder request payload for multimodal processing."""

    question: str = Field(..., description="User question or instruction")
    modality: Modality = Field(default=Modality.TEXT)
    payload_uri: Optional[str] = Field(default=None, description="URI to external media")
    metadata: Dict[str, str] = Field(default_factory=dict)


class MultimodalResponse(BaseModel):
    """Placeholder response format for multimodal answers."""

    status: str = Field(default="not_implemented")
    message: str = Field(default="Multimodal pipeline not implemented yet.")
    data: Dict[str, str] = Field(default_factory=dict)
