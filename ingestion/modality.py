"""Utilities for detecting the modality of raw assets before ingestion."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from multimodal.base import Modality

# Common extensions mapped to their modality. The lists can be extended as needed.
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".pdf",
    ".doc",
    ".docx",
    ".pptx",
    ".ppt",
    ".rtf",
    ".html",
    ".htm",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx", ".parquet"}

EXTENSION_MODALITY_MAP = {
    **{ext: Modality.TEXT for ext in TEXT_EXTENSIONS},
    **{ext: Modality.IMAGE for ext in IMAGE_EXTENSIONS},
    **{ext: Modality.AUDIO for ext in AUDIO_EXTENSIONS},
    **{ext: Modality.VIDEO for ext in VIDEO_EXTENSIONS},
    **{ext: Modality.TABULAR for ext in TABULAR_EXTENSIONS},
}


def detect_modality(path: str | Path) -> Modality:
    """
    Infer the modality of the file located at *path*.

    Detection prioritizes explicit extension mappings and falls back to mimetype
    guesses. Defaults to Modality.TEXT so legacy ingestion behavior remains
    compatible for unknown formats.
    """

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in EXTENSION_MODALITY_MAP:
        return EXTENSION_MODALITY_MAP[suffix]

    mime, _ = mimetypes.guess_type(p.name)
    if mime:
        if mime.startswith("text/"):
            return Modality.TEXT
        if mime.startswith("image/"):
            return Modality.IMAGE
        if mime.startswith("audio/"):
            return Modality.AUDIO
        if mime.startswith("video/"):
            return Modality.VIDEO

    return Modality.TEXT


__all__ = [
    "detect_modality",
    "Modality",
    "TEXT_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "TABULAR_EXTENSIONS",
]
