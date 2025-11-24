"""Shared SigLIP model loader utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import torch
    from transformers import AutoModel, AutoProcessor
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoProcessor = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = ROOT / "models"
# Prefer locally downloaded model if present; otherwise falls back to HF hub id.
LOCAL_SIGLIP_DIR = MODEL_CACHE / "siglip-base-patch16-384"
DEFAULT_SIGLIP_MODEL = str(LOCAL_SIGLIP_DIR) if LOCAL_SIGLIP_DIR.exists() else "google/siglip-base-patch16-384"


def _device() -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is required for SigLIP embeddings.")
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_siglip_components(model_name: str = DEFAULT_SIGLIP_MODEL):
    if AutoModel is None or AutoProcessor is None:
        raise RuntimeError("transformers must be installed for SigLIP embeddings.")
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=str(MODEL_CACHE),
        trust_remote_code=True,
        local_files_only=Path(model_name).exists(),
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=str(MODEL_CACHE),
        trust_remote_code=True,
        local_files_only=Path(model_name).exists(),
    )
    model.to(_device())
    model.eval()
    return model, processor


def encode_text(texts: Iterable[str]) -> np.ndarray:
    model, processor = get_siglip_components()
    inputs = processor(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(_device())
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.cpu().numpy()


def encode_images(images: List["Image.Image"]) -> np.ndarray:
    model, processor = get_siglip_components()
    inputs = processor(images=images, return_tensors="pt").to(_device())
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.cpu().numpy()


__all__ = [
    "DEFAULT_SIGLIP_MODEL",
    "get_siglip_components",
    "encode_text",
    "encode_images",
]
