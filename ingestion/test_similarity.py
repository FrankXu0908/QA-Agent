"""Similarity regression tests for the embedder (optional)."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"Skipping similarity tests because scikit-learn is missing: {exc}", allow_module_level=True)

try:
    from ingestion import embedder as embedder_module
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"Skipping similarity tests because Embedder module is missing: {exc}", allow_module_level=True)

EMB_MODEL_PATH = Path(getattr(embedder_module, "EMB_MODEL", ""))
if not EMB_MODEL_PATH.exists():
    pytest.skip(
        f"Skipping similarity tests because embedding model '{EMB_MODEL_PATH}' is not available.",
        allow_module_level=True,
    )

try:
    from ingestion.embedder import Embedder
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"Skipping similarity tests because sentence_transformers is missing: {exc}", allow_module_level=True)


SIMILAR_PAIRS = [
    ("我喜欢吃苹果", "苹果是一种水果"),
    ("今天天气很好", "今天是晴朗的一天"),
    ("他很高", "他的身高很高"),
]

DISSIMILAR_PAIRS = [
    ("我喜欢吃苹果", "计算机编程很有趣"),
    ("今天天气很好", "数学是一门科学"),
    ("他很高", "这本书很厚"),
]


# Image-text pairs for similarity testing
IMAGE_TEXT_PAIRS = [
    ("tests/data/apple.jpg", "苹果是一种水果"),
    ("tests/data/sunny.jpg", "今天天气很好"),
]


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.mark.parametrize("text1,text2", SIMILAR_PAIRS)
def test_similar_pairs(embedder, text1, text2):
    embeddings = embedder.encode([text1, text2])
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    assert similarity > 0.5


@pytest.mark.parametrize("text1,text2", DISSIMILAR_PAIRS)
def test_dissimilar_pairs(embedder, text1, text2):
    embeddings = embedder.encode([text1, text2])
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    assert similarity < 0.5


# Test image-text similarity
import pytest

@pytest.mark.parametrize("image_path,text", IMAGE_TEXT_PAIRS)
def test_image_text_similarity(embedder, image_path, text):
    # Load image
    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    # Encode: embedder.encode should accept a list of payloads;
    # here we pass one image and one text.
    embeddings = embedder.encode([img, text])

    img_emb, text_emb = embeddings

    similarity = cosine_similarity(
        img_emb.reshape(1, -1),
        text_emb.reshape(1, -1)
    )[0][0]

    assert similarity > 0.3
