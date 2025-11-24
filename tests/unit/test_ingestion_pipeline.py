"""Unit tests for the multimodal ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from ingestion.modality import Modality, detect_modality
from ingestion.pipeline import MultimodalIngestionPipeline
from ingestion.chunk_types import RawChunk
from ingestion.embedders import BaseEmbedder, EmbedderRegistry


class DummyTextEmbedder(BaseEmbedder):
    """Deterministic embedder for testing."""

    def __init__(self):
        self.model_name = "dummy-text"
        self.modality = Modality.TEXT

    def encode(self, chunks: Sequence[RawChunk], payloads: Sequence[str] | None = None):  # type: ignore[override]
        texts = payloads or [chunk.as_text() for chunk in chunks]
        vectors = []
        for text in texts:
            seed = float(len(text))
            vectors.append([seed, seed / 2, seed / 4])
        return np.asarray(vectors, dtype=float)


class DummyVisionEmbedder(BaseEmbedder):
    def __init__(self):
        self.model_name = "dummy-vision"
        self.modality = Modality.IMAGE

    def encode(self, chunks: Sequence[RawChunk], payloads: Sequence[bytes] | None = None):  # type: ignore[override]
        data = payloads or [chunk.as_bytes() for chunk in chunks]
        vectors = []
        for item in data:
            size = float(len(item))
            vectors.append([size, size / 2, size / 4])
        return np.asarray(vectors, dtype=float)


class DummyTabularEmbedder(BaseEmbedder):
    def __init__(self):
        self.model_name = "dummy-tabular"
        self.modality = Modality.TABULAR

    def encode(self, chunks: Sequence[RawChunk], payloads: Sequence[str] | None = None):  # type: ignore[override]
        texts = payloads or [chunk.as_text() for chunk in chunks]
        vectors = []
        for text in texts:
            seed = float(len(text))
            vectors.append([seed, seed / 3, seed / 5])
        return np.asarray(vectors, dtype=float)


class DummyVectorClient:
    """Captures namespace creation and upserts."""

    def __init__(self):
        self.namespaces = {}
        self.upserts = []

    def ensure_namespace(self, namespace: str, dim: int):
        self.namespaces[namespace] = dim

    def upsert(self, records):
        self.upserts.extend(records)


def test_detect_modality_basic():
    assert detect_modality(Path("doc.pdf")) == Modality.TEXT
    assert detect_modality("photo.PNG") == Modality.IMAGE
    assert detect_modality("clip.MP3") == Modality.AUDIO
    assert detect_modality("movie.mov") == Modality.VIDEO
    assert detect_modality("data.xlsx") == Modality.TABULAR


def test_pipeline_processes_text(tmp_path: Path):
    source = tmp_path / "sample.txt"
    source.write_text("第一段。\n\n第二段。")

    registry = EmbedderRegistry([DummyTextEmbedder(), DummyTabularEmbedder()])
    vector_client = DummyVectorClient()
    pipeline = MultimodalIngestionPipeline(
        embedder_registry=registry,
        vector_client=vector_client,
        chunk_size=8,
        namespace="test_ns",
    )
    report = pipeline.ingest([source])

    assert report.namespace == "test_ns"
    assert report.processed_files == 1
    assert report.written_chunks > 0
    assert not report.skipped_files
    assert vector_client.namespaces["test_ns"] == 3


def test_pipeline_skips_unsupported_modalities(tmp_path: Path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fakepng")

    registry = EmbedderRegistry([DummyTextEmbedder()])
    pipeline = MultimodalIngestionPipeline(
        embedder_registry=registry,
        vector_client=DummyVectorClient(),
    )
    report = pipeline.ingest([image_path])

    assert report.processed_files == 1
    assert report.written_chunks == 0
    assert report.skipped_files[0].reason == "unsupported_modality"


def test_pipeline_records_image_without_embedder(monkeypatch, tmp_path: Path):
    source = tmp_path / "sample.txt"
    source.write_text("placeholder")

    registry = EmbedderRegistry([DummyTextEmbedder()])
    pipeline = MultimodalIngestionPipeline(
        embedder_registry=registry,
        vector_client=DummyVectorClient(),
    )

    fake_chunks = [
        RawChunk(chunk_id="chunk-text", modality=Modality.TEXT, content="文本内容", metadata={}),
        RawChunk(chunk_id="chunk-img", modality=Modality.IMAGE, content=b"img-bytes", metadata={}),
    ]

    monkeypatch.setattr(
        MultimodalIngestionPipeline,
        "_extract_chunks",
        lambda self, path, modality: fake_chunks,
    )

    report = pipeline.ingest([source])
    result = report.results[0]

    assert result.chunk_counts[Modality.TEXT.value] == 1
    assert result.skipped_chunks["image_no_embedder"] == 1


def test_pipeline_embeds_images_when_embedder_available(monkeypatch, tmp_path: Path):
    source = tmp_path / "sample.txt"
    source.write_text("placeholder")

    registry = EmbedderRegistry([DummyTextEmbedder(), DummyVisionEmbedder(), DummyTabularEmbedder()])
    vector_client = DummyVectorClient()
    pipeline = MultimodalIngestionPipeline(
        embedder_registry=registry,
        vector_client=vector_client,
        namespace="base_ns",
    )

    fake_chunks = [
        RawChunk(chunk_id="chunk-img", modality=Modality.IMAGE, content=b"abcd", metadata={}),
    ]

    monkeypatch.setattr(
        MultimodalIngestionPipeline,
        "_extract_chunks",
        lambda self, path, modality: fake_chunks,
    )

    report = pipeline.ingest([source])
    result = report.results[0]

    assert result.chunk_counts[Modality.IMAGE.value] == 1
    assert vector_client.namespaces["base_ns"] == 3


def test_pipeline_embeds_tabular(monkeypatch, tmp_path: Path):
    source = tmp_path / "sample.csv"
    source.write_text("col1,col2\n1,2\n3,4")

    registry = EmbedderRegistry([DummyTextEmbedder(), DummyTabularEmbedder()])
    vector_client = DummyVectorClient()
    pipeline = MultimodalIngestionPipeline(
        embedder_registry=registry,
        vector_client=vector_client,
        namespace="base",
    )

    fake_chunks = [
        RawChunk(chunk_id="tab-1", modality=Modality.TABULAR, content="a,b\n1,2", metadata={}),
    ]

    monkeypatch.setattr(
        MultimodalIngestionPipeline,
        "_extract_chunks",
        lambda self, path, modality: fake_chunks,
    )

    report = pipeline.ingest([source])
    result = report.results[0]

    assert result.chunk_counts[Modality.TABULAR.value] == 1
    assert vector_client.namespaces["base"] == 3
