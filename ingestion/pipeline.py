"""Multimodal ingestion pipeline covering extraction, cleaning, and embedding."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from clients.vectorstore_client import VectorStoreClient
from core.config import get_settings
from core.models import VectorRecord
from ingestion.chunker import split_by_size
from ingestion.cleaner import clean_text
from ingestion.embedders import BaseEmbedder, EmbedderRegistry, get_default_registry
from ingestion.extractors import extract_pdf_chunks
from ingestion.modality import Modality, detect_modality
from ingestion.parser import parse_file
from ingestion.chunk_types import RawChunk
from ingestion.vector_writer import build_record

logger = logging.getLogger(__name__)


@dataclass
class FileIngestionResult:
    """Summary of an individual file ingestion attempt."""

    path: Path
    modality: Modality
    chunks_written: int = 0
    skipped: bool = False
    reason: Optional[str] = None
    chunk_counts: Dict[str, int] = field(default_factory=dict)
    skipped_chunks: Dict[str, int] = field(default_factory=dict)


@dataclass
class IngestionReport:
    """Aggregated ingestion outcome returned to callers."""

    namespace: str
    results: List[FileIngestionResult] = field(default_factory=list)

    @property
    def processed_files(self) -> int:
        return len(self.results)

    @property
    def written_chunks(self) -> int:
        return sum(sum(r.chunk_counts.values()) for r in self.results if not r.skipped)

    @property
    def skipped_files(self) -> List[FileIngestionResult]:
        return [r for r in self.results if r.skipped]


class MultimodalIngestionPipeline:
    """High-level pipeline orchestrating extraction, modality detection, cleaning, and embedding."""

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        namespace: Optional[str] = None,
        embedder: Optional[BaseEmbedder] = None,
        embedder_registry: Optional[EmbedderRegistry] = None,
        vector_client: Optional[VectorStoreClient] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size
        self.namespace = namespace or settings.vectorstore_default_namespace
        if embedder_registry is not None:
            self.registry = embedder_registry
        else:
            if embedder is not None:
                reg = EmbedderRegistry()
                reg.register(embedder)
                self.registry = reg
            else:
                self.registry = get_default_registry()
        self.vector_client = vector_client or VectorStoreClient()
        self._ensured_dims: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest(self, paths: Iterable[str | Path], namespace: Optional[str] = None) -> IngestionReport:
        """Run the pipeline for the given paths."""
        ns = namespace or self.namespace
        report = IngestionReport(namespace=ns)

        for path in paths:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning("Skipping missing file: %s", path)
                report.results.append(
                    FileIngestionResult(
                        path=path_obj,
                        modality=Modality.TEXT,
                        skipped=True,
                        reason="missing_file",
                        chunk_counts={},
                        skipped_chunks={},
                    )
                )
                continue
            result = self._process_file(path_obj, ns)
            report.results.append(result)
        return report

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _process_file(self, path: Path, namespace: str) -> FileIngestionResult:
        modality = detect_modality(path)
        logger.debug("Detected modality %s for %s", modality, path)

        try:
            raw_chunks = self._extract_chunks(path, modality)
        except NotImplementedError:
            logger.info("Modality %s not supported yet, skipping %s", modality, path)
            return FileIngestionResult(
                path=path,
                modality=modality,
                skipped=True,
                reason="unsupported_modality",
                chunk_counts={},
                skipped_chunks={},
            )
        except Exception as exc:  # pragma: no cover - surface extraction issues
            logger.exception("Extraction failed for %s: %s", path, exc)
            return FileIngestionResult(
                path=path,
                modality=modality,
                skipped=True,
                reason="extraction_failed",
                chunk_counts={},
                skipped_chunks={},
            )

        if not raw_chunks:
            return FileIngestionResult(
                path=path,
                modality=modality,
                skipped=True,
                reason="no_chunks",
                chunk_counts={},
                skipped_chunks={},
            )

        grouped = self._group_chunks(raw_chunks)
        chunk_counts: Dict[str, int] = {}
        skipped_counts: Dict[str, int] = defaultdict(int)
        total_records: List[VectorRecord] = []

        for mod, chunks in grouped.items():
            if not chunks:
                continue
            embedder = self.registry.get(mod)
            if embedder is None:
                skipped_counts[f"{mod.value}_no_embedder"] += len(chunks)
                continue

            prepared = self._prepare_chunks(mod, chunks, skipped_counts)
            if not prepared:
                continue
            prep_chunks, payloads = prepared
            try:
                embeddings = embedder.encode(prep_chunks, payloads)
            except NotImplementedError:
                skipped_counts[f"{mod.value}_not_implemented"] += len(prep_chunks)
                continue
            target_ns = self._namespace_for_modality(namespace, mod)
            embeddings = np.asarray(embeddings)
            if embeddings.ndim != 2:
                raise ValueError("Embedding output must be 2-dimensional")
            dim = embeddings.shape[1]
            self._ensure_namespace(dim, target_ns)
            records = self._build_records(path, target_ns, mod, embedder, prep_chunks, payloads, embeddings)
            if records:
                self.vector_client.upsert(records)
                total_records.extend(records)
                chunk_counts[mod.value] = len(records)

        if not total_records:
            return FileIngestionResult(
                path=path,
                modality=modality,
                skipped=True,
                reason="no_chunks_embedded",
                chunk_counts=chunk_counts,
                skipped_chunks=dict(skipped_counts),
            )

        logger.info("Ingested %s chunks for %s", len(total_records), path)
        return FileIngestionResult(
            path=path,
            modality=modality,
            chunks_written=len(total_records),
            chunk_counts=chunk_counts,
            skipped_chunks=dict(skipped_counts),
        )

    def _extract_chunks(self, path: Path, modality: Modality) -> List[RawChunk]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from ingestion.extractors import extract_pdf_chunks

            return extract_pdf_chunks(path, chunk_size=self.chunk_size)
        if suffix == ".docx":
            from ingestion.extractors import extract_docx_chunks

            return extract_docx_chunks(path, chunk_size=self.chunk_size)
        if suffix in {".pptx", ".ppt"}:
            from ingestion.extractors import extract_pptx_chunks

            return extract_pptx_chunks(path, chunk_size=self.chunk_size)
        if modality is Modality.IMAGE:
            # Treat standalone image files as a single chunk.
            data = path.read_bytes()
            chunk_id = f"{path.stem}-img"
            return [
                RawChunk(
                    chunk_id=chunk_id,
                    modality=Modality.IMAGE,
                    content=data,
                    metadata={
                        "source": str(path),
                        "modality": Modality.IMAGE.value,
                        "mime": f"image/{suffix.lstrip('.')}",
                    },
                )
            ]
        if modality is Modality.TABULAR:
            from ingestion.extractors import extract_tabular_chunks

            return extract_tabular_chunks(path, chunk_size=self.chunk_size)
        if modality is Modality.TEXT:
            raw_text = parse_file(str(path))
            if not raw_text:
                return []
            chunks = split_by_size(raw_text, max_chars=self.chunk_size)
            raw_chunks: List[RawChunk] = []
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{path.stem}-text-{idx}"
                raw_chunks.append(
                    RawChunk(
                        chunk_id=chunk_id,
                        modality=Modality.TEXT,
                        content=chunk_text,
                        metadata={"source_modality": modality.value},
                    )
                )
            return raw_chunks
        raise NotImplementedError

    def _group_chunks(self, chunks: Sequence[RawChunk]) -> Dict[Modality, List[RawChunk]]:
        grouped: Dict[Modality, List[RawChunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.modality].append(chunk)
        return grouped

    def _prepare_chunks(
        self,
        modality: Modality,
        chunks: Sequence[RawChunk],
        skipped_counts: Dict[str, int],
    ) -> Optional[Tuple[List[RawChunk], Optional[List[str | bytes]]]]:
        prepared_chunks: List[RawChunk] = []
        payloads: Optional[List[str | bytes]] = None

        if modality is Modality.TEXT:
            payloads = []
            for chunk in chunks:
                try:
                    cleaned = clean_text(chunk.as_text())
                except TypeError:
                    skipped_counts["text_binary"] += 1
                    continue
                if not cleaned:
                    skipped_counts["text_empty"] += 1
                    continue
                prepared_chunks.append(chunk)
                payloads.append(cleaned)
            if not prepared_chunks:
                return None
            return prepared_chunks, payloads

        if modality is Modality.IMAGE:
            payloads = []
            for chunk in chunks:
                try:
                    payload = chunk.as_bytes()
                except Exception:
                    skipped_counts["image_invalid"] += 1
                    continue
                if not payload:
                    skipped_counts["image_empty"] += 1
                    continue
                prepared_chunks.append(chunk)
                payloads.append(payload)
            if not prepared_chunks:
                return None
            return prepared_chunks, payloads

        if modality is Modality.TABULAR:
            payloads = []
            for chunk in chunks:
                try:
                    table_text = chunk.as_text()
                except TypeError:
                    skipped_counts["tabular_binary"] += 1
                    continue
                cleaned = clean_text(table_text)
                if not cleaned:
                    skipped_counts["tabular_empty"] += 1
                    continue
                prepared_chunks.append(chunk)
                payloads.append(cleaned)
            if not prepared_chunks:
                return None
            return prepared_chunks, payloads

        # For other modalities (audio/video) we currently skip.
        skipped_counts[f"{modality.value}_unsupported"] += len(chunks)
        return None

    def _namespace_for_modality(self, base_namespace: str, modality: Modality) -> str:
        return base_namespace

    def _ensure_namespace(self, dim: int, namespace: Optional[str] = None) -> None:
        ns = namespace or self.namespace
        known_dim = self._ensured_dims.get(ns)
        if known_dim is not None:
            if known_dim != dim:
                raise ValueError(f"Namespace '{ns}' already ensured with dim={known_dim}, cannot use dim={dim}")
            return
        self.vector_client.ensure_namespace(ns, dim)
        self._ensured_dims[ns] = dim

    def _build_records(
        self,
        path: Path,
        namespace: str,
        modality: Modality,
        embedder: BaseEmbedder,
        prepared_chunks: Sequence[RawChunk],
        payloads: Optional[Sequence[str | bytes]],
        embeddings: np.ndarray,
    ) -> List[VectorRecord]:
        records: List[VectorRecord] = []
        for idx, (raw_chunk, emb_vec) in enumerate(zip(prepared_chunks, embeddings)):
            chunk_id = raw_chunk.chunk_id
            extra_meta = dict(raw_chunk.metadata)
            extra_meta["modality"] = modality.value
            text_field: Optional[str] = None
            if payloads is not None:
                value = payloads[idx]
                if modality in (Modality.TEXT, Modality.TABULAR):
                    text_field = str(value)
                elif modality is Modality.IMAGE:
                    # Keep short placeholder; raw bytes are stored separately.
                    text_field = f"[image] {raw_chunk.metadata.get('source', path.name)}"
            if text_field is None:
                text_field = f"[{modality.value}] {raw_chunk.metadata.get('source', path.name)}"
            metadata = {
                "doc_id": chunk_id,
                "source": str(path),
                "text": text_field,
                "embedding_model": getattr(embedder, "model_name", None),
                "extra": extra_meta,
            }
            record = build_record(chunk_id, namespace, emb_vec.tolist(), metadata)
            records.append(record)
        return records

__all__ = [
    "MultimodalIngestionPipeline",
    "IngestionReport",
    "FileIngestionResult",
]
