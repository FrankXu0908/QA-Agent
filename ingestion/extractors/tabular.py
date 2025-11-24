"""Tabular data extraction into textual chunks."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ingestion.chunker import split_by_size
from ingestion.chunk_types import RawChunk
from multimodal.base import Modality


def _load_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pandas is required for tabular extraction") from exc
    return pd


def extract_tabular_chunks(path: Path, *, chunk_size: int = 800) -> List[RawChunk]:
    pd = _load_pandas()
    suffix = path.suffix.lower()
    chunks: List[RawChunk] = []

    def _emit(text: str, meta: dict) -> None:
        for idx, chunk_text in enumerate(split_by_size(text, max_chars=chunk_size)):
            chunk_id = f"{path.stem}-{meta.get('source', 'table')}-{idx}"
            chunks.append(
                RawChunk(
                    chunk_id=chunk_id,
                    modality=Modality.TABULAR,
                    content=chunk_text,
                    metadata={
                        "modality": Modality.TABULAR.value,
                        **meta,
                    },
                )
            )

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else None
        df = pd.read_csv(path, sep=sep)
        text = df.to_csv(index=False)
        _emit(text, {"source": "csv"})
    elif suffix in {".xls", ".xlsx"}:
        excel = pd.ExcelFile(path)
        for sheet in excel.sheet_names:
            df = excel.parse(sheet_name=sheet)
            text = df.to_csv(index=False)
            _emit(text, {"source": sheet})
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        text = df.to_csv(index=False)
        _emit(text, {"source": "parquet"})
    else:
        # fallback to plain text read (e.g., JSON tables)
        raw = path.read_text(encoding="utf-8", errors="ignore")
        _emit(raw, {"source": "raw"})

    return chunks


__all__ = ["extract_tabular_chunks"]
