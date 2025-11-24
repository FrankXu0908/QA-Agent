"""Manifest and checkpoint helpers for large-scale ingestion."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


@dataclass
class ManifestEntry:
    """Single document entry from the manifest."""

    doc_id: str
    source_path: Path
    metadata: Dict[str, str] = field(default_factory=dict)


class DocumentManifest:
    """Parses a CSV manifest describing documents to ingest."""

    REQUIRED_COLUMNS = ("doc_id", "source_path")

    def __init__(self, manifest_path: Path):
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self.path = manifest_path

    def entries(self) -> Iterator[ManifestEntry]:
        with self.path.open("r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Manifest file must include a header row.")
            missing = [col for col in self.REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
            for row in reader:
                doc_id = (row.get("doc_id") or "").strip()
                source = (row.get("source_path") or "").strip()
                if not doc_id or not source:
                    continue
                metadata = {k: v for k, v in row.items() if k not in self.REQUIRED_COLUMNS}
                source_path = Path(source).expanduser()
                yield ManifestEntry(doc_id=doc_id, source_path=source_path, metadata=metadata)


class CheckpointStore:
    """Persists document-level status so ingestion can resume."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                doc_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                chunks_written INTEGER DEFAULT 0,
                last_error TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()
        
    def close(self) -> None:
        self._conn.close()

    def get_status(self, doc_id: str) -> Optional[str]:
        cursor = self._conn.execute("SELECT status FROM checkpoints WHERE doc_id=?", (doc_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def mark_in_progress(self, doc_id: str) -> None:
        self._upsert(doc_id, "in_progress")

    def mark_failed(self, doc_id: str, error: str) -> None:
        self._upsert(doc_id, "failed", error=error)

    def mark_completed(self, doc_id: str, chunks: int) -> None:
        self._upsert(doc_id, "completed", chunks_written=chunks)

    def _upsert(self, doc_id: str, status: str, *, error: Optional[str] = None, chunks_written: int = 0) -> None:
        self._conn.execute(
            """
            INSERT INTO checkpoints (doc_id, status, chunks_written, last_error, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(doc_id) DO UPDATE SET
                status=excluded.status,
                chunks_written=excluded.chunks_written,
                last_error=excluded.last_error,
                updated_at=CURRENT_TIMESTAMP
            """,
            (doc_id, status, chunks_written, error),
        )
        self._conn.commit()


def build_manifest(
    sources: Iterable[str | Path],
    output_path: Path,
    *,
    recursive: bool = True,
    encoding: str = "utf-8",
) -> int:
    """
    Build a simple manifest CSV from the provided files/directories.

    Each discovered file becomes a row with a generated doc_id and absolute source path.
    Returns the number of entries written.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for entry in sources:
        entry_path = Path(entry).expanduser()
        if entry_path.is_dir():
            iterator = entry_path.rglob("*") if recursive else entry_path.glob("*")
            files.extend(p for p in iterator if p.is_file())
        elif entry_path.is_file():
            files.append(entry_path)

    files = sorted(set(files))
    count = 0
    with output_path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_id", "source_path"])
        for idx, file_path in enumerate(files, start=1):
            doc_id = f"doc-{idx:06d}"
            writer.writerow([doc_id, str(file_path.resolve())])
            count += 1
    return count

__all__ = ["ManifestEntry", "DocumentManifest", "CheckpointStore", "build_manifest"]
