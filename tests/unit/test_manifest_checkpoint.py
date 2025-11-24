"""Tests for manifest parsing and checkpoint bookkeeping."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ingestion.manifest import CheckpointStore, DocumentManifest, ManifestEntry, build_manifest


def _write_manifest(tmp_path: Path, rows):
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_id", "source_path", "note"])
        writer.writerows(rows)
    return manifest_path


def test_manifest_loader(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path, [["doc-1", "/tmp/a.pdf", "A"], ["doc-2", "/tmp/b.docx", "B"]])
    manifest = DocumentManifest(manifest_path)
    entries = list(manifest.entries())
    assert len(entries) == 2
    assert isinstance(entries[0], ManifestEntry)
    assert entries[0].doc_id == "doc-1"
    assert entries[0].metadata["note"] == "A"


def test_checkpoint_store(tmp_path: Path):
    db_path = tmp_path / "checkpoints.sqlite"
    store = CheckpointStore(db_path)
    store.mark_in_progress("doc-1")
    assert store.get_status("doc-1") == "in_progress"
    store.mark_failed("doc-1", "boom")
    assert store.get_status("doc-1") == "failed"
    store.mark_completed("doc-1", chunks=42)
    assert store.get_status("doc-1") == "completed"
    store.close()


def test_build_manifest(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_a = data_dir / "a.txt"
    file_b = data_dir / "b.txt"
    file_a.write_text("A")
    file_b.write_text("B")
    manifest_path = tmp_path / "generated.csv"
    count = build_manifest([data_dir], manifest_path)
    assert count == 2
    manifest = DocumentManifest(manifest_path)
    entries = list(manifest.entries())
    assert len(entries) == 2
