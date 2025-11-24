from ingestion.manifest import build_manifest
from pathlib import Path

"""Builds a manifest CSV file from given source paths."""

if __name__ == "__main__":
    data_path = Path("ingestion/data")
    source_path = data_path / "raw"
    output_manifest_path = data_path / "manifest.csv"
    count = build_manifest(sources=[source_path], output_path=output_manifest_path, recursive=True,encoding="utf-8-sig")
    print(f"Manifest built with {count} entries at {output_manifest_path}")
