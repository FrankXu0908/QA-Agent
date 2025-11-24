"""Batch ingestion runner using manifest + checkpoint configuration."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ingestion.manifest import CheckpointStore, DocumentManifest, build_manifest
from ingestion.models.config import JobConfig, load_config
from ingestion.pipeline import MultimodalIngestionPipeline


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multimodal ingestion pipeline using a manifest.")
    parser.add_argument(
        "--config",
        default="ingestion/config.yaml",
        help="Path to YAML configuration file (default: ingestion/config.yaml).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of manifest entries to process.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        default=False,
        action="store_true",
        help="Rebuild the manifest even if it exists.",
    )
    return parser.parse_args()


def build_pipeline(config: JobConfig) -> MultimodalIngestionPipeline:
    return MultimodalIngestionPipeline(chunk_size=config.vectorstore.chunk_size, namespace=config.vectorstore.namespace)


def run_job(config_path: Path, limit: int | None, rebuild_manifest: bool = False) -> None:
    config = load_config(config_path)
    if not Path(config.manifest.path).exists() or rebuild_manifest:
        logger.info("Manifest not found or rebuild requested, building...")
        count = build_manifest(sources=[Path(config.manifest.input_root)], output_path=Path(config.manifest.path), recursive=True, encoding="utf-8-sig")
        logger.info(f"Manifest built with {count} entries at {config.manifest.path}")
    manifest = DocumentManifest(config.manifest.path)
    checkpoint = CheckpointStore(config.manifest.checkpoint_db)
    pipeline = build_pipeline(config)
    

    processed = 0
    try:
        for entry in manifest.entries():
            # limit control
            if limit is not None and processed >= limit:
                break

            status = checkpoint.get_status(entry.doc_id)
            if status in ("completed", "error"):
                continue

            logger.info("Processing %s (%s)", entry.doc_id, entry.source_path)
            checkpoint.mark_in_progress(entry.doc_id)
            try:
                report = pipeline.ingest([str(entry.source_path)], namespace=config.vectorstore.namespace)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.exception("Failed to ingest %s", entry.doc_id)
                checkpoint.mark_failed(entry.doc_id, str(exc))
                continue

            chunks_written = next((r.chunks_written for r in report.results if r.path == entry.source_path), report.written_chunks)
            checkpoint.mark_completed(entry.doc_id, chunks_written)
            logger.info("Completed %s (chunks=%s)", entry.doc_id, chunks_written)
            processed += 1
    finally:
        checkpoint.close()



def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    try:
        run_job(Path(args.config), args.limit, args.rebuild_manifest)
    except Exception as exc:  # pragma: no cover - CLI entrypoint
        logger.exception("Ingestion job failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
