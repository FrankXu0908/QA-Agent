# QA-Agent
A QA-Agent that can assist QA engineers or any employee to clarify QA questions from sophisticated QA documents.

## High-level Architecture

- **Vector Store Service (`services/vectorstore_api`)** – FastAPI microservice exposing namespace management, upsert/delete, and search operations backed by Milvus, designed for independent scaling.
- **RAG Orchestrator (`services/rag_api`)** – FastAPI service handling question ingestion, vector retrieval via the vector store client, prompt construction, and proxying to the LLM backend.
- **Core Layer (`core/`)** – Shared configuration, Pydantic models, and vector store abstractions used by both services and offline ingestion jobs.
- **Ingestion Toolkit (`ingestion/`)** – Document parsing/cleaning/chunking utilities that build `VectorRecord` payloads and push them to the vector store service.
- **Multimodal Embedders (`ingestion/embedders/`)** – Registry-driven SigLIP-based encoders for text, images, and tabular payloads used by ingestion and online services.
- **Multimodal & Schema Placeholders (`multimodal/`, `schemas/`)** – Reserved modules that define the contract surface for future multimodal inputs and versioned data schemas.

Run the full stack with `docker compose up --build`, which now provisions dedicated containers for Milvus, the vector store API, RAG orchestrator, frontend, LLM proxy, and monitoring stack.

## Multimodal ingestion pipeline

`ingestion.pipeline.MultimodalIngestionPipeline` now wires the full ingestion workflow (extraction → per-chunk modality detection → cleaning → SigLIP embedding) and relies on a modality-aware embedder registry. Example:

```python
from ingestion.pipeline import MultimodalIngestionPipeline

pipeline = MultimodalIngestionPipeline(chunk_size=800, namespace="kb_default")
report = pipeline.ingest(["data/raw/doc1.pdf", "data/raw/doc2.docx"])
print("Chunks written:", report.written_chunks)
print("Skipped:", [(r.path, r.reason) for r in report.skipped_files])
```

- PDFs, DOCX, and PPTX files are parsed into both text and image chunks. Text payloads are cleaned/embedded; image chunks are pushed through the shared SigLIP vision encoder (byte payloads are preserved in metadata for downstream OCR if needed).
- Other text assets (`.txt/.docx/.md` etc.) continue to be chunked via the existing parser path.
- Tabular files (`.csv`, `.xlsx`, `.parquet`) are flattened into text chunks via the tabular extractor and embedded through the SigLIP-backed tabular encoder so all modalities live in the same vector space.
- Additional modalities (audio/video) surface as skipped chunks with descriptive reasons, making it easy to plug new embedders into the registry.

### Local development

- Bring up dependencies only: `docker compose up milvus vectorstore`.
- Run the RAG API locally: `uvicorn services.rag_api.app:app --reload`.
- Point `QA_AGENT_VECTORSTORE_API_URL` and `QA_AGENT_LLM_PROXY_URL` to local services via a `.env` file.
- Quick checks: `make test` for unit coverage, `make smoke` for the mocked end-to-end flow.
- Optional ingestion extras: install `torch`, `transformers`, `sentence-transformers`, `sentencepiece`, `protobuf`, `pillow`, `pymupdf`, `python-docx`, `python-pptx`, `pandas`, and `openpyxl` to unlock full SigLIP-based multimodal processing.
- Multimodal agent: `python -m agents.multimodal_agent "你的问题？"` (requires Milvus up and `QA_AGENT_LLM_PROXY_URL` set; internally uses the `multimodal_retrieve` tool to gather text/image/table context).

### Manifest-driven ingestion

Large backfills (tens of GB) should use the manifest + checkpoint flow:

1. Copy `ingestion/config.example.yaml` to `ingestion/config.yaml` and adjust the manifest path, checkpoint SQLite file, and namespace.
2. Populate the CSV manifest (see `ingestion/examples/manifest.csv`) with `doc_id` and `source_path` entries.
3. Run the job: `python ingestion/build_all.py --config ingestion/config.yaml`. The script marks document-level checkpoints in SQLite so you can resume safely after interruptions.
