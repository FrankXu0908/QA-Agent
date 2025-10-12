# QA-Agent
A QA-Agent that can assist QA engineers or any employee to clarify QA questions from sophisticated QA documents.

## High-level Architecture

- **Vector Store Service (`services/vectorstore_api`)** – FastAPI microservice exposing namespace management, upsert/delete, and search operations backed by Milvus, designed for independent scaling.
- **RAG Orchestrator (`services/rag_api`)** – FastAPI service handling question ingestion, vector retrieval via the vector store client, prompt construction, and proxying to the LLM backend.
- **Core Layer (`core/`)** – Shared configuration, Pydantic models, and vector store abstractions used by both services and offline ingestion jobs.
- **Ingestion Toolkit (`ingestion/`, `utils/build_all.py`)** – Document parsing/cleaning/chunking utilities that build `VectorRecord` payloads and push them to the vector store service.
- **Multimodal & Schema Placeholders (`multimodal/`, `schemas/`)** – Reserved modules that define the contract surface for future multimodal inputs and versioned data schemas.

Run the full stack with `docker compose up --build`, which now provisions dedicated containers for Milvus, the vector store API, RAG orchestrator, frontend, LLM proxy, and monitoring stack.

### Local development

- Bring up dependencies only: `docker compose up milvus vectorstore`.
- Run the RAG API locally: `uvicorn services.rag_api.app:app --reload`.
- Point `QA_AGENT_VECTORSTORE_API_URL` and `QA_AGENT_LLM_PROXY_URL` to local services via a `.env` file.
- Quick checks: `make test` for unit coverage, `make smoke` for the mocked end-to-end flow.
