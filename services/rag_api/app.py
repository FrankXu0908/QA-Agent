"""FastAPI application orchestrating retrieval-augmented generation."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prometheus_fastapi_instrumentator import Instrumentator

from core.config import get_settings
from rag_pipeline.retriever import retrieve as run_retrieve


settings = get_settings()


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    topk: int = Field(default=3, ge=1, le=10)
    namespace: Optional[str] = None
    scenario: Optional[str] = Field(default="technical")
    images: Optional[List[str]] = None



class QAResponse(BaseModel):
    answer: str
    sources: List[dict]


def build_prompt(question: str, contexts: List[dict]) -> str:
    """Compose the final prompt shown to the LLM."""
    ctx_texts = "\n\n".join(
        [f"内容: {ctx['metadata'].get('text', '')}" for ctx in contexts]
    )
    prompt = f"""你是质量管理文档（FMEA、PPAP 等）方面的专家。请基于下面的文档片段回答用户问题。如果文档里没有明确答案，请说明并给出建议的下一步（例如查阅某个标准或联系质量工程师）。
--- 文档片段开始 ---
{ctx_texts}
--- 文档片段结束 ---
用户问题: {question}
请用简体中文回答，回答中不要编造事实，如不确定请标注“不确定”。"""
    return prompt


SCENARIO_SETTINGS = {
    "technical": {"temperature": 0.1, "max_tokens": 500},
    "creative": {"temperature": 0.7, "max_tokens": 500},
    "story": {"temperature": 0.9, "max_tokens": 800},
    "code": {"temperature": 0.0, "max_tokens": 400},
    "translation": {"temperature": 0.3, "max_tokens": 300},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.settings = settings
    yield
    # Shutdown hooks (placeholder for cleanup)


app = FastAPI(title="QA-Agent RAG API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    scenario = SCENARIO_SETTINGS.get(req.scenario or "technical", SCENARIO_SETTINGS["technical"])
    try:
        hits = run_retrieve(
            req.question,
            images=req.images,
            topk=req.topk,
            namespace=req.namespace or settings.vectorstore_default_namespace,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve context: {exc}") from exc

    prompt = build_prompt(req.question, hits)
    payload = {
        "prompt": prompt,
        "max_tokens": scenario["max_tokens"],
        "temperature": scenario["temperature"],
    }
    try:
        response = requests.post(settings.llm_proxy_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("text") or data
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to query LLM proxy: {exc}") from exc

    return QAResponse(answer=answer, sources=hits)


@app.post("/multimodal/qa")
def multimodal_placeholder():
    """Placeholder endpoint for future multimodal flows."""
    raise HTTPException(status_code=501, detail="Multimodal QA is not implemented yet.")
