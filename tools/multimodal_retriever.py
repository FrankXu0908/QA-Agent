"""LangChain tool wrapping the multimodal retriever."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from rag_pipeline.retriever import retrieve as _retrieve

from pydantic import BaseModel
from enum import Enum

class NamespaceEnum(str, Enum):
    kb_default = "kb_default"
    # kb_quality = "kb_quality"
    # kb_supply_chain = "kb_supply_chain"

class RetrieveInput(BaseModel):
    question: str
    images: Optional[List[str]] = None
    topk: int = 5
    namespace: NamespaceEnum = NamespaceEnum.kb_default


@tool("multimodal_retrieve", description="""Retrieve multimodal evidence from the vector store. 
      Use when the user asks about content in documents such as PDF, Excel, PPT, DOCX, or 
      images or if the question is related specifically to Enterprise knowledge.""",
      return_direct=False,args_schema=RetrieveInput)
def multimodal_retrieve(
    question: str = "",
    images: Optional[List[str]] = None,
    topk: int = 5,
    namespace=NamespaceEnum.kb_default,
) -> List[dict]:
    """
    Retrieve multimodal evidence from the vector store.

    - question: text query (SigLIP text encoder)
    - images: optional list of base64-encoded images (SigLIP vision encoder)
    - topk: number of hits to return
    - namespace: vector store namespace
    """
    return _retrieve(question, images=images or [], topk=topk, namespace=namespace.value)


__all__ = ["multimodal_retrieve"]
