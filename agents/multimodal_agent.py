"""
LangChain function-calling agent harness that uses the multimodal retriever tool.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import requests

from core.config import get_settings
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, AIMessage
from langchain.tools import tool

from tools.multimodal_retriever import multimodal_retrieve


SYSTEM_PROMPT = """
你是企业内部的智能检索代理。遵循以下规则：

1. 当问题涉及企业内部知识，涉及文档内容（PDF、Excel、PPT、DOCX 等）必须调用工具 multimodal_retrieve。
2. 工具返回后，你必须基于内容回答，不要直接复制。
3. 如果用户只是问常识或闲聊，可以不调用工具。
4. 回答要简洁、结构化。
"""

weather_api = get_settings().weather_api_key

@tool("get_weather", description="Get the weather for a given city. Use when the user asks about current weather conditions in a specific location.", return_direct=False)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    BASE_URL = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": weather_api,
        "q": city,
    }
    resp = requests.get(BASE_URL, params=params)
    return resp.json()

def build_llm(model: str = "gpt-oss:20b") -> ChatOllama: # need a model that supports function calling
    # settings = get_settings()
    # api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
    return ChatOllama(
        model=model,
        validate_model_on_init=True,
        temperature=0.3,
        base_url="http://localhost:11434"
    )


def build_agent(llm: Optional[ChatOllama] = None):
    llm = llm or build_llm()
    tools = [multimodal_retrieve, get_weather]
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multimodal QA agent.")
    parser.add_argument("question", help="User question to answer")
    parser.add_argument(
        "--namespace",
        default="kb_default",
        help="Vector store namespace.",
    )
    args = parser.parse_args()

    agent_executor = build_agent()

    response = agent_executor.invoke({"messages": [args.question]})
    print(response["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())