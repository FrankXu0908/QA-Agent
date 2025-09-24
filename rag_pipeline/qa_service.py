# rag_pipeline/qa_service.py
from pathlib import Path
import sys
import os
# 获取当前文件的父目录的父目录（上一级目录）
parent_dir = Path(__file__).resolve().parent.parent
# 将上一级目录添加到系统路径
sys.path.append(str(parent_dir))
from fastapi import FastAPI
from pydantic import BaseModel
from vectorstore.query_index import retrieve
import requests

app = FastAPI()
LLM_PROXY = os.getenv("LLM_PROXY", "http://model:8001/v1/completions") # llm_proxy 的地址

class QARequest(BaseModel):
    question: str
    topk: int = 3

def build_prompt(question: str, contexts: list):
    # 自定义 Prompt 模板（中文）
    ctx_texts = "\n\n".join([f"内容: {c['meta'].get('text','')}" for c in contexts])
    prompt = f"""你是质量管理文档（FMEA、PPAP 等）方面的专家。请基于下面的文档片段回答用户问题。如果文档里没有明确答案，请说明并给出建议的下一步（例如查阅某个标准或联系质量工程师）。
--- 文档片段开始 ---
{ctx_texts}
--- 文档片段结束 ---
用户问题: {question}
请用简体中文回答，回答中不要编造事实，如不确定请标注“不确定”。"""
    return prompt

@app.post("/qa")
def qa(req: QARequest):
    try:
        hits = retrieve(req.question, topk=req.topk)
        # 取 hits 列表，构建 prompt
        prompt = build_prompt(req.question, hits)
        scenario_settings = {
        "technical": {"temperature": 0.1, "max_tokens": 500},
        "creative": {"temperature": 0.7, "max_tokens": 500},
        "story": {"temperature": 0.9, "max_tokens": 800},
        "code": {"temperature": 0.0, "max_tokens": 400},
        "translation": {"temperature": 0.3, "max_tokens": 300},
        }
        settings  = scenario_settings["technical"]
        payload = {"prompt": prompt, "max_tokens": settings["max_tokens"], "temperature": settings["temperature"]}
        r = requests.post(LLM_PROXY, json=payload, timeout=120)
        r.raise_for_status()
        gen = r.json()
        # 解析返回（依赖 llm_proxy -> 下游服务返回结构）
        # 假设返回是 {"text":"..."}
        answer = gen.get("text") or gen
        return {"answer": answer, "sources": hits}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}