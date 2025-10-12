# frontend/streamlit_app.py
import os
import streamlit as st
import requests

QA_API = os.getenv("QA_API", "http://localhost:8000/qa")  # rag_pipeline 服务地址

st.set_page_config(page_title="质量文档 QA", layout="wide")
st.title("质量文档问答（FMEA / PPAP） — Phase1")

question = st.text_area("请输入问题（中文）", height=120)
topk = st.slider("检索片段数量 (top-k)", 1, 10, 3)
scenario = st.selectbox(
    "回答风格",
    options=["technical", "creative", "story", "code", "translation"],
    index=0,
)

if st.button("问答"):
    with st.spinner("检索并询问模型中..."):
        payload = {"question": question, "topk": topk, "scenario": scenario}
        resp = requests.post(QA_API, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        st.subheader("模型回复")
        st.write(data.get("answer", ""))
        st.subheader("检索到的片段（供审阅）")
        for i, hit in enumerate(data.get("sources", [])):
            metadata = hit.get("metadata", {})
            st.markdown(f"**片段 {i+1}** (score: {hit.get('score', 0):.3f})")
            st.write(metadata.get("text", ""))
            st.write("来源:", metadata.get("source", "unknown"))
