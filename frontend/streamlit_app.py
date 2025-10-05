# frontend/streamlit_app.py
import os
import streamlit as st
import requests

QA_API = os.getenv("QA_API", "http://localhost:8000/qa")  # rag_pipeline 服务地址

st.set_page_config(page_title="质量文档 QA", layout="wide")
st.title("质量文档问答（FMEA / PPAP） — Phase1")

question = st.text_area("请输入问题（中文）", height=120)
topk = st.slider("检索片段数量 (top-k)", 1, 10, 3)

if st.button("问答"):
    with st.spinner("检索并询问模型中..."):
        resp = requests.post(QA_API, json={"question": question, "topk": topk}, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        st.subheader("模型回复")
        st.write(data.get("answer", {}).get("choices",[])[0].get("text", ""))
        st.subheader("检索到的片段（供审阅）")
        for i, hit in enumerate(data.get("sources", [])):
            st.markdown(f"**片段 {i+1}** (score: {hit['score']:.3f})")
            st.write(hit["meta"].get("text",""))
            st.write("来源:", hit["meta"].get("source","unknown"))