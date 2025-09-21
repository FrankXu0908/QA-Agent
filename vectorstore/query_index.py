# vectorstore/query_index.py
from pathlib import Path
import sys
# 获取当前文件的父目录的父目录（上一级目录）
parent_dir = Path(__file__).resolve().parent.parent
# 将上一级目录添加到系统路径
sys.path.append(str(parent_dir))

from ingestion.embedder import Embedder
from vectorstore.build_index import search
import numpy as np

embedder = Embedder()

def retrieve(text: str, topk=5):
    emb = embedder.encode([text])
    return search(emb, topk=topk)[0]  # 返回 hits 列表

def test():
    query = "项目领料流程是怎样的？"
    results = retrieve(query, topk=3)
    for r in results:
        print(f"Score: {r['score']:.4f}, Source: {r['meta']['source']}, Text: {r['meta']['text'][:50]}...")
        
if __name__ == "__main__":
    test()