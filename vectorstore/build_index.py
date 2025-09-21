# vectorstore/build_index.py
import faiss
import numpy as np
import json
from pathlib import Path

INDEX_PATH = "vectorstore/faiss.index"
META_PATH = "vectorstore/metadatas.jsonl"

def build_index(embeddings: np.ndarray, metadatas: list, dimension: int):
    idx = faiss.IndexFlatIP(dimension)  # 内积近似（使用归一化向量）
    faiss.normalize_L2(embeddings)
    idx.add(embeddings)
    faiss.write_index(idx, INDEX_PATH)
    # 保存元数据（按向量顺序）
    Path("vectorstore").mkdir(exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("Saved index and metadata")

def load_index():
    idx = faiss.read_index(INDEX_PATH)
    metas = [json.loads(line) for line in open(META_PATH, encoding="utf-8")]
    return idx, metas

def search(query_emb: np.ndarray, topk=5):
    idx, metas = load_index()
    faiss.normalize_L2(query_emb)
    D, I = idx.search(query_emb, topk)
    results = []
    for ids, scores in zip(I, D):
        hits = []
        for _id, sc in zip(ids, scores):
            hits.append({"score": float(sc), "meta": metas[_id]})
        results.append(hits)
    return results

def test():
    # 测试索引和搜索
    dim = 128
    data = np.random.random((100, dim)).astype('float32')
    data /= np.linalg.norm(data, axis=1, keepdims=True)  # 归一化
    metas = [{"id": i} for i in range(100)]
    build_index(data, metas, dim)
    
    query = np.random.random((1, dim)).astype('float32')
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    results = search(query, topk=3)
    print(results)
if __name__ == "__main__":
    test()