# utils/build_all.py
import json, os, sys
from pathlib import Path
# 获取当前文件的父目录的父目录（上一级目录）
parent_dir = Path(__file__).resolve().parent.parent
# 将上一级目录添加到系统路径
sys.path.append(str(parent_dir))

from ingestion.parser import parse_file
from ingestion.cleaner import clean_text
from ingestion.chunker import split_by_size
from ingestion.embedder import Embedder
from vectorstore.build_index import build_index



DATA_RAW = "data/raw/现场流程相关文件"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

embedder = Embedder()

metadatas = []
texts = []
for p in os.listdir(DATA_RAW):
    path = os.path.join(DATA_RAW, p)
    txt = parse_file(path)
    txt = clean_text(txt)
    chunks = split_by_size(txt, max_chars=800)
    for i, ch in enumerate(chunks):
        meta = {"id": f"{p}__{i}",
                "source": p, 
                "text": ch}
        metadatas.append(meta)
        texts.append(ch)
        with open(os.path.join(OUT, f"{meta['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")

# 生成 embeddings（批量）
embs = embedder.encode(texts)  # numpy array
dim = embs.shape[1]
# build faiss index
build_index(embs, metadatas, dim)
print("构建完成")