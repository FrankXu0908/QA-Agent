# utils/build_all.py
import sys
import json
import os
from pathlib import Path
from typing import List
# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 获取项目根目录
sys.path.insert(0, project_root)

from clients.vectorstore_client import VectorStoreClient
from core.config import get_settings
from core.models import VectorRecord
from ingestion.chunker import split_by_size
from ingestion.cleaner import clean_text
from ingestion.embedder import Embedder
from ingestion.parser import parse_file
from ingestion.vector_writer import build_record, write_upserts


settings = get_settings()

DATA_RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

embedder = Embedder()
vector_client = VectorStoreClient(base_url="http://localhost:8082")


def build_records(namespace: str, embeddings, metadatas: List[dict]) -> List[VectorRecord]:
    """Build VectorRecord entries from precomputed embeddings."""
    vector_client.ensure_namespace(namespace, embeddings.shape[1])
    records = []
    for emb, meta in zip(embeddings, metadatas):
        record_id = meta.get("doc_id")
        if not record_id:
            raise ValueError("metadata must include 'doc_id'")
        records.append(build_record(record_id, namespace, emb.tolist(), meta))
    return records


def main(namespace: str = settings.vectorstore_default_namespace):
    metadatas: List[dict] = []
    texts: List[str] = []

    for path in DATA_RAW.glob("**/*"):
        if not path.is_file():
            continue
        txt = parse_file(str(path))
        txt = clean_text(txt)
        chunks = split_by_size(txt, max_chars=800)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{path.stem}__{idx}"
            meta = {"doc_id": chunk_id, "source": str(path), "text": chunk}
            metadatas.append(meta)
            texts.append(chunk)
            with open(OUT / f"{chunk_id}.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
                fh.write("\n")

    if not texts:
        print("No documents found; skipping ingestion.")
        return

    embeddings = embedder.encode(texts)
    records = build_records(namespace, embeddings, metadatas)
    write_upserts(records)
    print(f"Ingestion complete for namespace '{namespace}' with {len(records)} records.")


if __name__ == "__main__":
    main()#namespace = "qa"
