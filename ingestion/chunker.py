# ingestion/chunker.py
from typing import List

def split_by_size(text: str, max_chars: int = 800) -> List[str]:
    # 尽量按段落/句子切分，保证语义完整
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def test():
    text = "这是一个测试文本。" * 100
    chunks = split_by_size(text, max_chars=50)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:30]}...")
if __name__ == "__main__":
    test()
