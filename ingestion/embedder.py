# embeddings/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
import os
# 获取项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 模型名可换为你本地下载的 text2vec-large-chinese
EMB_MODEL = os.path.join(MODELS_DIR, "m3e-base-st")  #使用m3e-base-st模型，st是指sentence-transformers格式
class Embedder:
    def __init__(self, model_name=EMB_MODEL):
        print("加载嵌入模型:", model_name)
        self.model = SentenceTransformer(model_name, cache_folder=MODELS_DIR)

    def encode(self, texts):
        # texts: list[str]
        emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return emb  # numpy array

if __name__ == "__main__":
    e = Embedder()
    # 测试文本
    test_texts = [
        "这是一个测试句子。",
        "这是另一个测试句子。",
        "今天天气真好，适合出去散步。",
        "人工智能是当前的热门技术领域。"
    ]
    
    embeddings = e.encode(test_texts)
    print("嵌入形状:", embeddings.shape)
    print(f"第一个句子的嵌入向量: {embeddings[0][:10]}...")