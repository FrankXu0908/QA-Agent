# test_similarity.py
from embedder import Embedder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 初始化嵌入器
embedder = Embedder()

# 测试相似度的句子对
similar_pairs = [
    ("我喜欢吃苹果", "苹果是一种水果"),
    ("今天天气很好", "今天是晴朗的一天"),
    ("他很高", "他的身高很高")
]

# 测试不相似的句子对
dissimilar_pairs = [
    ("我喜欢吃苹果", "计算机编程很有趣"),
    ("今天天气很好", "数学是一门科学"),
    ("他很高", "这本书很厚")
]

def test_similarity(pairs, expected_similar=True):
    print(f"\n测试{'相似' if expected_similar else '不相似'}的句子对:")
    
    for text1, text2 in pairs:
        # 编码句子
        embeddings = embedder.encode([text1, text2])
        
        # 计算余弦相似度
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1), 
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        print(f"'{text1}' vs '{text2}': {similarity:.4f}")
        if expected_similar:
            assert similarity > 0.5, f"相似度太低: {similarity}"
        else:
            assert similarity < 0.5, f"相似度太高: {similarity}"

test_similarity(similar_pairs, expected_similar=True)
test_similarity(dissimilar_pairs, expected_similar=False)

print("\n所有测试通过!")