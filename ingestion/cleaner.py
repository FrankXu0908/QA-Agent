# ingestion/cleaner.py
import re
import unicodedata

def clean_text(text: str) -> str:
    # 删除多余空白，统一换行，去掉页眉页脚（简单策略）
    # 移除控制字符
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # 标准化 Unicode 字符
    text = unicodedata.normalize('NFKC', text)
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去掉连续的空格
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # 去掉非打印字符
    text = ''.join(ch for ch in text if ord(ch) >= 9)
    return text.strip()