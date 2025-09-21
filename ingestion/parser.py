# ingestion/parser.py
import pdfplumber
from docx import Document
from pathlib import Path
import textract


def parse_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def parse_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_doc(path: str) -> str:
    try:
        # 使用 textract 解析 .doc 文件
        text = textract.process(path).decode('utf-8')
        return text
    except Exception as e:
        print(f"无法解析 .doc 文件 {path}: {e}")
        return ""

def parse_file(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return parse_pdf(str(p))
    elif p.suffix.lower() == ".docx":
        return parse_docx(str(p))
    elif p.suffix.lower() == ".doc":
        return parse_doc(str(p))        
    else:
        return p.read_text(encoding="utf-8", errors="ignore")

def main():
    DATA_RAW = "data/raw/现场流程相关文件"
    texts = []
    for p in list(Path(DATA_RAW).glob("*")):
        txt = parse_file(str(p))
        texts.append(txt)
    print(texts[3:4])
    
if __name__ == "__main__":
    main()