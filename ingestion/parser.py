# ingestion/parser.py
import pdfplumber
from docx import Document
from pathlib import Path

def parse_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def parse_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_file(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return parse_pdf(str(p))
    elif p.suffix.lower() in [".docx", ".doc"]:
        return parse_docx(str(p))
    else:
        return p.read_text(encoding="utf-8", errors="ignore")