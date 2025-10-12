"""Document parsing utilities without heavy textract dependency."""

from __future__ import annotations

import subprocess
from pathlib import Path


def parse_pdf(path: str) -> str:
    try:
        import pymupdf  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            raise RuntimeError(
                "pdf parsing requires pymupdf or pypdf; install one of them."
            )
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    doc = pymupdf.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def parse_docx(path: str) -> str:
    from docx import Document  # type: ignore

    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def parse_doc(path: str) -> str:
    # Prefer pypandoc for legacy doc files
    try:
        import pypandoc  # type: ignore

        return pypandoc.convert_file(path, "plain")
    except (ImportError, OSError):
        pass

    # Fall back to unoconv if available
    try:
        result = subprocess.run(
            ["unoconv", "-f", "txt", path],
            check=True,
            capture_output=True,
        )
        return result.stdout.decode("utf-8")
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "Unable to parse .doc file; install pypandoc or unoconv with LibreOffice."
        )


def parse_file(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(str(p))
    if suffix == ".docx":
        return parse_docx(str(p))
    if suffix == ".doc":
        return parse_doc(str(p))
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
