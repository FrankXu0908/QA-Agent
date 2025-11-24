"""Tests for multimodal extractors."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")
pptx = pytest.importorskip("pptx")
pandas = pytest.importorskip("pandas")

Document = docx.Document
Inches = docx.shared.Inches
Presentation = pptx.Presentation
PPTInches = pptx.util.Inches

from ingestion.extractors.docx import extract_docx_chunks
from ingestion.extractors.pptx import extract_pptx_chunks
from ingestion.extractors.tabular import extract_tabular_chunks
from multimodal.base import Modality


PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9W6w1roAAAAASUVORK5CYII="
)


def _write_png(path: Path) -> None:
    path.write_bytes(base64.b64decode(PNG_BASE64))


def test_docx_extractor_handles_text_and_images(tmp_path: Path):
    image_path = tmp_path / "tiny.png"
    _write_png(image_path)

    docx_path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("第一段文字")
    doc.add_paragraph("第二段文字")
    doc.add_picture(str(image_path), width=Inches(1))
    doc.save(docx_path)

    chunks = extract_docx_chunks(docx_path, chunk_size=50)
    modalities = {chunk.modality for chunk in chunks}
    assert Modality.TEXT in modalities
    assert Modality.IMAGE in modalities


def test_pptx_extractor_handles_text_and_images(tmp_path: Path):
    image_path = tmp_path / "tiny.png"
    _write_png(image_path)

    pptx_path = tmp_path / "sample.pptx"
    pres = Presentation()
    slide_layout = pres.slide_layouts[5]
    slide = pres.slides.add_slide(slide_layout)
    textbox = slide.shapes.add_textbox(PPTInches(0.5), PPTInches(0.5), PPTInches(5), PPTInches(1))
    textbox.text_frame.text = "幻灯片文字"
    slide.shapes.add_picture(str(image_path), PPTInches(1), PPTInches(2), PPTInches(2), PPTInches(2))
    pres.save(pptx_path)

    chunks = extract_pptx_chunks(pptx_path, chunk_size=50)
    modalities = {chunk.modality for chunk in chunks}
    assert Modality.TEXT in modalities
    assert Modality.IMAGE in modalities


def test_tabular_extractor_reads_csv(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    pandas.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(csv_path, index=False)

    chunks = extract_tabular_chunks(csv_path, chunk_size=50)
    assert chunks
    assert all(chunk.modality is Modality.TABULAR for chunk in chunks)
