"""loader.py：三种文档格式的解析与错误处理。"""

import pytest

from app.rag.loader import load_document
from tests.helpers import make_docx, make_text_pdf


def test_load_pdf_extracts_text_with_page_number(tmp_path):
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Period 1: Math", "Period 2: Science"])
    sections = load_document(pdf)
    assert len(sections) == 1
    assert sections[0].page == 1
    assert "Period 1: Math" in sections[0].text
    assert "Period 2: Science" in sections[0].text


def test_load_docx_extracts_heading_and_body(tmp_path):
    docx = make_docx(tmp_path / "plan.docx", "Warm-up", "Students complete worksheet A.")
    sections = load_document(docx)
    assert any(s.heading == "Warm-up" for s in sections)
    assert any("worksheet A" in s.text for s in sections)


def test_load_txt_returns_single_section_without_page(tmp_path):
    txt = tmp_path / "policy.txt"
    txt.write_text("Line one.\nLine two.")
    sections = load_document(txt)
    assert len(sections) == 1
    assert sections[0].page is None
    assert "Line two." in sections[0].text


def test_markdown_file_is_supported(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Hello\nWorld")
    sections = load_document(md)
    assert sections[0].text


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "notes.xlsx"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "nope.pdf")


def test_pdf_without_extractable_text_raises(tmp_path):
    # 图片型/扫描件 PDF：没有文本层，必须明确报错而不是悄悄返回空
    pdf = make_text_pdf(tmp_path / "scan.pdf", [])
    with pytest.raises(ValueError, match="No extractable text"):
        load_document(pdf)
