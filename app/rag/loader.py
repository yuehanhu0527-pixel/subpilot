"""文档解析：PDF / DOCX / TXT(MD) → ParsedSection 列表。"""

from pathlib import Path

from .models import ParsedSection

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(path: str | Path) -> list[ParsedSection]:
    """解析一个文档文件，返回带来源信息的文本段。

    - PDF：按页抽取文本（page=页码）；无文本层（扫描件）时明确报错。
    - DOCX：按段落抽取（page=None），Heading 样式作为后续段落的标题。
    - TXT/MD：整篇作为一个段落。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext} "
            f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    return _load_text(path)


def _load_pdf(path: Path) -> list[ParsedSection]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(ParsedSection(text=text, source_name=path.name, page=i))
    if not sections:
        raise ValueError(f"No extractable text in PDF (scanned image?): {path.name}")
    return sections


def _load_docx(path: Path) -> list[ParsedSection]:
    import docx

    document = docx.Document(str(path))
    sections = []
    heading = None
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            heading = text
            continue
        sections.append(ParsedSection(text=text, source_name=path.name, heading=heading))
    if not sections:
        raise ValueError(f"No text found in DOCX: {path.name}")
    return sections


def _load_text(path: Path) -> list[ParsedSection]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"File is empty: {path.name}")
    return [ParsedSection(text=text, source_name=path.name)]
