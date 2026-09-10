"""RAG 数据模型：解析段、chunk、检索结果。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSection:
    """文档解析后的一段文本（一页 PDF / 一个 DOCX 段落 / 一整篇 TXT）。"""

    text: str
    source_name: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class TextChunk:
    """切分后的检索单元，带来源信息。"""

    id: str
    text: str
    source_name: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """检索命中的片段，带来源信息与相似度。"""

    text: str
    source_name: str
    page: int | None = None
    heading: str | None = None
    similarity: float = 0.0

    def citation(self) -> str:
        """人类可读的来源引用，如 'policy.pdf (p.3) · § Cafeteria'。"""
        parts = [self.source_name]
        if self.page is not None:
            parts.append(f"(p.{self.page})")
        if self.heading:
            parts.append(f"§ {self.heading}")
        return " · ".join(parts)


@dataclass(frozen=True)
class SearchResult:
    """一次检索的完整结果；空结果时 found=False 并带明确提示。"""

    chunks: tuple[RetrievedChunk, ...]

    @property
    def found(self) -> bool:
        return bool(self.chunks)

    @property
    def message(self) -> str:
        """空结果时的明确提示（反幻觉要求：检索不到就明说）。"""
        return "" if self.found else "No relevant information found."
