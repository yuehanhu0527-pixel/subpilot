"""基础分块：段落聚合到目标长度、带重叠、捕获标题。

不依赖任何模型——纯规则切分，可单测。
"""

import re

from .models import ParsedSection, TextChunk

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


def _split_paragraphs(text: str) -> list[str]:
    """按换行拆成非空段落。"""
    return [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


def _hard_split(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """单个超长段落按字符窗口强制切分（带重叠）。"""
    pieces = []
    start = 0
    while start < len(text):
        piece = text[start : start + target_chars].strip()
        if piece:
            pieces.append(piece)
        if start + target_chars >= len(text):
            break
        start += target_chars - overlap_chars
    return pieces


def chunk_sections(
    sections: list[ParsedSection],
    target_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[TextChunk]:
    """把解析后的文本段切分为检索 chunk。

    - 每个 chunk 不超过 target_chars
    - 相邻 chunk 之间带 overlap_chars 量级的重叠（按整段携带，不会从句子中间截断）
    - heading 取 chunk 起始段落的活跃标题（段落自带标题或前置的 Markdown 标题）
    """
    chunks: list[TextChunk] = []
    for section in sections:
        paragraphs = _split_paragraphs(section.text)
        if not paragraphs:
            continue

        # 每个段落绑定"活跃标题"：section 自带标题，或段落前的 Markdown 标题
        bound: list[tuple[str, str | None]] = []
        active = section.heading
        for para in paragraphs:
            match = _HEADING_RE.match(para)
            if match:
                active = match.group(1)
            bound.append((para, active))

        buffer: list[tuple[str, str | None]] = []
        buffer_len = 0
        for para, heading in bound:
            if len(para) > target_chars:
                # 超长段落：先结算 buffer，再把该段落硬切
                if buffer:
                    chunks.append(_make_chunk(section, buffer, len(chunks)))
                    buffer, buffer_len = [], 0
                for piece in _hard_split(para, target_chars, overlap_chars):
                    chunks.append(_make_chunk(section, [(piece, heading)], len(chunks)))
                continue

            if buffer and buffer_len + len(para) + 1 > target_chars:
                # 装不下了：结算，并把尾部 ≤ overlap_chars 的段落带入下一个 chunk
                chunks.append(_make_chunk(section, buffer, len(chunks)))
                carry: list[tuple[str, str | None]] = []
                carry_len = 0
                for p, h in reversed(buffer):
                    if carry_len + len(p) + 1 > overlap_chars:
                        break
                    carry.insert(0, (p, h))
                    carry_len += len(p) + 1
                buffer, buffer_len = carry, carry_len

            buffer.append((para, heading))
            buffer_len += len(para) + 1

        if buffer:
            chunks.append(_make_chunk(section, buffer, len(chunks)))

    return chunks


def _make_chunk(
    section: ParsedSection,
    buffer: list[tuple[str, str | None]],
    idx: int,
) -> TextChunk:
    """把段落缓冲打包成一个 chunk；标题取第一个段落的活跃标题。"""
    return TextChunk(
        id=f"{section.source_name}:{idx}",
        text="\n\n".join(p for p, _ in buffer),
        source_name=section.source_name,
        page=section.page,
        heading=buffer[0][1],
    )
