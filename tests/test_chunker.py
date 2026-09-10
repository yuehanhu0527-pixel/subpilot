"""chunker.py：chunk 大小上限、重叠、标题捕获。"""

from app.rag.chunker import chunk_sections
from app.rag.models import ParsedSection


def _section(text, name="doc.txt", page=None, heading=None):
    return ParsedSection(text=text, source_name=name, page=page, heading=heading)


def test_chunks_stay_within_target_size():
    text = "\n\n".join(
        f"Paragraph {i} with some words to fill space reasonably well." for i in range(60)
    )
    chunks = chunk_sections([_section(text)], target_chars=500, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 550 for c in chunks)


def test_adjacent_chunks_overlap():
    text = "\n\n".join(
        f"Unique sentence number {i} for overlap checking purposes." for i in range(40)
    )
    chunks = chunk_sections([_section(text)], target_chars=300, overlap_chars=80)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        shared = set(a.text.split()) & set(b.text.split())
        assert len(shared) > 3, "相邻 chunk 应共享重叠段落"


def test_markdown_heading_is_captured():
    chunks = chunk_sections([_section("## Lunch Duty\n\nRules for lunch supervision.")])
    assert chunks[0].heading == "Lunch Duty"


def test_section_heading_is_used():
    chunks = chunk_sections([_section("Some rules text here.", heading="Bathroom Passes")])
    assert chunks[0].heading == "Bathroom Passes"


def test_short_section_produces_one_chunk_with_source():
    chunks = chunk_sections([_section("Just a short line.", name="policy.pdf", page=2)])
    assert len(chunks) == 1
    assert chunks[0].source_name == "policy.pdf"
    assert chunks[0].page == 2


def test_empty_sections_produce_no_chunks():
    assert chunk_sections([_section("   \n  ")]) == []


def test_single_oversized_paragraph_is_split():
    huge = "word " * 3000  # 单个无换行的超长段落
    chunks = chunk_sections([_section(huge)], target_chars=500, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 550 for c in chunks)
