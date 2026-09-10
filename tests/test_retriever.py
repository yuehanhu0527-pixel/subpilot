"""retriever.py：来源引用、相似度阈值过滤、空结果的明确提示。"""

import pytest

from app.rag.models import TextChunk
from app.rag.retriever import Retriever
from app.rag.store import ChromaStore
from tests.helpers import FakeEmbedder

VOCAB = {
    "cafeteria": [1, 0, 0, 0, 0, 0],
    "bathroom": [0, 1, 0, 0, 0, 0],
    "homework": [0, 0, 1, 0, 0, 0],
    "dismissal": [0, 0, 0, 1, 0, 0],
}


@pytest.fixture
def index(collection):
    """一个小型学校文档索引：假 embedder + 临时 Chroma（3 个 chunk）。"""
    store = ChromaStore.ephemeral()
    embedder = FakeEmbedder(VOCAB, dim=6)
    chunks = [
        TextChunk(id="p:0", text="Cafeteria duty is in the main hall.",
                  source_name="policy.pdf", page=3, heading="Cafeteria"),
        TextChunk(id="p:1", text="Bathroom passes require a signed note.",
                  source_name="policy.pdf", page=5, heading="Bathroom Passes"),
        TextChunk(id="l:0", text="Homework is due on Friday.",
                  source_name="lesson_plan.docx", page=None, heading="Homework"),
    ]
    store.add_chunks(collection, chunks, embedder.encode([c.text for c in chunks]))
    return store, embedder, collection


def test_relevant_query_returns_chunk_with_full_source(index):
    store, embedder, collection = index
    result = Retriever(store, embedder).search(collection, "cafeteria")
    assert result.found
    assert result.message == ""
    top = result.chunks[0]
    assert top.source_name == "policy.pdf"
    assert top.page == 3
    assert top.heading == "Cafeteria"
    assert top.similarity > 0.9
    assert "policy.pdf" in top.citation()
    assert "(p.3)" in top.citation()


def test_docx_source_without_page_has_no_page_in_citation(index):
    store, embedder, collection = index
    result = Retriever(store, embedder).search(collection, "homework")
    top = result.chunks[0]
    assert top.source_name == "lesson_plan.docx"
    assert top.page is None
    assert "(p." not in top.citation()


def test_irrelevant_query_returns_explicit_no_results(index):
    store, embedder, collection = index
    result = Retriever(store, embedder).search(collection, "zzz nothing relevant here")
    assert not result.found
    assert result.chunks == ()
    assert result.message == "No relevant information found."


def test_threshold_filters_low_similarity_hits(index):
    store, embedder, collection = index
    # "cafeteria bathroom" 与 cafeteria 文档的余弦相似度 ≈ 0.707
    strict = Retriever(store, embedder, threshold=0.9).search(collection, "cafeteria bathroom")
    assert not strict.found
    # 阈值放松后能命中
    loose = Retriever(store, embedder, threshold=0.5).search(collection, "cafeteria bathroom")
    assert loose.found
    assert loose.chunks[0].heading == "Cafeteria"
