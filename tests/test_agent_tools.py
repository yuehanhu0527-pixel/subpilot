"""tools.py：retrieve_documents 工具（包装 Phase 1 rag.query）。"""

from app.agent.tools import make_retrieve_documents_tool
from app.rag.models import TextChunk
from app.rag.store import ChromaStore
from tests.helpers import FakeEmbedder

VOCAB = {
    "bathroom": [1, 0, 0, 0],
    "cafeteria": [0, 1, 0, 0],
    "homework": [0, 0, 1, 0],
}


def _make_tool_with_chunks(chunks, collection_name="tools-docs"):
    store = ChromaStore.ephemeral()
    embedder = FakeEmbedder(VOCAB, dim=4)
    store.add_chunks(collection_name, chunks, embedder.encode([c.text for c in chunks]))
    return make_retrieve_documents_tool(
        store=store, embedder=embedder, collection_name=collection_name
    )


def test_tool_returns_formatted_hits_with_citations():
    tool = _make_tool_with_chunks(
        [
            TextChunk(
                id="p:0",
                text="Bathroom passes require a signed note from the office.",
                source_name="policy.pdf",
                page=3,
                heading="Bathroom Passes",
            )
        ]
    )
    result = tool.invoke({"query": "bathroom pass policy"})
    assert "Source: policy.pdf · (p.3) · § Bathroom Passes" in result
    assert "signed note" in result


def test_tool_returns_explicit_no_information_when_nothing_found():
    # 独立 collection 名：ephemeral Chroma 在同进程内共享底层状态（见 Phase 1 conftest）
    tool = _make_tool_with_chunks([], collection_name="tools-empty")
    result = tool.invoke({"query": "quantum chromodynamics"})
    assert result == "No relevant information found."
