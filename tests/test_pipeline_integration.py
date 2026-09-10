"""端到端集成测试：真实 embedding 模型 + 本地 Chroma。

标记为 integration（首次运行会下载 all-MiniLM-L6-v2 模型，较慢）。
运行方式：pytest -m integration -v
"""

import pytest

from app.rag.embedder import LocalEmbedder
from app.rag.pipeline import ingest_file, query
from app.rag.store import ChromaStore


@pytest.mark.integration
def test_ingest_and_query_end_to_end(tmp_path):
    txt = tmp_path / "rules.txt"
    txt.write_text(
        "Students must sign out before leaving the classroom.\n\n"
        "All visitors must report to the front office."
    )
    store = ChromaStore.persistent(tmp_path / "chroma")
    embedder = LocalEmbedder()

    n = ingest_file(str(txt), collection_name="integration", store=store, embedder=embedder)
    assert n >= 1

    result = query(
        "what should students do before leaving?",
        collection_name="integration", store=store, embedder=embedder,
    )
    assert result.found
    assert result.chunks[0].source_name == "rules.txt"
    assert "sign out" in result.chunks[0].text

    nothing = query(
        "quantum chromodynamics gauge theory",
        collection_name="integration", store=store, embedder=embedder,
    )
    assert not nothing.found
    assert nothing.message == "No relevant information found."
