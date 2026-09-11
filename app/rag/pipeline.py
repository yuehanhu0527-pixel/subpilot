"""RAG 入口：文件入库（ingest）与查询（query）。

供测试、CLI 以及后续阶段的 Agent 工具层调用。
embedder / store 可注入（测试用），默认使用共享单例。
"""

from pathlib import Path

from ..config import settings
from .chunker import chunk_sections
from .embedder import LocalEmbedder
from .loader import load_document
from .models import SearchResult
from .retriever import DEFAULT_SIMILARITY_THRESHOLD, Retriever
from .store import ChromaStore

_embedder = None  # LocalEmbedder（默认）或 LexicalEmbedder（SUBPILOT_EMBEDDER=lexical）
_store: ChromaStore | None = None


def get_embedder():
    """共享的 embedding 单例。

    SUBPILOT_EMBEDDER=lexical 时返回轻量词法 embedder（无 torch，供 Render
    Free 512MB hosted demo 使用）；默认仍是本地 sentence-transformers 模型
    （只加载一次），本地 / Live 模式绝不降级。
    """
    global _embedder
    if _embedder is None:
        if settings.rag_embedder == "lexical":
            from .lexical import LexicalEmbedder

            _embedder = LexicalEmbedder()
        elif settings.rag_embedder == "local":
            _embedder = LocalEmbedder(cache_dir=str(settings.data_dir / "models"))
        else:
            raise ValueError(
                f"Unknown SUBPILOT_EMBEDDER value: {settings.rag_embedder!r} "
                "(expected 'local' or 'lexical')"
            )
    return _embedder


def get_store() -> ChromaStore:
    """共享的本地 Chroma 单例。"""
    global _store
    if _store is None:
        _store = ChromaStore.persistent(settings.data_dir / "chroma")
    return _store


def ingest_file(
    path: str | Path,
    collection_name: str = "default",
    store: ChromaStore | None = None,
    embedder: LocalEmbedder | None = None,
) -> int:
    """解析 → 分块 → embedding → 入库；返回入库的 chunk 数量。"""
    store = store or get_store()
    embedder = embedder or get_embedder()
    sections = load_document(path)
    chunks = chunk_sections(sections)
    if chunks:
        embeddings = embedder.encode([c.text for c in chunks])
        store.add_chunks(collection_name, chunks, embeddings)
    return len(chunks)


def query(
    text: str,
    collection_name: str = "default",
    top_k: int = 5,
    store: ChromaStore | None = None,
    embedder: LocalEmbedder | None = None,
) -> SearchResult:
    """检索查询；空结果时 message 为 'No relevant information found.'。"""
    store = store or get_store()
    embedder = embedder or get_embedder()
    # 阈值跟随 embedder 空间：dense 默认 0.30，词法（lexical）自带更低的经验值
    return Retriever(
        store,
        embedder,
        threshold=getattr(embedder, "threshold", DEFAULT_SIMILARITY_THRESHOLD),
    ).search(collection_name, text, top_k=top_k)
