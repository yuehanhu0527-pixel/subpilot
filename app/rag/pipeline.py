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
from .retriever import Retriever
from .store import ChromaStore

_embedder: LocalEmbedder | None = None
_store: ChromaStore | None = None


def get_embedder() -> LocalEmbedder:
    """共享的本地 embedding 单例（模型只加载一次）。"""
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder(cache_dir=str(settings.data_dir / "models"))
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
    return Retriever(store, embedder).search(collection_name, text, top_k=top_k)
