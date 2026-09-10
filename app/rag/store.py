"""Chroma 向量存储封装（余弦距离空间）。"""

from pathlib import Path

import chromadb

from .models import TextChunk


class ChromaStore:
    """对 chromadb 的薄封装：chunk 入库与向量检索。

    collection 使用 cosine 空间；search 返回的 similarity = 1 - 距离，
    取值范围约 [0, 1]，越大越相关。
    """

    def __init__(self, client):
        self._client = client

    @classmethod
    def persistent(cls, path: str | Path) -> "ChromaStore":
        """本地持久化存储（生产用）。"""
        return cls(chromadb.PersistentClient(path=str(path)))

    @classmethod
    def ephemeral(cls) -> "ChromaStore":
        """内存存储（测试用）。"""
        return cls(chromadb.EphemeralClient())

    def _collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """把 chunk 文本 + 向量 + 来源元数据写入 collection。"""
        if not chunks:
            return
        metadatas = []
        for c in chunks:
            meta = {"source_name": c.source_name, "heading": c.heading or ""}
            if c.page is not None:
                meta["page"] = c.page
            metadatas.append(meta)
        self._collection(collection_name).add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, dict, float]]:
        """返回 [(text, metadata, similarity)]，按相似度降序；空库返回 []。"""
        col = self._collection(collection_name)
        if col.count() == 0:
            return []
        result = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append((doc, meta, round(1.0 - dist, 6)))
        return hits
