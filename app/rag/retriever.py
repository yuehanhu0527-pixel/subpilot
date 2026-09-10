"""检索器：查询 embed → 向量检索 → 相似度阈值过滤 → 带来源的结果。"""

from .models import RetrievedChunk, SearchResult

# all-MiniLM-L6-v2 的余弦相似度经验阈值：低于此值视为不相关
DEFAULT_SIMILARITY_THRESHOLD = 0.30


class Retriever:
    def __init__(self, store, embedder, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self._store = store
        self._embedder = embedder
        self._threshold = threshold

    def search(self, collection_name: str, query: str, top_k: int = 5) -> SearchResult:
        """检索与查询最相关的片段。

        相似度低于阈值的命中会被丢弃（反幻觉：低分片段不喂给下游）。
        没有任何命中时返回 found=False 且 message='No relevant information found.'。
        """
        query_embedding = self._embedder.encode([query])[0]
        hits = self._store.search(collection_name, query_embedding, top_k=top_k)
        chunks = []
        for text, meta, similarity in hits:
            if similarity < self._threshold:
                continue
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source_name=meta["source_name"],
                    page=meta.get("page"),
                    heading=meta.get("heading") or None,
                    similarity=similarity,
                )
            )
        return SearchResult(chunks=tuple(chunks))
