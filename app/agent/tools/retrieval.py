"""retrieve_documents：Phase 3 唯一的工具，包装 Phase 1 dense retrieval。

store / embedder 可注入（测试）；缺省时在调用期解析 rag 的单例。
无命中时返回 Phase 1 的明确提示，绝不编造。
"""

from langchain_core.tools import tool

from ...rag.pipeline import get_embedder, get_store
from ...rag.pipeline import query as rag_query


def make_retrieve_documents_tool(
    store=None,
    embedder=None,
    collection_name: str = "default",
    top_k: int = 3,
):
    """构造检索工具。"""

    @tool
    def retrieve_documents(query: str) -> str:
        """Search the school's documents (handbooks, policies, lesson plans) for the query. Use this whenever the user asks about school rules, procedures, or anything likely written in a document."""
        result = rag_query(
            query,
            collection_name,
            top_k=top_k,
            store=store or get_store(),
            embedder=embedder or get_embedder(),
        )
        if not result.found:
            return result.message  # "No relevant information found."
        return "\n\n".join(
            f"Source: {chunk.citation()}\n{chunk.text}" for chunk in result.chunks
        )

    return retrieve_documents
