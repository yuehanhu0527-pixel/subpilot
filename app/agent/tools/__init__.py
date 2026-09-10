"""Agent tool registry（Phase 3 起）。

Phase 3：仅 retrieve_documents（dense retrieval）。
Phase 4：在此包内逐个新增工具并 re-export。
"""

from .retrieval import make_retrieve_documents_tool

__all__ = ["make_retrieve_documents_tool"]
