"""最基本 session state：按 session_id 保存对话历史。

定义清晰接口（SessionStore Protocol），当前提供内存实现；
后续可无缝替换为 SQLite checkpoint 等持久化实现——Agent 只依赖协议。
"""

from typing import Protocol, Sequence

from langchain_core.messages import BaseMessage


class SessionStore(Protocol):
    """Agent 依赖的 session 存储接口。"""

    def get(self, session_id: str) -> list[BaseMessage]:
        """返回该 session 的完整历史（副本）；未知 session 返回空列表。"""
        ...

    def append(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        """向该 session 追加一轮消息。"""
        ...

    def clear(self, session_id: str) -> None:
        """删除该 session；对未知 session 幂等。"""
        ...


class InMemorySessionStore:
    """进程内内存实现——Phase 3 的默认（最小）存储。"""

    def __init__(self):
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
