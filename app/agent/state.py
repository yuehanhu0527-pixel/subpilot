"""LangGraph agent 的状态定义。"""

from datetime import datetime
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """图内共享状态。

    - messages：对话历史（add_messages 累加）
    - now：调用方注入的 aware datetime（contextualize 的输入，可缺省）
    - schedule_context：contextualize 写入的确定性上下文（供检查/调试）
    """

    messages: Annotated[list, add_messages]
    now: datetime | None
    schedule_context: str
