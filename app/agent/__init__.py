"""LangGraph agent（Phase 3）：最小 agent loop + Agent 门面。

Agent.run(session_id, text, now)：
- 载入 session 历史 → 图执行（contextualize → agent ⇄ tools）→ 保存本轮问答。
- now 由调用方注入（agent 不自己取时钟）；缺省则不注入 schedule context。
"""

from datetime import datetime

from langchain_core.messages import HumanMessage

from ..config import settings
from ..llm import get_provider
from ..schedule.engine import ScheduleEngine
from ..schedule.loader import load_schedule
from .graph import build_agent_graph
from .sessions import InMemorySessionStore, SessionStore
from .tools import make_retrieve_documents_tool


def _load_schedule_engine() -> ScheduleEngine | None:
    """默认 schedule 来源。刻意区分两种状态：

    - data/schedule.json 不存在 → None：不注入 schedule context；
    - 文件存在（哪怕当天无条目）→ 引擎就位，无课时给出
      "No classes are scheduled today." 的明确 context。
    """
    path = settings.data_dir / "schedule.json"
    if not path.exists():
        return None
    # 显式传入 tz：与 engine 内部回退等价，但组合关系明确、可测试
    return ScheduleEngine(load_schedule(path), tz=settings.timezone)


class Agent:
    """SubPilot agent 门面。model / tools / engine / sessions 均可注入（测试）。"""

    def __init__(
        self,
        model=None,
        tools=None,
        schedule_engine=None,
        sessions: SessionStore | None = None,
        store=None,
        embedder=None,
    ):
        self._model = model if model is not None else get_provider()
        self._sessions = sessions if sessions is not None else InMemorySessionStore()
        self._engine = _load_schedule_engine() if schedule_engine is None else schedule_engine
        if tools is not None:
            self._tools = list(tools)
        else:
            self._tools = [make_retrieve_documents_tool(store=store, embedder=embedder)]
        self._graph = build_agent_graph(self._model, self._tools, self._engine)

    def run(self, session_id: str, text: str, now: datetime | None = None) -> str:
        """执行一轮对话，返回最终回答文本。"""
        history = self._sessions.get(session_id)
        user_message = HumanMessage(content=text)
        initial: dict = {"messages": [*history, user_message]}
        if now is not None:
            initial["now"] = now

        result = self._graph.invoke(initial)
        answer = result["messages"][-1]
        self._sessions.append(session_id, [user_message, answer])
        return answer.content
