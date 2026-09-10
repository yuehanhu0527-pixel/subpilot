"""Agent tool registry。

Phase 3：retrieve_documents（dense retrieval）。
Phase 4：bathroom pass ×3、课堂事件 ×2；build_default_tools 组装默认 6 件套。
"""

from datetime import datetime, timezone
from pathlib import Path

from ...services.tracker import BathroomPassRepo, ClassroomEventRepo
from .bathroom import (
    make_end_bathroom_pass_tool,
    make_get_bathroom_pass_status_tool,
    make_start_bathroom_pass_tool,
)
from .events import make_list_today_events_tool, make_log_classroom_event_tool
from .retrieval import make_retrieve_documents_tool

__all__ = [
    "build_default_tools",
    "make_retrieve_documents_tool",
    "make_start_bathroom_pass_tool",
    "make_end_bathroom_pass_tool",
    "make_get_bathroom_pass_status_tool",
    "make_log_classroom_event_tool",
    "make_list_today_events_tool",
]


def build_default_tools(
    db_path: str | Path,
    day_fn,
    schedule_engine=None,
    clock=None,
    tz_name: str | None = None,
    store=None,
    embedder=None,
):
    """组装默认工具集（6 个）：retrieval + bathroom ×3 + events ×2。

    - clock：默认系统时钟（UTC）；测试注入固定时间。
    - day_fn：aware datetime → 学校本地日期（school_day_fn）。
    - schedule_engine：供事件工具解析/校验 period。
    """
    clock = clock or (lambda: datetime.now(timezone.utc))
    passes = BathroomPassRepo(db_path)
    events = ClassroomEventRepo(db_path)
    return [
        make_retrieve_documents_tool(store=store, embedder=embedder),
        make_start_bathroom_pass_tool(passes, clock, day_fn, tz_name),
        make_end_bathroom_pass_tool(passes, clock, tz_name),
        make_get_bathroom_pass_status_tool(passes, clock, tz_name),
        make_log_classroom_event_tool(events, clock, day_fn, schedule_engine, tz_name),
        make_list_today_events_tool(events, clock, day_fn, tz_name),
    ]
