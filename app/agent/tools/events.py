"""课堂事件工具：log_classroom_event / list_today_events。

- 事实性记录：无任何分类/标签字段；工具说明明确「只描述可观察事实」。
- period 省略时按 schedule engine + 时钟自动解析；显式提供时校验当天课表。
- 时间戳一律由注入的 clock 提供。
"""

from datetime import date, datetime
from typing import Callable

from langchain_core.tools import tool

from ...schedule.engine import ScheduleEngine
from ...services.tracker import ClassroomEventRepo, local_hhmm


def make_log_classroom_event_tool(
    repo: ClassroomEventRepo,
    clock: Callable[[], datetime],
    day_fn: Callable[[datetime], str],
    schedule_engine: ScheduleEngine | None = None,
    tz_name: str | None = None,
):
    @tool
    def log_classroom_event(
        description: str,
        period: str | None = None,
        students: list[str] | None = None,
    ) -> str:
        """Record a factual classroom event for the end-of-day substitute note. Describe only observable facts (what happened); never label or judge student behavior. Omit period to use the current one automatically."""
        description = (description or "").strip()
        if not description:
            return "Error: event description must not be empty."
        now = clock()
        if period:
            if schedule_engine is None:
                return "Error: cannot validate period without a schedule."
            day = date.fromisoformat(day_fn(now))
            valid = sorted(p.name for p in schedule_engine.periods_on(day))
            if period not in valid:
                return (
                    f"Error: unknown period {period!r} for today's schedule "
                    f"(valid: {', '.join(valid) or 'none'})."
                )
        elif schedule_engine is not None:
            status = schedule_engine.status_at(now)
            if status.status == "in_period":
                period = status.current_period.name

        repo.add_event(day_fn(now), now, description, period=period, students=students)
        stamp = f"[{period}] " if period else ""
        return f"Event logged: {stamp}{local_hhmm(now, tz_name)} — {description}"

    return log_classroom_event


def make_list_today_events_tool(
    repo: ClassroomEventRepo,
    clock: Callable[[], datetime],
    day_fn: Callable[[datetime], str],
    tz_name: str | None = None,
):
    @tool
    def list_today_events() -> str:
        """List all classroom events logged today, in chronological order. Use this to compose the end-of-day substitute note."""
        rows = repo.events_on(day_fn(clock()))
        if not rows:
            return "No events logged today."
        lines = []
        for row in rows:
            stamp = f"[{row['period']}] " if row["period"] else ""
            line = f"{stamp}{local_hhmm(row['recorded_at'], tz_name)} — {row['description']}"
            if row["students"]:
                line += f" (students: {', '.join(row['students'])})"
            lines.append(line)
        return "\n".join(lines)

    return list_today_events
