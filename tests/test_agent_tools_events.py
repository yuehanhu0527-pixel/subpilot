"""events tools：log_classroom_event / list_today_events。

重点：period 自动解析与校验（防编造）、事实性记录、按时间排序输出。
"""

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from app.agent.tools.events import (
    make_list_today_events_tool,
    make_log_classroom_event_tool,
)
from app.schedule.engine import ScheduleEngine
from app.schedule.models import Period, Schedule
from app.services.tracker import ClassroomEventRepo

TZ = "America/New_York"


def utc(hour, minute):
    return datetime(2026, 9, 10, hour, minute, tzinfo=timezone.utc)


@pytest.fixture
def schedule_engine():
    schedule = Schedule(
        days={
            date(2026, 9, 10): (
                Period(name="P1", kind="class", start=time(8, 0), end=time(8, 45), subject="Math"),
                Period(name="P2", kind="class", start=time(8, 55), end=time(9, 40), subject="English"),
                Period(name="P3", kind="class", start=time(13, 0), end=time(13, 45), subject="Science"),
            )
        }
    )
    return ScheduleEngine(schedule, tz=TZ)


@pytest.fixture
def env(tmp_path):
    repo = ClassroomEventRepo(tmp_path / "subpilot.db")
    box = {"now": utc(17, 30)}  # = 13:30 EDT，P3 中

    def clock():
        return box["now"]

    def day_fn(now):
        return "2026-09-10"

    return {"repo": repo, "box": box, "clock": clock, "day_fn": day_fn}


def _log_tool(env, schedule_engine=None):
    return make_log_classroom_event_tool(
        env["repo"], env["clock"], env["day_fn"], schedule_engine=schedule_engine, tz_name=TZ
    )


def _list_tool(env):
    return make_list_today_events_tool(env["repo"], env["clock"], env["day_fn"], tz_name=TZ)


def test_log_resolves_current_period_automatically(env, schedule_engine):
    tool = _log_tool(env, schedule_engine)
    result = tool.invoke({"description": "Students finished the quiz early."})
    assert result == "Event logged: [P3] 13:30 — Students finished the quiz early."
    rows = env["repo"].events_on("2026-09-10")
    assert rows[0]["period"] == "P3"


def test_log_explicit_valid_period(env, schedule_engine):
    tool = _log_tool(env, schedule_engine)
    result = tool.invoke({"description": "Warm-up went well.", "period": "P1"})
    assert result == "Event logged: [P1] 13:30 — Warm-up went well."


def test_log_invalid_period_rejected(env, schedule_engine):
    tool = _log_tool(env, schedule_engine)
    result = tool.invoke({"description": "Something happened.", "period": "P9"})
    assert result.startswith("Error: unknown period")
    assert env["repo"].events_on("2026-09-10") == []


def test_log_period_without_schedule_rejected(env):
    tool = _log_tool(env, None)
    result = tool.invoke({"description": "Something happened.", "period": "P1"})
    assert result.startswith("Error: cannot validate period")


def test_log_empty_description_rejected(env, schedule_engine):
    tool = _log_tool(env, schedule_engine)
    result = tool.invoke({"description": "   "})
    assert result.startswith("Error:")


def test_log_with_students(env, schedule_engine):
    tool = _log_tool(env, schedule_engine)
    tool.invoke(
        {"description": "Group work on worksheets.", "students": ["Alice", "Bob"]}
    )
    rows = env["repo"].events_on("2026-09-10")
    assert rows[0]["students"] == ["Alice", "Bob"]


def test_list_events_chronological_with_students(env, schedule_engine):
    tool = _log_tool(env, schedule_engine)
    tool.invoke({"description": "First event."})
    tool.invoke({"description": "Second event.", "students": ["Alice"]})
    result = _list_tool(env).invoke({})
    lines = result.split("\n")
    assert len(lines) == 2
    assert lines[0].endswith("First event.")
    assert "Second event." in lines[1]
    assert "(students: Alice)" in lines[1]


def test_list_events_empty_day(env):
    assert _list_tool(env).invoke({}) == "No events logged today."
