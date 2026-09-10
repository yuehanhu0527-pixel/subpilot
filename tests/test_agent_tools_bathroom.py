"""bathroom tools：start / end / status 的输入输出与错误文案。

时钟、repo、day_fn 全部注入；tz 用 America/New_York 验证本地时间显示。
"""

from datetime import datetime, timezone

import pytest

from app.agent.tools.bathroom import (
    make_end_bathroom_pass_tool,
    make_get_bathroom_pass_status_tool,
    make_start_bathroom_pass_tool,
)
from app.services.tracker import BathroomPassRepo

TZ = "America/New_York"


def utc(hour, minute):
    return datetime(2026, 9, 10, hour, minute, tzinfo=timezone.utc)


@pytest.fixture
def env(tmp_path):
    repo = BathroomPassRepo(tmp_path / "subpilot.db")
    box = {"now": utc(13, 15)}  # = 09:15 EDT

    def clock():
        return box["now"]

    def day_fn(now):
        return "2026-09-10"

    tools = {
        "start": make_start_bathroom_pass_tool(repo, clock, day_fn, tz_name=TZ),
        "end": make_end_bathroom_pass_tool(repo, clock, tz_name=TZ),
        "status": make_get_bathroom_pass_status_tool(repo, clock, tz_name=TZ),
    }
    return {"tools": tools, "box": box, "repo": repo}


def test_start_confirms_with_local_time(env):
    result = env["tools"]["start"].invoke({"student": "Alice"})
    assert result == "Recorded: Alice left at 09:15."


def test_start_duplicate_returns_error(env):
    env["tools"]["start"].invoke({"student": "Alice"})
    result = env["tools"]["start"].invoke({"student": "alice"})  # casefold 变体
    assert "alice" in result  # 原样显示输入
    assert "already has an open bathroom pass" in result


def test_start_empty_name_returns_error(env):
    result = env["tools"]["start"].invoke({"student": "  "})
    assert result.startswith("Error:")


def test_end_reports_duration(env):
    env["tools"]["start"].invoke({"student": "Alice"})
    env["box"]["now"] = utc(13, 27)
    result = env["tools"]["end"].invoke({"student": "Alice"})
    assert result == "Alice returned at 09:27 (out 12 minutes)."


def test_end_without_open_pass_returns_error(env):
    result = env["tools"]["end"].invoke({"student": "Bob"})
    assert result == "Bob has no open bathroom pass."


def test_status_lists_open_passes(env):
    env["tools"]["start"].invoke({"student": "Alice"})
    env["box"]["now"] = utc(13, 24)  # Alice 出门 9 分钟
    env["tools"]["start"].invoke({"student": "Bob"})
    result = env["tools"]["status"].invoke({})
    assert "Alice (9 min)" in result
    assert "Bob (0 min)" in result


def test_status_nobody_out(env):
    assert env["tools"]["status"].invoke({}) == "No students are currently out."


def test_status_specific_student_out(env):
    env["tools"]["start"].invoke({"student": "Alice"})
    env["box"]["now"] = utc(13, 24)
    result = env["tools"]["status"].invoke({"student": "Alice"})
    assert "Alice has been out 9 minutes" in result
    assert "left at 09:15" in result


def test_status_specific_student_back_with_history(env):
    env["tools"]["start"].invoke({"student": "Alice"})
    env["box"]["now"] = utc(13, 27)
    env["tools"]["end"].invoke({"student": "Alice"})
    result = env["tools"]["status"].invoke({"student": "Alice"})
    assert "is not currently out" in result
    assert "last pass" in result


def test_status_unknown_student(env):
    result = env["tools"]["status"].invoke({"student": "Zoe"})
    assert "no bathroom pass records" in result
