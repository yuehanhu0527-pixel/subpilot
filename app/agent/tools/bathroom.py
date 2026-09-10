"""Bathroom pass 工具：start / end / status。

- 时间戳一律由注入的 clock 提供——LLM 不提供时间。
- 重复开启未结束的 pass 会被仓储层拒绝（业务规则 + 部分唯一索引）。
- 显示用学校本地时间（tz_name）。
"""

from datetime import datetime
from typing import Callable

from langchain_core.tools import tool

from ...services.tracker import (
    BathroomPassRepo,
    DuplicateOpenPassError,
    NoOpenPassError,
    local_hhmm,
)


def make_start_bathroom_pass_tool(
    repo: BathroomPassRepo,
    clock: Callable[[], datetime],
    day_fn: Callable[[datetime], str],
    tz_name: str | None = None,
):
    @tool
    def start_bathroom_pass(student: str) -> str:
        """Record that a student has just left the classroom for a bathroom pass. Call this when a student asks to go to the bathroom and leaves."""
        student = student.strip()
        if not student:
            return "Error: student name must not be empty."
        now = clock()
        try:
            repo.start_pass(student, now, day_fn(now))
        except DuplicateOpenPassError as exc:
            return (
                f"{exc.student} already has an open bathroom pass "
                f"(left at {local_hhmm(exc.departed_at, tz_name)})."
            )
        return f"Recorded: {student} left at {local_hhmm(now, tz_name)}."

    return start_bathroom_pass


def make_end_bathroom_pass_tool(
    repo: BathroomPassRepo,
    clock: Callable[[], datetime],
    tz_name: str | None = None,
):
    @tool
    def end_bathroom_pass(student: str) -> str:
        """Record that a student has returned from a bathroom pass. Call this when the student comes back to the classroom."""
        student = student.strip()
        if not student:
            return "Error: student name must not be empty."
        now = clock()
        try:
            name, _, minutes_out = repo.end_pass(student, now)
        except NoOpenPassError as exc:
            return f"{exc.student} has no open bathroom pass."
        return f"{name} returned at {local_hhmm(now, tz_name)} (out {minutes_out} minutes)."

    return end_bathroom_pass


def make_get_bathroom_pass_status_tool(
    repo: BathroomPassRepo,
    clock: Callable[[], datetime],
    tz_name: str | None = None,
):
    @tool
    def get_bathroom_pass_status(student: str | None = None) -> str:
        """Check who is currently out on a bathroom pass and for how long, or check one specific student's status. Call this before starting a new pass for a student or when asked who is out."""
        now = clock()
        if student:
            name = student.strip()
            status = repo.student_status(name, now)
            if status is not None:
                return (
                    f"{status['student']} has been out {status['minutes_out']} minutes "
                    f"(left at {local_hhmm(status['departed_at'], tz_name)})."
                )
            last = repo.last_pass(name)
            if last is None:
                return f"{name} has no bathroom pass records."
            departed, returned = last
            if returned is None:
                return f"{name} is not currently out."
            return (
                f"{name} is not currently out (last pass: "
                f"left at {local_hhmm(departed, tz_name)}, "
                f"returned at {local_hhmm(returned, tz_name)})."
            )
        open_passes = repo.open_passes(now)
        if not open_passes:
            return "No students are currently out."
        listing = ", ".join(
            f"{p['student']} ({p['minutes_out']} min)" for p in open_passes
        )
        return f"Currently out: {listing}."

    return get_bathroom_pass_status
