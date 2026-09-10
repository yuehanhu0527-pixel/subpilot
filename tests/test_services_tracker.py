"""tracker.py：bathroom pass 与课堂事件的 sqlite 仓储。

时间全部显式注入（aware datetime）；day 由调用方（tool 层）计算后传入。
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.tracker import (
    BathroomPassRepo,
    ClassroomEventRepo,
    DuplicateOpenPassError,
    NoOpenPassError,
)

UTC = timezone.utc
DAY = "2026-09-10"


def dt(hour, minute, second=0):
    return datetime(2026, 9, 10, hour, minute, second, tzinfo=UTC)


@pytest.fixture
def passes(tmp_path):
    return BathroomPassRepo(tmp_path / "subpilot.db")


@pytest.fixture
def events(tmp_path):
    return ClassroomEventRepo(tmp_path / "subpilot.db")


# --- BathroomPassRepo ---


def test_start_and_open_passes(passes):
    passes.start_pass("Alice", dt(9, 15), DAY)
    open_now = passes.open_passes(dt(9, 24))
    assert len(open_now) == 1
    assert open_now[0]["student"] == "Alice"
    assert open_now[0]["departed_at"] == dt(9, 15)
    assert open_now[0]["minutes_out"] == 9


def test_minutes_out_floors_to_whole_minutes(passes):
    passes.start_pass("Alice", dt(9, 15), DAY)
    # 09:24:59 → 9 分钟 59 秒 → 向下取整 9（不夸大离开时长）
    status = passes.student_status("Alice", dt(9, 24, 59))
    assert status["minutes_out"] == 9


def test_duplicate_open_pass_rejected(passes):
    passes.start_pass("Alice", dt(9, 15), DAY)
    with pytest.raises(DuplicateOpenPassError):
        passes.start_pass("Alice", dt(9, 20), DAY)


def test_duplicate_rejected_across_case_variants(passes):
    passes.start_pass("Alice", dt(9, 15), DAY)
    with pytest.raises(DuplicateOpenPassError):
        passes.start_pass("ALICE", dt(9, 20), DAY)


def test_end_pass_returns_duration(passes):
    passes.start_pass("Alice", dt(9, 15), DAY)
    name, departed, minutes_out = passes.end_pass("Alice", dt(9, 27))
    assert name == "Alice"
    assert departed == dt(9, 15)
    assert minutes_out == 12
    assert passes.open_passes(dt(9, 30)) == []


def test_end_pass_without_open_raises(passes):
    with pytest.raises(NoOpenPassError):
        passes.end_pass("Alice", dt(9, 27))


def test_student_status_not_out_returns_none(passes):
    assert passes.student_status("Alice", dt(9, 20)) is None
    passes.start_pass("Alice", dt(9, 15), DAY)
    passes.end_pass("Alice", dt(9, 27))
    assert passes.student_status("Alice", dt(9, 30)) is None


def test_last_pass_history(passes):
    assert passes.last_pass("Alice") is None
    passes.start_pass("Alice", dt(9, 15), DAY)
    passes.end_pass("Alice", dt(9, 27))
    departed, returned = passes.last_pass("Alice")
    assert departed == dt(9, 15)
    assert returned == dt(9, 27)


# --- ClassroomEventRepo ---


def test_add_and_list_events_chronological(events):
    events.add_event(DAY, dt(13, 30), "Students finished the quiz early.", period="P3")
    events.add_event(
        DAY, dt(8, 20), "Fire drill practice announcement.", students=["Alice", "Bob"]
    )
    rows = events.events_on(DAY)
    assert [r["description"] for r in rows] == [
        "Fire drill practice announcement.",
        "Students finished the quiz early.",
    ]
    assert rows[0]["period"] is None
    assert rows[0]["students"] == ["Alice", "Bob"]
    assert rows[1]["period"] == "P3"


def test_events_filtered_by_day(events):
    events.add_event(DAY, dt(9, 0), "event on day 1")
    events.add_event("2026-09-11", dt(9, 0), "event on day 2")
    assert [r["description"] for r in events.events_on(DAY)] == ["event on day 1"]
    assert [r["description"] for r in events.events_on("2026-09-11")] == ["event on day 2"]


def test_events_empty_day_returns_empty(events):
    assert events.events_on(DAY) == []


def test_empty_description_rejected(events):
    with pytest.raises(ValueError, match="description"):
        events.add_event(DAY, dt(9, 0), "   ")


def test_repos_share_one_database_file(tmp_path):
    path = tmp_path / "subpilot.db"
    BathroomPassRepo(path)
    ClassroomEventRepo(path)  # 第二次初始化 schema 应幂等，不报错
    passes2 = BathroomPassRepo(path)
    passes2.start_pass("Alice", dt(9, 15), DAY)
    assert len(passes2.open_passes(dt(9, 20))) == 1
