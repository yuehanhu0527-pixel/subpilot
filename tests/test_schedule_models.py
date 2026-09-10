"""models.py：Period / Schedule 的构造校验与排序。"""

from datetime import date, time

import pytest

from app.schedule.models import Period, Schedule


def _period(name="P1", kind="class", start="08:00", end="08:45", subject="Math"):
    return Period(
        name=name,
        kind=kind,
        start=time.fromisoformat(start),
        end=time.fromisoformat(end),
        subject=subject,
    )


def test_period_accepts_controlled_kinds():
    for kind in ("class", "lunch", "prep", "other"):
        p = _period(kind=kind)
        assert p.kind == kind


def test_period_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        _period(kind="homeroom")


def test_period_rejects_start_equal_to_end():
    with pytest.raises(ValueError, match="start"):
        _period(start="09:00", end="09:00")


def test_period_rejects_start_after_end():
    with pytest.raises(ValueError, match="start"):
        _period(start="10:00", end="09:00")


def test_period_subject_defaults_to_none():
    p = _period(subject=None)
    assert p.subject is None


def test_schedule_sorts_periods_by_start():
    p1 = _period(name="P1", start="08:00", end="08:45")
    p2 = _period(name="P2", start="08:55", end="09:40")
    schedule = Schedule(days={date(2026, 9, 10): (p2, p1)})  # 故意乱序
    assert schedule.days[date(2026, 9, 10)] == (p1, p2)


def test_schedule_rejects_overlapping_periods():
    p1 = _period(name="P1", start="08:00", end="09:00")
    p2 = _period(name="P2", start="08:30", end="09:40")
    with pytest.raises(ValueError, match="[Oo]verlap"):
        Schedule(days={date(2026, 9, 10): (p1, p2)})


def test_schedule_accepts_adjacent_periods():
    p1 = _period(name="Lunch", kind="lunch", start="11:30", end="12:15", subject=None)
    p2 = _period(name="Prep", kind="prep", start="12:15", end="12:55", subject=None)
    schedule = Schedule(days={date(2026, 9, 10): (p1, p2)})
    assert len(schedule.days[date(2026, 9, 10)]) == 2


def test_schedule_accepts_empty_day():
    schedule = Schedule(days={date(2026, 9, 10): ()})
    assert schedule.days[date(2026, 9, 10)] == ()
