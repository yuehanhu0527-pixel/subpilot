"""context.py：确定性 schedule context 文本（零 LLM，直接单测）。"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from app.agent.context import build_schedule_context
from app.schedule.models import Period, ScheduleStatus

NY = ZoneInfo("America/New_York")


def _period(name, kind="class", start="08:00", end="08:45", subject=None):
    return Period(
        name=name,
        kind=kind,
        start=time.fromisoformat(start),
        end=time.fromisoformat(end),
        subject=subject,
    )


P1 = _period("P1", start="08:00", end="08:45", subject="Math")
P2 = _period("P2", start="08:55", end="09:40", subject="English")
LUNCH = _period("Lunch", kind="lunch", start="11:30", end="12:15")
P3 = _period("P3", start="13:00", end="13:45", subject="Science")


def _now(hour, minute):
    return datetime(2026, 9, 10, hour, minute, tzinfo=NY)


def test_in_period_context_has_time_period_subject_and_remaining():
    status = ScheduleStatus(
        status="in_period", now=_now(8, 30),
        current_period=P1, minutes_remaining=15,
        next_period=P2, minutes_until_next=25,
        previous_period=None,
    )
    text = build_schedule_context(status)
    assert "08:30" in text
    assert "America/New_York" in text
    assert "P1" in text and "Math" in text
    assert "15 minutes remaining" in text
    assert "Next period: P2" in text


def test_in_period_last_period_has_no_next():
    status = ScheduleStatus(
        status="in_period", now=_now(13, 30),
        current_period=P3, minutes_remaining=15,
        next_period=None, minutes_until_next=None,
        previous_period=LUNCH,
    )
    text = build_schedule_context(status)
    assert "Next period: none." in text


def test_between_periods_context():
    status = ScheduleStatus(
        status="between_periods", now=_now(8, 50),
        current_period=None,
        previous_period=P1,
        next_period=P2, minutes_until_next=5,
    )
    text = build_schedule_context(status)
    assert "Between periods" in text
    assert "P1" in text and "ended at 08:45" in text
    assert "P2" in text and "starts at 08:55" in text
    assert "in 5 minutes" in text


def test_before_school_context():
    status = ScheduleStatus(
        status="before_school", now=_now(7, 59),
        next_period=P1, minutes_until_next=1,
    )
    text = build_schedule_context(status)
    assert "Before school" in text
    assert "First period: P1" in text
    assert "in 1 minute" in text


def test_after_school_context():
    status = ScheduleStatus(
        status="after_school", now=_now(14, 0),
        previous_period=P3,
    )
    text = build_schedule_context(status)
    assert "After school" in text
    assert "P3" in text and "ended at 13:45" in text


def test_no_classes_context_is_explicit():
    status = ScheduleStatus(status="no_classes", now=_now(13, 0))
    text = build_schedule_context(status)
    assert "No classes are scheduled today." in text


def test_lunch_context_uses_name_without_subject():
    status = ScheduleStatus(
        status="in_period", now=_now(12, 0),
        current_period=LUNCH, minutes_remaining=15,
        next_period=P3, minutes_until_next=60,
        previous_period=P2,
    )
    text = build_schedule_context(status)
    assert "Lunch" in text
    assert "Math" not in text and "English" not in text


def test_build_schedule_context_is_deterministic():
    status = ScheduleStatus(status="no_classes", now=_now(13, 0))
    assert build_schedule_context(status) == build_schedule_context(status)
