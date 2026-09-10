"""engine.py：确定性状态判断。

全部用例注入显式时刻与时区，不依赖服务器本地时区与系统时钟。
基准日程时区 America/New_York；2026-09-10 为 EDT (UTC-4)。
"""

from datetime import date, datetime, time, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import app.schedule.engine as engine_mod
from app.schedule.engine import ScheduleEngine
from app.schedule.models import Period, Schedule

NY = "America/New_York"
LA = "America/Los_Angeles"
DAY = date(2026, 9, 10)
UTC = timezone.utc


def _period(name, kind, start, end, subject=None):
    return Period(
        name=name,
        kind=kind,
        start=time.fromisoformat(start),
        end=time.fromisoformat(end),
        subject=subject,
    )


DEFAULT_PERIODS = (
    _period("P1", "class", "08:00", "08:45", subject="Math"),
    _period("P2", "class", "08:55", "09:40", subject="English"),
    _period("Lunch", "lunch", "11:30", "12:15"),
    _period("Prep", "prep", "12:15", "12:55"),
    _period("P3", "class", "13:00", "13:45", subject="Science"),
)


def make_schedule(day=DAY, periods=DEFAULT_PERIODS):
    return Schedule(days={day: periods})


def local_dt(day, hour, minute, second=0):
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=ZoneInfo(NY))


def utc_dt(day, hour, minute):
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


@pytest.fixture
def engine():
    return ScheduleEngine(make_schedule(), tz=NY)


# --- 五种状态 ---


def test_before_first_period_reports_next_and_minutes_until(engine):
    s = engine.status_at(local_dt(DAY, 7, 59))
    assert s.status == "before_school"
    assert s.current_period is None
    assert s.previous_period is None
    assert s.next_period.name == "P1"
    assert s.minutes_until_next == 1


def test_at_exact_start_is_in_period_with_full_minutes_remaining(engine):
    s = engine.status_at(local_dt(DAY, 8, 0))
    assert s.status == "in_period"
    assert s.current_period.name == "P1"
    assert s.minutes_remaining == 45
    assert s.next_period.name == "P2"
    assert s.minutes_until_next == 55


def test_mid_period_reports_current_subject_remaining_and_next(engine):
    s = engine.status_at(local_dt(DAY, 8, 30))
    assert s.status == "in_period"
    assert s.current_period.name == "P1"
    assert s.current_period.subject == "Math"
    assert s.minutes_remaining == 15
    assert s.next_period.name == "P2"
    assert s.minutes_until_next == 25


def test_at_exact_end_is_between_periods(engine):
    s = engine.status_at(local_dt(DAY, 8, 45))
    assert s.status == "between_periods"
    assert s.current_period is None
    assert s.previous_period.name == "P1"
    assert s.next_period.name == "P2"
    assert s.minutes_until_next == 10


def test_between_periods_reports_previous_and_next(engine):
    s = engine.status_at(local_dt(DAY, 8, 50))
    assert s.status == "between_periods"
    assert s.previous_period.name == "P1"
    assert s.next_period.name == "P2"
    assert s.minutes_until_next == 5


def test_lunch_is_in_period_with_kind_and_no_subject(engine):
    s = engine.status_at(local_dt(DAY, 12, 0))
    assert s.status == "in_period"
    assert s.current_period.kind == "lunch"
    assert s.current_period.subject is None
    assert s.minutes_remaining == 15
    assert s.next_period.name == "Prep"
    assert s.minutes_until_next == 15


def test_prep_is_in_period_with_kind_and_no_subject(engine):
    s = engine.status_at(local_dt(DAY, 12, 30))
    assert s.status == "in_period"
    assert s.current_period.kind == "prep"
    assert s.current_period.subject is None
    assert s.next_period.name == "P3"
    assert s.minutes_until_next == 30


def test_last_period_has_no_next(engine):
    s = engine.status_at(local_dt(DAY, 13, 30))
    assert s.status == "in_period"
    assert s.current_period.name == "P3"
    assert s.minutes_remaining == 15
    assert s.next_period is None
    assert s.minutes_until_next is None


def test_after_last_period_is_after_school(engine):
    s = engine.status_at(local_dt(DAY, 14, 0))
    assert s.status == "after_school"
    assert s.current_period is None
    assert s.next_period is None
    assert s.previous_period.name == "P3"


def test_missing_day_is_no_classes(engine):
    s = engine.status_at(local_dt(date(2026, 9, 11), 8, 30))
    assert s.status == "no_classes"
    assert s.current_period is None
    assert s.next_period is None
    assert s.previous_period is None


def test_empty_day_is_no_classes():
    engine = ScheduleEngine(make_schedule(periods=()), tz=NY)
    s = engine.status_at(local_dt(DAY, 8, 30))
    assert s.status == "no_classes"


# --- 时间与时区 ---


def test_naive_now_raises(engine):
    naive = datetime(2026, 9, 10, 8, 30)
    with pytest.raises(ValueError, match="aware"):
        engine.status_at(naive)


def test_utc_now_is_converted_to_schedule_tz(engine):
    # 12:00 UTC = 08:00 EDT → 正好第一节
    s = engine.status_at(utc_dt(DAY, 12, 0))
    assert s.status == "in_period"
    assert s.current_period.name == "P1"


def test_same_instant_in_different_tz_gives_same_result(engine):
    a = engine.status_at(utc_dt(DAY, 12, 0))
    b = engine.status_at(local_dt(DAY, 8, 0))
    assert a == b


def test_constructor_tz_param_determines_interpretation():
    schedule = make_schedule()
    # 同一时刻 17:00 UTC：NY 引擎解释为 13:00（P3 上课中），
    # LA 引擎解释为 10:00（P2 与 Lunch 之间）——课表墙钟时间按引擎时区解释
    instant = utc_dt(DAY, 17, 0)
    ny = ScheduleEngine(schedule, tz=NY).status_at(instant)
    assert ny.status == "in_period"
    assert ny.current_period.name == "P3"
    la = ScheduleEngine(schedule, tz=LA).status_at(instant)
    assert la.status == "between_periods"
    assert la.previous_period.name == "P2"
    assert la.next_period.name == "Lunch"


def test_settings_timezone_used_when_tz_omitted(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "settings", SimpleNamespace(timezone=NY)
    )
    engine = ScheduleEngine(make_schedule())
    s = engine.status_at(local_dt(DAY, 8, 30))
    assert s.status == "in_period"
    assert s.current_period.name == "P1"


def test_missing_timezone_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "settings", SimpleNamespace(timezone=None)
    )
    with pytest.raises(ValueError, match="SUBPILOT_TIMEZONE"):
        ScheduleEngine(make_schedule())


def test_unknown_timezone_raises():
    with pytest.raises(ValueError, match="Not/AZone"):
        ScheduleEngine(make_schedule(), tz="Not/AZone")


def test_dst_spring_forward_day():
    # 2026-03-08 凌晨 2 点跳至 3 点（EDT, UTC-4）；3-09 是切换后的周一
    day = date(2026, 3, 9)
    engine = ScheduleEngine(make_schedule(day=day), tz=NY)
    s = engine.status_at(utc_dt(day, 12, 0))  # = 08:00 EDT
    assert s.status == "in_period"
    assert s.current_period.name == "P1"


def test_dst_fall_back_day():
    # 2026-11-01 凌晨 2 点回拨至 1 点（EST, UTC-5）；11-02 是切换后的周一
    day = date(2026, 11, 2)
    engine = ScheduleEngine(make_schedule(day=day), tz=NY)
    s = engine.status_at(utc_dt(day, 13, 0))  # = 08:00 EST
    assert s.status == "in_period"
    assert s.current_period.name == "P1"


# --- 确定性 ---


def test_status_at_is_deterministic(engine):
    now = local_dt(DAY, 8, 30)
    assert engine.status_at(now) == engine.status_at(now)


def test_minutes_remaining_rounds_up(engine):
    # 08:44:30 → 还剩 30 秒，按整分钟向上取整为 1
    s = engine.status_at(local_dt(DAY, 8, 44, second=30))
    assert s.status == "in_period"
    assert s.minutes_remaining == 1


def test_minutes_until_next_rounds_up(engine):
    # 07:59:30 → 距 08:00 还有 30 秒，向上取整为 1
    s = engine.status_at(local_dt(DAY, 7, 59, second=30))
    assert s.status == "before_school"
    assert s.minutes_until_next == 1


# --- Phase 4：全天课表访问器（contextualize 注入用） ---


def test_periods_on_returns_todays_periods():
    engine = ScheduleEngine(make_schedule(), tz=NY)
    assert [p.name for p in engine.periods_on(DAY)] == ["P1", "P2", "Lunch", "Prep", "P3"]


def test_periods_on_missing_day_returns_empty():
    engine = ScheduleEngine(make_schedule(), tz=NY)
    assert engine.periods_on(date(2026, 9, 11)) == ()
