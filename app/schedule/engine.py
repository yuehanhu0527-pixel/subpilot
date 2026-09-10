"""确定性日程引擎：给定课表、时区与显式时刻 → 当前状态。

- 不调用 datetime.now()：当前时间由上层（后续的 service / tool）取得后传入。
- 不依赖服务器本地时区：时区优先取构造参数 tz，其次 SUBPILOT_TIMEZONE；
  两者都没有时明确报错。
- 纯函数：同样的 (schedule, tz, now) 输入必得同样输出。
"""

from datetime import date, datetime, time, timedelta
from math import ceil
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import settings
from .models import Schedule, ScheduleStatus


def _resolve_tz(tz: str | None) -> ZoneInfo:
    name = tz if tz is not None else settings.timezone
    if name is None:
        raise ValueError(
            "No timezone configured: pass tz= to ScheduleEngine "
            "or set SUBPILOT_TIMEZONE"
        )
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Unknown timezone: {name}")


def _ceil_minutes(delta: timedelta) -> int:
    """向上取整到整分钟：还剩 30 秒也算 1 分钟。"""
    return ceil(delta.total_seconds() / 60)


class ScheduleEngine:
    def __init__(self, schedule: Schedule, tz: str | None = None):
        self._schedule = schedule
        self._tzinfo = _resolve_tz(tz)

    def status_at(self, now: datetime) -> ScheduleStatus:
        """判断 now 时刻的日程状态。now 必须带时区（任何时区均可）。"""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        local = now.astimezone(self._tzinfo)
        periods = self._schedule.days.get(local.date(), ())

        if not periods:
            return ScheduleStatus(status="no_classes", now=local)

        t = local.time()
        if t < periods[0].start:
            first = periods[0]
            return ScheduleStatus(
                status="before_school",
                now=local,
                next_period=first,
                minutes_until_next=_ceil_minutes(self._at(local.date(), first.start) - local),
            )

        for i, period in enumerate(periods):
            if t < period.start:
                # 课间：已过 periods[i-1].end（相邻时等于），未到 periods[i].start
                return ScheduleStatus(
                    status="between_periods",
                    now=local,
                    previous_period=periods[i - 1],
                    next_period=period,
                    minutes_until_next=_ceil_minutes(self._at(local.date(), period.start) - local),
                )
            if t < period.end:
                # start 闭区间、end 开区间：恰好 end 已落入课间分支
                nxt = periods[i + 1] if i + 1 < len(periods) else None
                return ScheduleStatus(
                    status="in_period",
                    now=local,
                    current_period=period,
                    minutes_remaining=_ceil_minutes(self._at(local.date(), period.end) - local),
                    next_period=nxt,
                    minutes_until_next=(
                        _ceil_minutes(self._at(local.date(), nxt.start) - local) if nxt else None
                    ),
                    previous_period=periods[i - 1] if i > 0 else None,
                )

        return ScheduleStatus(
            status="after_school",
            now=local,
            previous_period=periods[-1],
        )

    def periods_on(self, day: date) -> tuple:
        """某天的完整课表（按 start 排序）；无课 → 空元组。

        Phase 4 起供 contextualize 注入全天课表。
        """
        return self._schedule.days.get(day, ())

    def _at(self, day: date, t: time) -> datetime:
        """把纯时间挂到日程时区的某一天上。"""
        return datetime.combine(day, t, tzinfo=self._tzinfo)
