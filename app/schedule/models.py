"""日程数据模型：period、按日期键控的课表、状态查询结果。

Phase 2：纯确定性数据结构，不涉及任何时间判断逻辑（见 engine.py）。
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Literal, Mapping

PeriodKind = Literal["class", "lunch", "prep", "other"]
StatusValue = Literal[
    "before_school", "in_period", "between_periods", "after_school", "no_classes"
]

VALID_PERIOD_KINDS = frozenset({"class", "lunch", "prep", "other"})
VALID_STATUS_VALUES = frozenset(
    {"before_school", "in_period", "between_periods", "after_school", "no_classes"}
)


@dataclass(frozen=True)
class Period:
    """一节课/一段日程：课堂、午餐、备课或其它。

    start 闭区间、end 开区间（恰好 end 时不属于该 period）。
    subject 仅对 kind="class" 有意义，lunch/prep 为 None。
    """

    name: str
    kind: PeriodKind
    start: time
    end: time
    subject: str | None = None

    def __post_init__(self):
        if self.kind not in VALID_PERIOD_KINDS:
            raise ValueError(
                f"Invalid period kind: {self.kind!r} "
                f"(valid: {', '.join(sorted(VALID_PERIOD_KINDS))})"
            )
        if self.start >= self.end:
            raise ValueError(
                f"Period {self.name!r}: start {self.start} must be before end {self.end}"
            )


@dataclass(frozen=True)
class Schedule:
    """按日期键控的课表：days[date] = 当天 periods（按 start 排序）。

    某天无条目或条目为空 → 当天无课。构造时校验：排序 + 拒绝重叠。
    """

    days: Mapping[date, tuple[Period, ...]]

    def __post_init__(self):
        object.__setattr__(
            self, "days", {day: self._checked(periods) for day, periods in self.days.items()}
        )

    @staticmethod
    def _checked(periods: tuple[Period, ...]) -> tuple[Period, ...]:
        ordered = tuple(sorted(periods, key=lambda p: (p.start, p.end, p.name)))
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.start < prev.end:
                raise ValueError(
                    f"Overlapping periods: {prev.name!r} (ends {prev.end}) and "
                    f"{nxt.name!r} (starts {nxt.start})"
                )
        return ordered


@dataclass(frozen=True)
class ScheduleStatus:
    """一次状态查询的完整结果；各字段随 status 语义见 engine.py。"""

    status: StatusValue
    now: datetime
    current_period: Period | None = None
    minutes_remaining: int | None = None
    next_period: Period | None = None
    minutes_until_next: int | None = None
    previous_period: Period | None = None

    def __post_init__(self):
        if self.status not in VALID_STATUS_VALUES:
            raise ValueError(f"Invalid status: {self.status!r}")
