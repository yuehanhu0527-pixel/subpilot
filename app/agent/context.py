"""确定性 contextualize：把 ScheduleStatus 转成注入 Agent 的文本上下文。

零 LLM 调用——纯字符串模板。注意两种状态的区分（由 Agent 层保证）：
- schedule 文件不存在 → 不注入任何 context（本模块不参与）；
- 文件存在但当天无课 → ScheduleStatus.no_classes → 明确说 "No classes are scheduled today."。
"""

from app.schedule.models import Period, ScheduleStatus


def _label(period: Period) -> str:
    """P1 — Math（有 subject 的课）；Lunch / Prep 只有名字。"""
    if period.subject:
        return f"{period.name} — {period.subject}"
    return period.name


def _minutes(n: int) -> str:
    return f"{n} minute{'s' if n != 1 else ''}"


def _fmt(t) -> str:
    return t.strftime("%H:%M")


def build_schedule_context(status: ScheduleStatus) -> str:
    """按五种状态生成确定性上下文文本。"""
    lines = [
        f"It is currently {status.now:%H:%M} ({status.now.tzinfo}).",
        "",
    ]

    if status.status == "in_period":
        current = status.current_period
        lines.append(
            f"Current period: {_label(current)} "
            f"({_fmt(current.start)}–{_fmt(current.end)}). "
            f"{status.minutes_remaining} minutes remaining."
        )
        if status.next_period is not None:
            lines.append(
                f"Next period: {_label(status.next_period)} "
                f"at {_fmt(status.next_period.start)}."
            )
        else:
            lines.append("Next period: none.")

    elif status.status == "between_periods":
        prev, nxt = status.previous_period, status.next_period
        lines.append(
            f"Between periods: {_label(prev)} ended at {_fmt(prev.end)}."
        )
        lines.append(
            f"Next period: {_label(nxt)} starts at {_fmt(nxt.start)}, "
            f"in {_minutes(status.minutes_until_next)}."
        )

    elif status.status == "before_school":
        first = status.next_period
        lines.append("Before school.")
        lines.append(
            f"First period: {_label(first)} starts at {_fmt(first.start)}, "
            f"in {_minutes(status.minutes_until_next)}."
        )

    elif status.status == "after_school":
        last = status.previous_period
        lines.append(f"After school. Last period: {_label(last)} ended at {_fmt(last.end)}.")

    elif status.status == "no_classes":
        lines.append("No classes are scheduled today.")

    return "\n".join(lines)
