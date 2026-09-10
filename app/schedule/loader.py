"""结构化课程时间的读取：JSON 文件或 dict → Schedule。

JSON 结构：
{
  "days": {
    "2026-09-10": [
      {"name": "Period 1", "kind": "class", "start": "08:00", "end": "08:45", "subject": "Math"},
      {"name": "Lunch", "kind": "lunch", "start": "11:30", "end": "12:15"}
    ]
  }
}

校验：日期键格式、period 字段、kind 取值、时间格式；start<end 与
重叠检查由 models.Period / models.Schedule 构造时完成。
"""

import json
import re
from datetime import datetime
from pathlib import Path

from .models import Period, Schedule, VALID_PERIOD_KINDS

_TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def load_schedule(source: str | Path | dict) -> Schedule:
    """从 JSON 文件路径或 dict 读取课表。"""
    if isinstance(source, dict):
        data = source
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))

    if not isinstance(data, dict) or "days" not in data or not isinstance(data["days"], dict):
        raise ValueError("Schedule must be a JSON object with a 'days' object")

    days = {}
    for key, raw_periods in data["days"].items():
        try:
            day = datetime.strptime(key, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid day key: {key!r} (expected YYYY-MM-DD)")
        if not isinstance(raw_periods, list):
            raise ValueError(f"Periods for {key!r} must be a list")
        days[day] = tuple(_parse_period(raw) for raw in raw_periods)

    return Schedule(days=days)


def _parse_period(raw: dict) -> Period:
    if not isinstance(raw, dict):
        raise ValueError(f"Period must be an object, got: {raw!r}")
    try:
        name = raw["name"]
        kind = raw["kind"]
        start = raw["start"]
        end = raw["end"]
    except KeyError as exc:
        raise ValueError(f"Period is missing required field {exc.args[0]!r}: {raw!r}")

    if kind not in VALID_PERIOD_KINDS:
        raise ValueError(
            f"Invalid period kind: {kind!r} (valid: {', '.join(sorted(VALID_PERIOD_KINDS))})"
        )
    return Period(
        name=str(name),
        kind=kind,
        start=_parse_time(start, "start"),
        end=_parse_time(end, "end"),
        subject=raw.get("subject"),
    )


def _parse_time(value, field: str):
    if not isinstance(value, str) or not _TIME_RE.fullmatch(value):
        raise ValueError(f"Invalid {field} time: {value!r} (expected HH:MM)")
    return datetime.strptime(value, "%H:%M").time()
