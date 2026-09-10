"""Phase 4 领域仓储：bathroom pass 与课堂事件（标准库 sqlite，单文件）。

设计要点：
- 时间一律由调用方（tool 层）显式注入 aware datetime；day 为学校本地日期字符串。
- 每个学生至多一个未结束的 pass：事务内先查后写 + 部分唯一索引双保险。
- 学生名用 casefold 键比较，防大小写变体绕过唯一约束。
- 事件为 append-only 事实日志：无任何分类/标签字段。
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bathroom_passes (
    id INTEGER PRIMARY KEY,
    student TEXT NOT NULL,
    student_key TEXT NOT NULL,
    day TEXT NOT NULL,
    departed_at TEXT NOT NULL,
    returned_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_open_pass
    ON bathroom_passes(student_key) WHERE returned_at IS NULL;
CREATE TABLE IF NOT EXISTS classroom_events (
    id INTEGER PRIMARY KEY,
    day TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    period TEXT,
    description TEXT NOT NULL,
    students TEXT NOT NULL
);
"""


class DuplicateOpenPassError(Exception):
    def __init__(self, student: str, departed_at: datetime):
        self.student = student
        self.departed_at = departed_at
        super().__init__(f"{student} already has an open bathroom pass (left at {departed_at})")


class NoOpenPassError(Exception):
    def __init__(self, student: str):
        self.student = student
        super().__init__(f"{student} has no open bathroom pass")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _ensure_schema(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)


def _floor_minutes(delta_seconds: float) -> int:
    """整分钟向下取整：已离开时长宁可少报，不夸大。"""
    return int(delta_seconds // 60)


def school_day_fn(tz_name: str | None):
    """返回 (aware datetime) -> 'YYYY-MM-DD' 学校本地日期的函数；tz 缺失时报错。"""
    if not tz_name:
        raise ValueError("No timezone configured: set SUBPILOT_TIMEZONE")
    tz = ZoneInfo(tz_name)

    def day(now: datetime) -> str:
        return now.astimezone(tz).date().isoformat()

    return day


def local_hhmm(dt: datetime, tz_name: str | None) -> str:
    """aware 时刻 → 学校本地 'HH:MM'；tz 缺失时退化为 UTC 显示。"""
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    return dt.astimezone(tz).strftime("%H:%M")


class BathroomPassRepo:
    def __init__(self, path: str | Path):
        self._path = str(path)
        _ensure_schema(self._path)

    def start_pass(self, student: str, departed_at: datetime, day: str) -> None:
        """记录学生离开；已有未结束 pass 时抛 DuplicateOpenPassError。"""
        with _connect(self._path) as conn:
            try:
                conn.execute(
                    "INSERT INTO bathroom_passes (student, student_key, day, departed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (student, student.casefold(), day, _iso(departed_at)),
                )
            except sqlite3.IntegrityError:
                raise DuplicateOpenPassError(student, departed_at)

    def end_pass(self, student: str, returned_at: datetime) -> tuple[str, datetime, int]:
        """记录学生返回；返回 (原名, 离开时间, 在外分钟数)。无未结束 pass 时抛错。"""
        with _connect(self._path) as conn:
            row = conn.execute(
                "SELECT student, departed_at FROM bathroom_passes "
                "WHERE student_key = ? AND returned_at IS NULL",
                (student.casefold(),),
            ).fetchone()
            if row is None:
                raise NoOpenPassError(student)
            name, departed = row[0], _from_iso(row[1])
            conn.execute(
                "UPDATE bathroom_passes SET returned_at = ? "
                "WHERE student_key = ? AND returned_at IS NULL",
                (_iso(returned_at), student.casefold()),
            )
        minutes_out = _floor_minutes((returned_at - departed).total_seconds())
        return name, departed, minutes_out

    def open_passes(self, now: datetime) -> list[dict]:
        """所有未结束的 pass（按离开时间排序），含已离开分钟数。"""
        with _connect(self._path) as conn:
            rows = conn.execute(
                "SELECT student, departed_at FROM bathroom_passes "
                "WHERE returned_at IS NULL ORDER BY departed_at"
            ).fetchall()
        return [
            {
                "student": name,
                "departed_at": _from_iso(departed),
                "minutes_out": _floor_minutes((now - _from_iso(departed)).total_seconds()),
            }
            for name, departed in rows
        ]

    def student_status(self, student: str, now: datetime) -> dict | None:
        """某个学生当前是否在外；不在外返回 None。"""
        with _connect(self._path) as conn:
            row = conn.execute(
                "SELECT student, departed_at FROM bathroom_passes "
                "WHERE student_key = ? AND returned_at IS NULL",
                (student.casefold(),),
            ).fetchone()
        if row is None:
            return None
        departed = _from_iso(row[1])
        return {
            "student": row[0],
            "departed_at": departed,
            "minutes_out": _floor_minutes((now - departed).total_seconds()),
        }

    def last_pass(self, student: str) -> tuple[datetime, datetime | None] | None:
        """最近一次 pass 的 (离开, 返回)；从未记录过返回 None。"""
        with _connect(self._path) as conn:
            row = conn.execute(
                "SELECT departed_at, returned_at FROM bathroom_passes "
                "WHERE student_key = ? ORDER BY id DESC LIMIT 1",
                (student.casefold(),),
            ).fetchone()
        if row is None:
            return None
        return _from_iso(row[0]), (_from_iso(row[1]) if row[1] else None)


class ClassroomEventRepo:
    def __init__(self, path: str | Path):
        self._path = str(path)
        _ensure_schema(self._path)

    def add_event(
        self,
        day: str,
        recorded_at: datetime,
        description: str,
        period: str | None = None,
        students: list[str] | None = None,
    ) -> int:
        """记录一条事实性课堂事件；description 必须非空。"""
        if not description.strip():
            raise ValueError("Event description must not be empty")
        with _connect(self._path) as conn:
            cursor = conn.execute(
                "INSERT INTO classroom_events (day, recorded_at, period, description, students) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    day,
                    _iso(recorded_at),
                    period,
                    description.strip(),
                    json.dumps(students or []),
                ),
            )
            return int(cursor.lastrowid)

    def events_on(self, day: str) -> list[dict]:
        """某天的全部事件，按记录时间先后排序。"""
        with _connect(self._path) as conn:
            rows = conn.execute(
                "SELECT recorded_at, period, description, students "
                "FROM classroom_events WHERE day = ? ORDER BY recorded_at, id",
                (day,),
            ).fetchall()
        return [
            {
                "recorded_at": _from_iso(recorded_at),
                "period": period,
                "description": description,
                "students": json.loads(students),
            }
            for recorded_at, period, description, students in rows
        ]
