"""loader.py：JSON 课表读取与校验。"""

import json

import pytest

from app.schedule.loader import load_schedule

VALID_DATA = {
    "days": {
        "2026-09-10": [
            {"name": "P1", "kind": "class", "start": "08:00", "end": "08:45", "subject": "Math"},
            {"name": "Lunch", "kind": "lunch", "start": "11:30", "end": "12:15"},
            {"name": "Prep", "kind": "prep", "start": "12:15", "end": "12:55"},
        ],
        "2026-09-11": [],
    }
}


def test_load_dict_returns_schedule():
    schedule = load_schedule(VALID_DATA)
    from datetime import date

    periods = schedule.days[date(2026, 9, 10)]
    assert [p.name for p in periods] == ["P1", "Lunch", "Prep"]
    assert periods[0].kind == "class"
    assert periods[0].subject == "Math"
    assert periods[1].subject is None
    assert periods[0].start.strftime("%H:%M") == "08:00"


def test_load_json_file(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(VALID_DATA))
    schedule = load_schedule(path)
    from datetime import date

    assert schedule.days[date(2026, 9, 10)][0].name == "P1"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_schedule(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        load_schedule(path)


def test_missing_days_key_raises():
    with pytest.raises(ValueError, match="days"):
        load_schedule({"schedule": {}})


def test_invalid_day_key_raises():
    data = {"days": {"Sept-10": VALID_DATA["days"]["2026-09-10"]}}
    with pytest.raises(ValueError, match="Sept-10"):
        load_schedule(data)


def test_invalid_kind_raises():
    data = {
        "days": {
            "2026-09-10": [
                {"name": "Assembly", "kind": "assembly", "start": "08:00", "end": "09:00"},
            ]
        }
    }
    with pytest.raises(ValueError, match="kind"):
        load_schedule(data)


def test_missing_required_field_raises():
    data = {
        "days": {
            "2026-09-10": [
                {"name": "P1", "kind": "class", "start": "08:00"},
            ]
        }
    }
    with pytest.raises(ValueError, match="end"):
        load_schedule(data)


def test_invalid_time_format_raises():
    data = {
        "days": {
            "2026-09-10": [
                {"name": "P1", "kind": "class", "start": "8:00", "end": "08:45"},
            ]
        }
    }
    with pytest.raises(ValueError, match="HH:MM"):
        load_schedule(data)


def test_start_not_before_end_raises():
    data = {
        "days": {
            "2026-09-10": [
                {"name": "P1", "kind": "class", "start": "09:00", "end": "09:00"},
            ]
        }
    }
    with pytest.raises(ValueError, match="start"):
        load_schedule(data)


def test_overlapping_periods_raise():
    data = {
        "days": {
            "2026-09-10": [
                {"name": "P1", "kind": "class", "start": "08:00", "end": "09:00"},
                {"name": "P2", "kind": "class", "start": "08:30", "end": "09:30"},
            ]
        }
    }
    with pytest.raises(ValueError, match="[Oo]verlap"):
        load_schedule(data)


def test_empty_day_list_is_allowed():
    schedule = load_schedule({"days": {"2026-09-12": []}})
    from datetime import date

    assert schedule.days[date(2026, 9, 12)] == ()


def test_days_is_not_dict_raises():
    with pytest.raises(ValueError, match="days"):
        load_schedule({"days": [1, 2, 3]})
