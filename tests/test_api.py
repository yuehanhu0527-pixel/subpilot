"""app/api.py：UI 与既有 Agent / schedule / tracker 之间的薄适配层。

依赖全部可注入（monkeypatch api_mod 的钩子），全程离线。
"""

from datetime import date, datetime, time, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app.api as api_mod
from app.agent import Agent
from app.agent.sessions import InMemorySessionStore
from app.agent.tools import make_retrieve_documents_tool
from app.llm.fake import FakeChatModel
from app.rag import pipeline as pipeline_mod
from app.rag.models import TextChunk
from app.rag.store import ChromaStore
from app.schedule.engine import ScheduleEngine
from app.schedule.loader import load_schedule
from app.schedule.models import Period, Schedule
from app.main import app
from tests.helpers import FakeEmbedder

VOCAB = {"bathroom": [1, 0, 0, 0, 0, 0]}
NY = "America/New_York"


@pytest.fixture
def client():
    return TestClient(app)


class FakeDatetime(datetime):
    """把 api 的 datetime.now 固定到 2026-09-10 12:30 UTC（= 08:30 EDT）。"""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc)


@pytest.fixture
def fixed_now(monkeypatch):
    monkeypatch.setattr(api_mod, "datetime", FakeDatetime)


@pytest.fixture
def fake_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_mod, "settings", SimpleNamespace(data_dir=tmp_path, timezone=NY)
    )
    return tmp_path


def _scripted_agent():
    store = ChromaStore.ephemeral()
    embedder = FakeEmbedder(VOCAB, dim=6)
    store.add_chunks(
        "api-docs",
        [
            TextChunk(
                id="p:0",
                text="Bathroom passes require a signed note from the office.",
                source_name="policy.pdf",
                page=3,
                heading="Bathroom Passes",
            )
        ],
        embedder.encode(["Bathroom passes require a signed note from the office."]),
    )
    tool = make_retrieve_documents_tool(
        store=store, embedder=embedder, collection_name="api-docs"
    )
    model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_documents", "args": {"query": "bathroom pass"}, "id": "c1"}
                ],
            ),
            AIMessage(content="Students need a signed note."),
        ]
    )
    return Agent(model=model, tools=[tool], sessions=InMemorySessionStore())


# --- chat ---


def test_chat_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_get_agent", lambda: _scripted_agent())
    r = client.post("/api/chat", json={"session_id": "s1", "text": "bathroom pass policy?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Students need a signed note."
    assert body["sources"] == ["policy.pdf · (p.3) · § Bathroom Passes"]


def test_chat_missing_config_returns_503(client, monkeypatch):
    def boom():
        raise ValueError("Missing environment variable(s): SUBPILOT_LLM_API_KEY")

    monkeypatch.setattr(api_mod, "_get_agent", boom)
    r = client.post("/api/chat", json={"session_id": "s1", "text": "hi"})
    assert r.status_code == 503
    assert "SUBPILOT_LLM_API_KEY" in r.json()["detail"]


def test_chat_llm_runtime_failure_returns_502(client, monkeypatch):
    class BoomModel(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("provider down")

    agent = Agent(model=BoomModel(), tools=[], sessions=InMemorySessionStore())
    monkeypatch.setattr(api_mod, "_get_agent", lambda: agent)
    r = client.post("/api/chat", json={"session_id": "s1", "text": "hi"})
    assert r.status_code == 502
    assert "provider down" in r.json()["detail"]


def test_chat_requires_text(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_get_agent", lambda: _scripted_agent())
    r = client.post("/api/chat", json={"session_id": "s1"})
    assert r.status_code == 422


# --- status ---


def test_status_without_schedule_file(client, fake_settings, fixed_now, monkeypatch):
    monkeypatch.setattr(api_mod, "_load_engine", lambda: None)
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["schedule"] is None
    assert body["bathroom"] == []
    assert body["events"] == []
    assert body["documents"] == []
    assert any("schedule" in n for n in body["notes"])


def test_status_with_schedule_shows_current_period(client, fake_settings, fixed_now, monkeypatch):
    schedule = Schedule(
        days={
            date(2026, 9, 10): (
                Period(name="P1", kind="class", start=time(8, 0), end=time(8, 45), subject="Math"),
                Period(name="Lunch", kind="lunch", start=time(11, 30), end=time(12, 15)),
            )
        }
    )
    monkeypatch.setattr(
        api_mod, "_load_engine", lambda: ScheduleEngine(schedule, tz=NY)
    )
    r = client.get("/api/status")
    body = r.json()
    assert body["schedule"]["status"] == "in_period"
    assert body["schedule"]["current_period"]["name"] == "P1"
    assert body["schedule"]["current_period"]["subject"] == "Math"
    assert body["schedule"]["minutes_remaining"] == 15
    assert [p["name"] for p in body["schedule"]["periods_today"]] == ["P1", "Lunch"]


def test_status_bathroom_and_events_from_db(client, fake_settings, fixed_now, monkeypatch):
    from app.services.tracker import BathroomPassRepo, ClassroomEventRepo

    monkeypatch.setattr(api_mod, "_load_engine", lambda: None)
    BathroomPassRepo(fake_settings / "subpilot.db").start_pass(
        "Alice", datetime(2026, 9, 10, 12, 15, tzinfo=timezone.utc), "2026-09-10"
    )
    ClassroomEventRepo(fake_settings / "subpilot.db").add_event(
        "2026-09-10", datetime(2026, 9, 10, 12, 20, tzinfo=timezone.utc),
        "Students finished the quiz.", period="P1",
    )
    r = client.get("/api/status")
    body = r.json()
    assert body["bathroom"][0]["student"] == "Alice"
    assert body["bathroom"][0]["minutes_out"] == 15
    assert body["events"][0]["description"] == "Students finished the quiz."
    assert body["events"][0]["time"] == "08:20"  # 本地时间显示
    assert body["events"][0]["period"] == "P1"


# --- documents upload / list ---


def test_upload_ingests_and_lists(client, fake_settings, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "_store", ChromaStore.ephemeral())
    monkeypatch.setattr(pipeline_mod, "_embedder", FakeEmbedder(VOCAB, dim=6))
    r = client.post(
        "/api/documents",
        files={"file": ("rules.txt", b"Bathroom passes require a signed note.", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "rules.txt"
    assert r.json()["chunks"] >= 1

    docs = client.get("/api/documents").json()
    assert [d["name"] for d in docs] == ["rules.txt"]


def test_upload_unsupported_type_rejected(client, fake_settings, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "_store", ChromaStore.ephemeral())
    monkeypatch.setattr(pipeline_mod, "_embedder", FakeEmbedder(VOCAB, dim=6))
    r = client.post(
        "/api/documents",
        files={"file": ("sheet.xlsx", b"x", "application/octet-stream")},
    )
    assert r.status_code == 400


# --- report ---


def test_report_returns_text(client, monkeypatch):
    agent = Agent(
        model=FakeChatModel(responses=[AIMessage(content="End-of-day note: quiet day.")]),
        tools=[],
    )
    monkeypatch.setattr(api_mod, "_get_agent", lambda: agent)
    r = client.post("/api/report", json={"session_id": "s1"})
    assert r.status_code == 200
    assert r.json()["report"] == "End-of-day note: quiet day."


# --- 静态页面（Phase 5 UI）---


def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "What should I do now?" in r.text
