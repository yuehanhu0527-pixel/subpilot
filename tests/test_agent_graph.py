"""graph.py + Agent 门面：LangGraph 最小 agent loop 的行为验证。

全部离线：FakeChatModel（脚本化）+ FakeEmbedder + ephemeral Chroma。
验证的是图的行为（路由、context 注入、真实工具执行、session 历史），
而非模型生成质量——反幻觉由「tool 返回 No relevant information found. +
系统提示禁止编造」机制保证。
"""

from datetime import date, datetime, time, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import app.agent as agent_mod
from app.agent import Agent
from app.agent.graph import build_agent_graph
from app.agent.sessions import InMemorySessionStore
from app.agent.tools import make_retrieve_documents_tool
from app.agent.tools.bathroom import (
    make_get_bathroom_pass_status_tool,
    make_start_bathroom_pass_tool,
)
from app.agent.tools.events import (
    make_list_today_events_tool,
    make_log_classroom_event_tool,
)
from app.llm.fake import FakeChatModel
from app.rag.models import TextChunk
from app.rag.store import ChromaStore
from app.schedule.engine import ScheduleEngine
from app.schedule.models import Period, Schedule
from app.services.tracker import BathroomPassRepo, ClassroomEventRepo
from tests.helpers import FakeEmbedder

NY = "America/New_York"
DAY = date(2026, 9, 10)

VOCAB = {
    "bathroom": [1, 0, 0, 0, 0, 0],
    "cafeteria": [0, 1, 0, 0, 0, 0],
    "homework": [0, 0, 1, 0, 0, 0],
    "dismissal": [0, 0, 0, 1, 0, 0],
}


def _period(name, kind, start, end, subject=None):
    return Period(
        name=name, kind=kind,
        start=time.fromisoformat(start), end=time.fromisoformat(end),
        subject=subject,
    )


@pytest.fixture
def schedule_engine():
    schedule = Schedule(
        days={
            DAY: (
                _period("P1", "class", "08:00", "08:45", subject="Math"),
                _period("P2", "class", "08:55", "09:40", subject="English"),
                _period("Lunch", "lunch", "11:30", "12:15"),
                _period("P3", "class", "13:00", "13:45", subject="Science"),
            )
        }
    )
    return ScheduleEngine(schedule, tz=NY)


@pytest.fixture
def retrieve_tool_with_docs():
    store = ChromaStore.ephemeral()
    embedder = FakeEmbedder(VOCAB, dim=6)
    store.add_chunks(
        "graph-docs",  # 独立 collection：ephemeral Chroma 进程内共享底层状态
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
    return make_retrieve_documents_tool(
        store=store, embedder=embedder, collection_name="graph-docs"
    )


def _schedule_msgs(model):
    """所有喂给模型的 SystemMessage。"""
    return [
        m
        for batch in model.recorded
        for m in batch
        if isinstance(m, SystemMessage)
    ]


# --- 要求 11 的四项行为 ---


def test_no_tool_question_answers_directly():
    model = FakeChatModel(
        responses=[AIMessage(content="I'm SubPilot, a substitute teacher assistant.")]
    )
    graph = build_agent_graph(model, [])
    result = graph.invoke({"messages": [HumanMessage(content="What are you?")]})

    assert result["messages"][-1].content == "I'm SubPilot, a substitute teacher assistant."
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])
    # 系统提示已注入
    assert any("SubPilot" in m.content for m in _schedule_msgs(model))


def test_schedule_context_injected_when_now_provided(schedule_engine):
    model = FakeChatModel(responses=[AIMessage(content="You are in P1 — Math.")])
    graph = build_agent_graph(model, [], schedule_engine=schedule_engine)
    now = datetime(2026, 9, 10, 8, 30, tzinfo=ZoneInfo(NY))
    result = graph.invoke({"messages": [HumanMessage(content="What period is it?")], "now": now})

    context_msgs = _schedule_msgs(model)
    assert any(
        "P1" in m.content and "Math" in m.content and "08:30" in m.content
        for m in context_msgs
    ), "schedule context 应作为 SystemMessage 注入"
    assert result["schedule_context"]
    assert result["messages"][-1].content == "You are in P1 — Math."


def test_document_question_triggers_real_retrieval(retrieve_tool_with_docs):
    model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_documents", "args": {"query": "bathroom pass policy"}, "id": "call_1"}
                ],
            ),
            AIMessage(content="Based on the handbook: a signed note is required."),
        ]
    )
    graph = build_agent_graph(model, [retrieve_tool_with_docs])
    result = graph.invoke({"messages": [HumanMessage(content="What's the bathroom pass policy?")]})

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "policy.pdf" in tool_msgs[0].content  # 真实检索执行了（Phase 1 管道）
    assert "signed note" in tool_msgs[0].content
    # 工具结果后回到 agent，产出最终回答
    assert result["messages"][-1].content == "Based on the handbook: a signed note is required."
    assert len(model.recorded) == 2  # agent 节点被调用两次（决策 + 综合）


def test_no_relevant_document_does_not_fabricate():
    empty_store = ChromaStore.ephemeral()
    empty_tool = make_retrieve_documents_tool(
        store=empty_store, embedder=FakeEmbedder(VOCAB, dim=6), collection_name="graph-empty"
    )
    model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_documents", "args": {"query": "fire drill procedure"}, "id": "call_2"}
                ],
            ),
            AIMessage(content="I don't know — the documents don't cover fire drill procedures."),
        ]
    )
    graph = build_agent_graph(model, [empty_tool])
    result = graph.invoke({"messages": [HumanMessage(content="What's the fire drill procedure?")]})

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content == "No relevant information found."
    # 第二轮模型收到的消息里带「无相关信息」的工具结果——模型据此答「不知道」
    second_call_msgs = model.recorded[1]
    assert any(
        isinstance(m, ToolMessage) and m.content == "No relevant information found."
        for m in second_call_msgs
    )
    assert "don't know" in result["messages"][-1].content


# --- 边界：now / engine 缺失 ---


def test_missing_now_skips_schedule_context(schedule_engine):
    model = FakeChatModel(responses=[AIMessage(content="ok")])
    graph = build_agent_graph(model, [], schedule_engine=schedule_engine)
    result = graph.invoke({"messages": [HumanMessage(content="hi")]})
    assert result["messages"][-1].content == "ok"
    assert not any("Current period" in m.content for m in _schedule_msgs(model))


def test_engine_none_skips_context_even_with_now():
    model = FakeChatModel(responses=[AIMessage(content="ok")])
    graph = build_agent_graph(model, [], schedule_engine=None)
    now = datetime(2026, 9, 10, 8, 30, tzinfo=ZoneInfo(NY))
    result = graph.invoke({"messages": [HumanMessage(content="hi")], "now": now})
    assert result["messages"][-1].content == "ok"
    assert not any("Current period" in m.content for m in _schedule_msgs(model))


# --- Agent 门面：session 历史与 schedule 文件加载（调整 4：两种状态区分）---


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        agent_mod,
        "settings",
        SimpleNamespace(data_dir=tmp_path, timezone=NY),
    )
    return tmp_path


def test_agent_carries_session_history(tmp_data_dir):
    model = FakeChatModel(
        responses=[AIMessage(content="hello!"), AIMessage(content="hello again!")]
    )
    agent = Agent(model=model, tools=[], sessions=InMemorySessionStore())

    assert agent.run("s1", "hi") == "hello!"
    assert agent.run("s1", "again") == "hello again!"

    # 第二轮发往模型的消息包含第一轮问答
    second_call = [m for batch in model.recorded for m in batch]
    contents = [m.content for m in second_call if isinstance(m, (HumanMessage, AIMessage))]
    assert "hi" in contents and "hello!" in contents and "again" in contents


def test_agent_without_schedule_file_injects_no_context(tmp_data_dir):
    # 调整 4：schedule 文件不存在 → 不注入 schedule context（区别于「当天无课」）
    model = FakeChatModel(responses=[AIMessage(content="ok")])
    agent = Agent(model=model, tools=[])
    assert agent._engine is None
    now = datetime(2026, 9, 10, 8, 30, tzinfo=ZoneInfo(NY))
    agent.run("s1", "what period is it?", now=now)
    assert not any("Current period" in m.content for m in _schedule_msgs(model))


def test_agent_with_schedule_file_loads_engine(tmp_data_dir):
    import json

    (tmp_data_dir / "schedule.json").write_text(
        json.dumps(
            {
                "days": {
                    "2026-09-10": [
                        {"name": "P1", "kind": "class", "start": "08:00", "end": "08:45", "subject": "Math"}
                    ]
                }
            }
        )
    )
    model = FakeChatModel(responses=[AIMessage(content="You are in P1 — Math.")])
    agent = Agent(model=model, tools=[])
    assert agent._engine is not None
    now = datetime(2026, 9, 10, 8, 30, tzinfo=ZoneInfo(NY))
    agent.run("s1", "what period is it?", now=now)
    assert any(
        "P1" in m.content and "Math" in m.content for m in _schedule_msgs(model)
    )


def test_agent_with_empty_schedule_file_says_no_classes(tmp_data_dir):
    # 调整 4：文件存在但无当日条目 → 明确「当天无课」context（不是静默）
    (tmp_data_dir / "schedule.json").write_text('{"days": {}}')
    model = FakeChatModel(responses=[AIMessage(content="ok")])
    agent = Agent(model=model, tools=[])
    assert agent._engine is not None
    now = datetime(2026, 9, 10, 8, 30, tzinfo=ZoneInfo(NY))
    agent.run("s1", "what period is it?", now=now)
    assert any(
        "No classes are scheduled today." in m.content for m in _schedule_msgs(model)
    )


def test_agent_sessions_are_isolated(tmp_data_dir):
    model = FakeChatModel(
        responses=[AIMessage(content="s1 answer"), AIMessage(content="s2 answer")]
    )
    agent = Agent(model=model, tools=[], sessions=InMemorySessionStore())
    agent.run("s1", "question for s1")
    agent.run("s2", "question for s2")
    assert model.recorded[0][-1].content == "question for s1"
    assert model.recorded[1][-1].content == "question for s2"


# --- Phase 4：全天课表注入、多工具流程、默认工具组装 ---


def test_contextualize_includes_full_day_schedule(schedule_engine):
    model = FakeChatModel(responses=[AIMessage(content="ok")])
    graph = build_agent_graph(model, [], schedule_engine=schedule_engine)
    now = datetime(2026, 9, 10, 8, 30, tzinfo=ZoneInfo(NY))
    graph.invoke({"messages": [HumanMessage(content="when is lunch?")], "now": now})
    assert any(
        "Today's schedule:" in m.content and "Lunch" in m.content and "P3" in m.content
        for m in _schedule_msgs(model)
    )


def _fixed_clock_box(hour, minute):
    box = {"now": datetime(2026, 9, 10, hour, minute, tzinfo=timezone.utc)}
    return box


def test_bathroom_pass_flow_with_multiple_tool_calls(tmp_path, schedule_engine):
    repo = BathroomPassRepo(tmp_path / "subpilot.db")
    box = _fixed_clock_box(13, 15)  # = 09:15 EDT
    tools = [
        make_get_bathroom_pass_status_tool(repo, lambda: box["now"], tz_name=NY),
        make_start_bathroom_pass_tool(repo, lambda: box["now"], lambda now: "2026-09-10", tz_name=NY),
    ]
    model = FakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "get_bathroom_pass_status", "args": {"student": "Alice"}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "start_bathroom_pass", "args": {"student": "Alice"}, "id": "c2"}]),
            AIMessage(content="Alice's bathroom pass is recorded."),
        ]
    )
    graph = build_agent_graph(model, tools, schedule_engine=schedule_engine)
    result = graph.invoke({"messages": [HumanMessage(content="Alice is going to the bathroom")]})

    tool_contents = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_contents[0] == "Alice has no bathroom pass records."
    assert tool_contents[1] == "Recorded: Alice left at 09:15."
    assert result["messages"][-1].content == "Alice's bathroom pass is recorded."
    assert len(model.recorded) == 3  # status → start → 最终回答


def test_duplicate_bathroom_start_rejected_in_loop(tmp_path):
    repo = BathroomPassRepo(tmp_path / "subpilot.db")
    box = _fixed_clock_box(13, 15)
    start = make_start_bathroom_pass_tool(repo, lambda: box["now"], lambda now: "2026-09-10", tz_name=NY)
    model = FakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "start_bathroom_pass", "args": {"student": "Alice"}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "start_bathroom_pass", "args": {"student": "Alice"}, "id": "c2"}]),
            AIMessage(content="Alice already has an open pass, so I didn't create a duplicate."),
        ]
    )
    graph = build_agent_graph(model, [start])
    result = graph.invoke({"messages": [HumanMessage(content="record Alice leaving, twice")]})

    tool_contents = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_contents[0].startswith("Recorded:")
    assert "already has an open bathroom pass" in tool_contents[1]
    assert result["messages"][-1].content.startswith("Alice already has an open pass")


def test_event_log_and_report_grounded_in_real_data(tmp_path, schedule_engine):
    repo = ClassroomEventRepo(tmp_path / "subpilot.db")
    box = _fixed_clock_box(12, 30)  # = 08:30 EDT → P1
    tools = [
        make_log_classroom_event_tool(repo, lambda: box["now"], lambda now: "2026-09-10",
                                      schedule_engine=schedule_engine, tz_name=NY),
        make_list_today_events_tool(repo, lambda: box["now"], lambda now: "2026-09-10", tz_name=NY),
    ]
    model = FakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "log_classroom_event", "args": {"description": "Students worked quietly on the math quiz."}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "list_today_events", "args": {}, "id": "c2"}]),
            AIMessage(content="End-of-day note: Students worked quietly on the math quiz."),
        ]
    )
    graph = build_agent_graph(model, tools, schedule_engine=schedule_engine)
    result = graph.invoke({"messages": [HumanMessage(content="Log an event, then give me today's summary")]})

    tool_contents = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    # period 由引擎自动解析为 P1；事件真实入库
    assert tool_contents[0] == "Event logged: [P1] 08:30 — Students worked quietly on the math quiz."
    # 报告数据 = 真实入库事件列表（防幻觉机制断言）
    assert "Students worked quietly" in tool_contents[1]
    final_batch = model.recorded[-1]
    assert any(
        isinstance(m, ToolMessage) and "Students worked quietly" in m.content
        for m in final_batch
    )
    assert result["messages"][-1].content == "End-of-day note: Students worked quietly on the math quiz."


def test_invalid_period_rejected_in_loop(tmp_path, schedule_engine):
    repo = ClassroomEventRepo(tmp_path / "subpilot.db")
    box = _fixed_clock_box(12, 30)
    tool = make_log_classroom_event_tool(repo, lambda: box["now"], lambda now: "2026-09-10",
                                         schedule_engine=schedule_engine, tz_name=NY)
    model = FakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "log_classroom_event", "args": {"description": "something", "period": "P9"}, "id": "c1"}]),
            AIMessage(content="I couldn't log that: P9 isn't in today's schedule."),
        ]
    )
    graph = build_agent_graph(model, [tool], schedule_engine=schedule_engine)
    result = graph.invoke({"messages": [HumanMessage(content="log an event in P9")]})
    tool_contents = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_contents[0].startswith("Error: unknown period")
    assert repo.events_on("2026-09-10") == []


def test_agent_default_tools_assembled(tmp_data_dir):
    model = FakeChatModel(responses=[AIMessage(content="ok")])
    agent = Agent(model=model)  # tools=None → 默认组装
    names = sorted(t.name for t in agent._tools)
    assert names == [
        "end_bathroom_pass",
        "get_bathroom_pass_status",
        "list_today_events",
        "log_classroom_event",
        "retrieve_documents",
        "start_bathroom_pass",
    ]
