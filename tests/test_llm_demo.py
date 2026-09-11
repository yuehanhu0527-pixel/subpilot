"""DemoChatModel：确定性零 key provider 的行为测试。"""

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, ToolMessage

import app.llm as llm_mod
from app.llm.demo import DEMO_NOTE, DemoChatModel
from app.agent.tools import make_retrieve_documents_tool


def test_human_message_triggers_retrieval_tool_call():
    model = DemoChatModel()
    tool = make_retrieve_documents_tool()
    model.bind_tools([tool])
    msg = model.invoke([HumanMessage(content="bathroom pass policy?")])
    assert msg.tool_calls
    call = msg.tool_calls[0]
    assert call["name"] == "retrieve_documents"
    assert call["args"] == {"query": "bathroom pass policy?"}


def test_tool_result_is_answered_with_citation():
    model = DemoChatModel()
    model.bind_tools([make_retrieve_documents_tool()])
    tool_msg = ToolMessage(
        content="Source: handbook.txt\nBathroom passes require a signed note.",
        tool_call_id="c1",
    )
    answer = model.invoke([HumanMessage(content="q"), tool_msg]).content
    assert "Bathroom passes require a signed note." in answer
    assert "Source: handbook.txt" in answer
    assert DEMO_NOTE in answer


def test_no_hits_is_stated_plainly():
    model = DemoChatModel()
    model.bind_tools([make_retrieve_documents_tool()])
    tool_msg = ToolMessage(content="No relevant information found.", tool_call_id="c1")
    answer = model.invoke([HumanMessage(content="q"), tool_msg]).content
    assert "don't cover" in answer
    assert DEMO_NOTE in answer


def test_get_provider_demo(monkeypatch):
    monkeypatch.setattr(llm_mod, "settings", SimpleNamespace(llm_provider="demo"))
    assert llm_mod.get_provider()._llm_type == "demo"


def test_full_graph_loop_answers_with_citation(store, collection):
    """回归：graph 里 contextualize 追加 SystemMessage 后仍能触发检索并作答。"""
    from app.agent import Agent
    from app.agent.sessions import InMemorySessionStore
    from app.rag.models import TextChunk
    from tests.helpers import FakeEmbedder

    vocab = {"bathroom": [1, 0, 0, 0, 0, 0]}
    embedder = FakeEmbedder(vocab, dim=6)
    store.add_chunks(
        collection,
        [TextChunk(id="p:0", text="Bathroom passes require a signed note.", source_name="handbook.txt")],
        embedder.encode(["Bathroom passes require a signed note."]),
    )
    agent = Agent(
        model=DemoChatModel(),
        tools=[make_retrieve_documents_tool(store=store, embedder=embedder, collection_name=collection)],
        sessions=InMemorySessionStore(),
    )
    turn = agent.run_turn("s1", "What is the bathroom pass policy?")
    assert "Bathroom passes require a signed note." in turn.answer
    assert list(turn.sources) == ["handbook.txt"]
