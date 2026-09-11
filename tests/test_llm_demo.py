"""DemoChatModel：确定性零 key provider 的行为测试。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

import app.llm as llm_mod
from app.llm.demo import DEMO_NOTE, DemoChatModel
from app.agent import Agent
from app.agent.sessions import InMemorySessionStore
from app.agent.tools import make_retrieve_documents_tool
from app.rag.lexical import LexicalEmbedder
from app.rag.pipeline import ingest_file

HANDBOOK = Path(__file__).resolve().parent.parent / "sample_data" / "substitute-handbook.md"

LESSON_PLAN = """# Science Lesson Plan — Period 3 (Grade 6)

## Objective
Students will complete a guided inquiry lab on plant transpiration.

## Period 1: Warm-up
Complete the vocabulary worksheet in the science folder.

## Period 3: Lab Activity
Students should be doing the transpiration lab in pairs: set up the celery
experiment, then record observations every 10 minutes in their science
notebook. The lab must be supervised at all times. Students may start the
science lab today only after passing the safety quiz on Friday.

## Period 5: Wrap-up
Collect the observation sheets and have students write a one-sentence
summary of their results.
"""


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


def _agent_with_docs(store, collection, tmp_path) -> Agent:
    """真实链路：示例手册 + lesson plan 经 loader/chunker/lexical 入库。"""
    embedder = LexicalEmbedder()
    ingest_file(HANDBOOK, collection_name=collection, store=store, embedder=embedder)
    lesson = tmp_path / "lesson_plan.md"
    lesson.write_text(LESSON_PLAN)
    ingest_file(lesson, collection_name=collection, store=store, embedder=embedder)
    tool = make_retrieve_documents_tool(store=store, embedder=embedder, collection_name=collection)
    return Agent(model=DemoChatModel(), tools=[tool], sessions=InMemorySessionStore())


@pytest.mark.parametrize(
    "question,expect,absent,source",
    [
        # 只引用相关句子，不整块贴出 chunk
        ("What should Period 3 be doing?", "transpiration lab", "vocabulary worksheet", "lesson_plan.md"),
        ("Can students start the science lab today?", "safety quiz", "vocabulary worksheet", "lesson_plan.md"),
        # 跨 chunk 续接：断句被第二命中补齐
        ("What should I do during a fire drill?", "single file", "Thank you for covering", "substitute-handbook.md"),
        ("How long can a student be out for the bathroom?", "5 minutes", "nut-free", "substitute-handbook.md"),
    ],
)
def test_demo_answers_quote_relevant_sentences(tmp_path, store, question, expect, absent, source):
    # collection 名不能来自 nodeid（含参数化文本会超长/以 - 结尾）
    name = f"demo-{abs(hash(question))}"
    agent = _agent_with_docs(store, name, tmp_path)
    turn = agent.run_turn("s", question)
    assert expect in turn.answer, turn.answer
    assert absent not in turn.answer, turn.answer
    assert source in turn.answer  # 保留原 citation
    assert len(turn.answer) < 600  # 不再整块输出 chunk（chunk 本身 640–958 字符）
