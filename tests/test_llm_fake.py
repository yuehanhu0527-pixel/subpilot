"""fake.py：脚本化确定性响应模型（仅测试 / Demo 使用）。"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.llm.fake import FakeChatModel


def test_returns_scripted_responses_in_order():
    model = FakeChatModel(responses=[AIMessage(content="first"), AIMessage(content="second")])
    assert model.invoke([HumanMessage(content="hi")]).content == "first"
    assert model.invoke([HumanMessage(content="hi again")]).content == "second"


def test_returns_scripted_tool_call():
    model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "retrieve_documents", "args": {"query": "bathroom"}, "id": "call_1"}],
            )
        ]
    )
    out = model.invoke([HumanMessage(content="policy question")])
    assert out.content == ""
    # langchain-core 会给 tool_calls 补 "type": "tool_call"，这里只断言语义字段
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["name"] == "retrieve_documents"
    assert out.tool_calls[0]["args"] == {"query": "bathroom"}
    assert out.tool_calls[0]["id"] == "call_1"


def test_records_every_message_batch():
    model = FakeChatModel(
        responses=[AIMessage(content="ok"), AIMessage(content="ok2")]
    )
    model.invoke([SystemMessage(content="sys"), HumanMessage(content="q1")])
    model.invoke([HumanMessage(content="q2")])
    assert len(model.recorded) == 2
    assert [type(m).__name__ for m in model.recorded[0]] == ["SystemMessage", "HumanMessage"]
    assert model.recorded[1][0].content == "q2"


def test_raises_when_script_exhausted():
    model = FakeChatModel(responses=[])
    with pytest.raises(RuntimeError, match="exhausted"):
        model.invoke([HumanMessage(content="hi")])


def test_bind_tools_records_tool_names():
    @tool
    def echo(text: str) -> str:
        """Echo back the text."""
        return text

    model = FakeChatModel(responses=[AIMessage(content="ok")])
    bound = model.bind_tools([echo])
    assert bound is model  # fake 不需要真实绑定，保持同一实例以共享脚本
    assert model.bound_tool_names == ["echo"]
