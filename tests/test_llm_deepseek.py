"""deepseek.py：OpenAI-compatible API 适配（httpx.MockTransport，全离线）。"""

import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import app.llm as llm_mod
from app.llm.deepseek import DeepSeekChatModel
from app.llm.fake import FakeChatModel

MODEL = "deepseek-chat"
API_KEY = "sk-test"
BASE_URL = "https://api.example.com/v1"


@pytest.fixture
def captured():
    """收集每次请求的 method/url/headers/json。"""
    box = {}

    def handler(request: httpx.Request) -> httpx.Response:
        box["request"] = request
        return box["response"](request)

    box["handler"] = handler
    return box


def make_model(captured, response: dict):
    transport = httpx.MockTransport(captured["handler"])
    captured["response"] = lambda _: httpx.Response(200, json=response)
    return DeepSeekChatModel(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL, client=httpx.Client(transport=transport)
    )


def test_sends_openai_format_payload(captured):
    model = make_model(
        captured,
        {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
    )
    out = model.invoke([SystemMessage(content="sys"), HumanMessage(content="hi")])
    assert out.content == "hello"

    req = captured["request"]
    assert req.url == "https://api.example.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == MODEL
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert "tools" not in body


def test_bind_tools_sends_tool_schema(captured):
    @tool
    def retrieve_documents(query: str) -> str:
        """Search the school's documents for information."""
        return ""

    model = make_model(
        captured,
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    ).bind_tools([retrieve_documents])
    model.invoke([HumanMessage(content="policy question")])

    body = json.loads(captured["request"].content)
    assert body["tool_choice"] == "auto"
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "retrieve_documents",
                "description": "Search the school's documents for information.",
                "parameters": retrieve_documents.args,
            },
        }
    ]


def test_tool_calls_response_parsed(captured):
    model = make_model(
        captured,
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "type": "function",
                                "function": {
                                    "name": "retrieve_documents",
                                    "arguments": '{"query": "bathroom pass"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )
    out = model.invoke([HumanMessage(content="policy question")])
    assert out.tool_calls[0]["name"] == "retrieve_documents"
    assert out.tool_calls[0]["args"] == {"query": "bathroom pass"}
    assert out.tool_calls[0]["id"] == "call_9"


def test_tool_message_converted_with_tool_call_id(captured):
    model = make_model(
        captured,
        {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
    )
    model.invoke(
        [
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "call_1"}]),
            ToolMessage(content="result text", tool_call_id="call_1"),
        ]
    )
    body = json.loads(captured["request"].content)
    assert body["messages"] == [
        {
            "role": "assistant",
            # OpenAI 协议：带 tool_calls 的 assistant 消息 content 必须为 null
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "result text", "tool_call_id": "call_1"},
    ]


def test_http_error_raises_clear_error(captured):
    captured["response"] = lambda _: httpx.Response(401, json={"error": "bad key"})
    transport = httpx.MockTransport(captured["handler"])
    model = DeepSeekChatModel(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL, client=httpx.Client(transport=transport)
    )
    with pytest.raises(RuntimeError, match="401"):
        model.invoke([HumanMessage(content="hi")])


# --- get_provider：env 配置解析 ---


def test_get_provider_deepseek_requires_all_env(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "settings",
        SimpleNamespace(llm_provider="deepseek", llm_model=None, llm_api_key=None, llm_base_url=None),
    )
    with pytest.raises(ValueError) as exc:
        llm_mod.get_provider()
    for name in ("SUBPILOT_LLM_MODEL", "SUBPILOT_LLM_API_KEY", "SUBPILOT_LLM_BASE_URL"):
        assert name in str(exc.value)


def test_get_provider_deepseek_lists_only_missing_vars(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "settings",
        SimpleNamespace(
            llm_provider="deepseek", llm_model="m", llm_api_key=None, llm_base_url=None
        ),
    )
    with pytest.raises(ValueError) as exc:
        llm_mod.get_provider()
    assert "SUBPILOT_LLM_MODEL" not in str(exc.value)
    assert "SUBPILOT_LLM_API_KEY" in str(exc.value)
    assert "SUBPILOT_LLM_BASE_URL" in str(exc.value)


def test_get_provider_fake_for_demo(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "settings",
        SimpleNamespace(llm_provider="fake", llm_model=None, llm_api_key=None, llm_base_url=None),
    )
    provider = llm_mod.get_provider()
    assert isinstance(provider, FakeChatModel)


def test_get_provider_unknown_raises(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "settings",
        SimpleNamespace(llm_provider="anthropic", llm_model=None, llm_api_key=None, llm_base_url=None),
    )
    with pytest.raises(ValueError, match="anthropic"):
        llm_mod.get_provider()
