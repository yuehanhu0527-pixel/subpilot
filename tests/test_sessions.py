"""sessions.py：最小 session state（内存实现 + 清晰接口）。"""

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.sessions import InMemorySessionStore


def test_unknown_session_returns_empty():
    store = InMemorySessionStore()
    assert store.get("nope") == []


def test_append_and_get_roundtrip():
    store = InMemorySessionStore()
    store.append("s1", [HumanMessage(content="hi")])
    store.append("s1", [AIMessage(content="hello")])
    history = store.get("s1")
    assert [m.content for m in history] == ["hi", "hello"]


def test_sessions_are_isolated():
    store = InMemorySessionStore()
    store.append("s1", [HumanMessage(content="one")])
    store.append("s2", [HumanMessage(content="two")])
    assert [m.content for m in store.get("s1")] == ["one"]
    assert [m.content for m in store.get("s2")] == ["two"]


def test_clear_removes_session():
    store = InMemorySessionStore()
    store.append("s1", [HumanMessage(content="hi")])
    store.clear("s1")
    assert store.get("s1") == []
    store.clear("s1")  # 幂等


def test_store_keeps_caller_lists_independent():
    store = InMemorySessionStore()
    messages = [HumanMessage(content="hi")]
    store.append("s1", messages)
    messages.clear()
    assert [m.content for m in store.get("s1")] == ["hi"]
