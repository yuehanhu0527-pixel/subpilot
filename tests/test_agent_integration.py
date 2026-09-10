"""真实 DeepSeek-compatible API 的集成测试。

需要网络 + SUBPILOT_LLM_MODEL / SUBPILOT_LLM_API_KEY / SUBPILOT_LLM_BASE_URL；
无 key 自动 skip。运行：pytest -m integration
"""

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            os.environ.get("SUBPILOT_LLM_API_KEY")
            and os.environ.get("SUBPILOT_LLM_MODEL")
            and os.environ.get("SUBPILOT_LLM_BASE_URL")
        ),
        reason="SUBPILOT_LLM_MODEL / _API_KEY / _BASE_URL not set",
    ),
]


def test_real_provider_answers_simple_question():
    from app.agent import Agent
    from app.llm import get_provider

    agent = Agent(model=get_provider(), tools=[])
    answer = agent.run("integration-test", "What is 2+2? Answer with just the number.")
    assert answer.strip()
