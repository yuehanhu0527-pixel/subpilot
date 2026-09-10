"""FakeChatModel：脚本化确定性响应，仅用于测试与 Demo 模式。

⚠️ 生产逻辑（graph / tools / Agent）不得依赖本模块——
它只通过 get_provider() 在 SUBPILOT_LLM_PROVIDER=fake 时被选用，
或被测试直接 import。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """按预置脚本依次返回响应；每次调用记录收到的消息供断言。"""

    def __init__(self, responses: list[AIMessage] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._responses: list[AIMessage] = list(responses or [])
        self._recorded: list[list[BaseMessage]] = []
        self._bound_tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    @property
    def recorded(self) -> list[list[BaseMessage]]:
        """每次 _generate 收到的消息批次（按调用顺序）。"""
        return self._recorded

    @property
    def bound_tool_names(self) -> list[str]:
        return list(self._bound_tool_names)

    def bind_tools(self, tools, **kwargs):
        self._bound_tool_names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._recorded.append(list(messages))
        if not self._responses:
            raise RuntimeError("FakeChatModel script exhausted: no more scripted responses")
        return ChatResult(generations=[ChatGeneration(message=self._responses.pop(0))])
