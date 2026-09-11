"""DemoChatModel：确定性、零 API key 的演示 provider（Phase 7）。

规则（完全确定，不调用任何外部服务）：
- 本批最后一条是 HumanMessage → 发起 retrieve_documents 工具调用（query = 用户原文）；
- 本批存在 ToolMessage → 基于检索结果作答：
  有命中 → 引用第一条 chunk 的原文与 citation；
  无命中 → 明确说明文档没有覆盖，绝不编造。
供 README 的 Demo mode 使用；生产逻辑不依赖它（与 fake 同级）。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DEMO_NOTE = "(Demo mode — deterministic provider, no API key.)"

_NO_HITS = "No relevant information found."


class DemoChatModel(BaseChatModel):
    """把每个问题变成一次真实检索 + 引用回答，展示完整 RAG 回路。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bound_tool_names: list[str] = []
        self._call_count = 0

    @property
    def _llm_type(self) -> str:
        return "demo"

    @property
    def bound_tool_names(self) -> list[str]:
        return list(self._bound_tool_names)

    def bind_tools(self, tools, **kwargs):
        self._bound_tool_names = [t.name for t in tools]
        return self

    @staticmethod
    def _answer_from_retrieval(tool_message: ToolMessage) -> str:
        content = tool_message.content.strip()
        if content == _NO_HITS:
            return (
                "The uploaded documents don't cover that question — I won't guess. "
                f"Try uploading a document or rephrasing. {DEMO_NOTE}"
            )
        # 首个 chunk：第一行 "Source: …" 之后到空行前为原文
        lines = content.splitlines()
        citation = lines[0].removeprefix("Source: ").strip()
        body = "\n".join(line for line in lines[1:] if line.strip())
        return (
            f"Found in the uploaded documents:\n\n{body}\n\nSource: {citation} {DEMO_NOTE}"
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._call_count += 1
        # 消息流顺序为 [..., Human, System(context)]（contextualize 节点追加），
        # 因此按「最后一条 Human 是否晚于最后一条 Tool」判断新问题/回答时机。
        last_human = last_tool = None
        for i, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                last_human = (i, message)
            elif isinstance(message, ToolMessage):
                last_tool = (i, message)

        if (
            last_human is not None
            and "retrieve_documents" in self._bound_tool_names
            and (last_tool is None or last_human[0] > last_tool[0])
        ):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="",
                tool_calls=[{
                    "name": "retrieve_documents",
                    "args": {"query": last_human[1].content},
                    "id": f"demo-{self._call_count}",
                }],
            ))])

        if last_tool is not None:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=self._answer_from_retrieval(last_tool[1])
            ))])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=f"I can search the uploaded documents for that. {DEMO_NOTE}"
        ))])
