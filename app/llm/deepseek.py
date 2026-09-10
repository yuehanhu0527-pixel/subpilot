"""DeepSeek-compatible provider：OpenAI 风格的 chat completions 适配。

- model / api_key / base_url 全部来自环境变量（config.settings），代码零硬编码。
- 通过 langchain-core 的 BaseChatModel 暴露，支持 bind_tools → OpenAI
  function-calling 格式，供 LangGraph 的 tool 循环使用。
- 测试注入 httpx.Client（MockTransport），不碰网络。
"""

import json

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool


def _to_openai_message(message: BaseMessage) -> dict:
    if message.type == "system":
        return {"role": "system", "content": message.content}
    if message.type == "human":
        return {"role": "user", "content": message.content}
    if message.type == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    if message.type == "ai":
        payload: dict = {"role": "assistant", "content": message.content or None}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in message.tool_calls
            ]
        return payload
    raise ValueError(f"Unsupported message type: {message.type}")


def _to_openai_tool(tool: BaseTool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args,
        },
    }


class DeepSeekChatModel(BaseChatModel):
    """OpenAI-compatible chat model（DeepSeek / 兼容端点）。

    model / api_key / base_url 必填；client 可注入（测试用 MockTransport）。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        client: httpx.Client | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client()
        self._tools: list[BaseTool] = []

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def bind_tools(self, tools, **kwargs):
        bound = self.__class__(
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
            client=self._client,
        )
        bound._tools = list(tools)
        return bound

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload: dict = {
            "model": self._model,
            "messages": [_to_openai_message(m) for m in messages],
        }
        if self._tools:
            payload["tools"] = [_to_openai_tool(t) for t in self._tools]
            payload["tool_choice"] = "auto"

        response = self._client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"DeepSeek API error {response.status_code}: {response.text[:500]}"
            )

        message = response.json()["choices"][0]["message"]
        content = message.get("content") or ""
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            tool_calls.append(
                {
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"] or "{}"),
                    "id": tc["id"],
                }
            )
        ai = AIMessage(content=content)
        if tool_calls:
            ai.tool_calls = tool_calls
        return ChatResult(generations=[ChatGeneration(message=ai)])
