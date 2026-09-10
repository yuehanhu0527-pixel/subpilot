"""最小 LangGraph agent loop。

START → contextualize → agent ⇄ tools(ToolNode) → END

- contextualize：确定性节点，把 ScheduleStatus 转成 SystemMessage 注入消息流；
  无 engine 或无 now 时跳过（schedule 文件不存在 ≠ 当天无课，前者不注入）。
- agent：模型 + 工具绑定，系统提示约束「无依据就说不知道」。
- tools：tool 列表通过参数注入（Phase 4 扩展点），tools_condition 自动路由。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..schedule.engine import ScheduleEngine
from .context import build_schedule_context
from .state import AgentState

SYSTEM_PROMPT = """You are SubPilot, an assistant that helps substitute teachers get through their day at an unfamiliar school.

Rules:
- For questions about school rules, procedures, or document contents, call retrieve_documents first and base your answer only on what it returns.
- If retrieve_documents returns "No relevant information found.", tell the user plainly that you don't know — never invent policy details.
- Schedule information provided in the conversation is authoritative; use it for time/period questions.
- Be concise and practical."""


def build_agent_graph(
    model: BaseChatModel,
    tools,
    schedule_engine: ScheduleEngine | None = None,
    system_prompt: str = SYSTEM_PROMPT,
):
    """构建 agent 图。tools 由参数注入，便于 Phase 4 扩展工具集。"""
    bound_model = model.bind_tools(tools)

    def contextualize(state: AgentState) -> dict:
        now = state.get("now")
        if now is None or schedule_engine is None:
            return {}
        status = schedule_engine.status_at(now)
        # Phase 4：注入当天完整课表（数据小、确定性，无需专门 schedule tool）
        periods = schedule_engine.periods_on(status.now.date())
        context = build_schedule_context(status, periods)
        return {"schedule_context": context, "messages": [SystemMessage(content=context)]}

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = bound_model.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("contextualize", contextualize)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "agent")
    graph.add_conditional_edges("agent", tools_condition)  # "tools" 或 END
    graph.add_edge("tools", "agent")
    return graph.compile()
