"""DemoChatModel：确定性、零 API key 的演示 provider（Phase 7）。

规则（完全确定，不调用任何外部服务）：
- 本批存在新的人类问题（最后一条 Human 晚于最后一条 Tool）→ 发起
  retrieve_documents 工具调用（query = 用户原文）；
- 本批存在 ToolMessage → 基于检索结果作答：chunk 按「空行分段 → 句末标点切句」
  拆成句子单元，按 IDF 加权的 query token overlap 打分，只引用最相关的
  1–3 句（可来自前两个命中 chunk），保留原 citation。绝不整块贴出 chunk。
- 无命中 → 明确说明文档没有覆盖，绝不编造。
供 README 的 Demo mode 使用；生产逻辑不依赖它（与 fake 同级）。
"""

import math
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..rag.lexical import _tokens

DEMO_NOTE = "(Demo mode — deterministic provider, no API key.)"

_NO_HITS = "No relevant information found."

_SOURCE_RE = re.compile(r"^Source: (.*)$")
_HEADING_RE = re.compile(r"^#{1,6}\s")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _parse_blocks(content: str) -> list[tuple[str, str]]:
    """把工具结果按 "Source: …" 行切成 [(citation, text)]。"""
    blocks: list[tuple[str, str]] = []
    citation: str | None = None
    lines: list[str] = []
    for line in content.splitlines():
        match = _SOURCE_RE.match(line.strip())
        if match:
            if citation is not None:
                blocks.append((citation, "\n".join(lines)))
            citation, lines = match.group(1), []
        elif citation is not None:
            lines.append(line)
    if citation is not None:
        blocks.append((citation, "\n".join(lines)))
    return blocks


def _terminated(unit: str) -> bool:
    return unit[-1:] in ".!?"


def _units(body: str) -> list[str]:
    """把 chunk 文本切成可引用的句子单元。

    chunker 会把段落按行宽切成以空行分隔的碎片（可能切在句子中间），
    因此先按标题行分段，段内把所有换行并成空格，再按句末标点切句；
    开头是小写的碎片并回上一句（修复 "10 minutes." 之类误切）。
    标题行独立成单元；段落末无句末标点的残句保留为未终止单元。
    """
    units: list[str] = []
    segment: list[str] = []

    def flush() -> None:
        if not segment:
            return
        text = re.sub(r"\s+", " ", " ".join(segment)).strip()
        for sent in _SENT_SPLIT_RE.split(text):
            sent = sent.strip()
            if not sent:
                continue
            if units and sent[:1].islower():
                units[-1] = units[-1] + " " + sent
            else:
                units.append(sent)
        segment.clear()

    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _HEADING_RE.match(line):
            flush()
            units.append(line)
        else:
            segment.append(line)
    flush()
    return units


def _idf(token: str, all_unit_tokens: list[set[str]]) -> float:
    """token 在全部候选句中的 IDF（低频词更值钱）。"""
    n = len(all_unit_tokens)
    return math.log(1 + n / (1 + sum(1 for toks in all_unit_tokens if token in toks)))


def _answer_from_retrieval(query: str, content: str) -> str:
    text = content.strip()
    if text == _NO_HITS:
        return (
            "The uploaded documents don't cover that question — I won't guess. "
            f"Try uploading a document or rephrasing. {DEMO_NOTE}"
        )
    blocks = _parse_blocks(text)
    if not blocks:
        return f"The documents returned an empty result. {DEMO_NOTE}"

    # 收集前两个命中的句子单元；若首块末句被 chunk 边界切开、
    # 且第二块首句以小写开头（续写），先拼接再打分。
    units_by_block = [_units(body) for _, body in blocks[:2]]
    if len(units_by_block) == 2:
        first, second = units_by_block
        if first and not _terminated(first[-1]) and second and second[0][:1].islower():
            first[-1] = first[-1] + " " + second[0]
            second.pop(0)

    all_units: list[tuple[int, str, str]] = [
        (bidx, blocks[bidx][0], unit)
        for bidx in range(len(units_by_block))
        for unit in units_by_block[bidx]
    ]
    all_unit_tokens = [set(_tokens(u)) for _, _, u in all_units]

    query_tokens = set(_tokens(query))
    scored = []
    for i, ((bidx, citation, unit), toks) in enumerate(zip(all_units, all_unit_tokens)):
        if unit.startswith("#") or not _terminated(unit):
            continue
        overlap = query_tokens & toks
        if not overlap:
            continue
        # 排序：命中 token 数 → IDF 和 → 靠后的文档序（打破 TOC 句与正文句的平局）
        scored.append(
            (len(overlap), sum(_idf(t, all_unit_tokens) for t in overlap), i, unit, citation)
        )
    scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
    selected = [(unit, citation) for _, _, _, unit, citation in scored[:3]]

    if not selected:
        # 兜底：引用首块前两句（保证回答总有出处）
        citation, body = blocks[0]
        units = [u for u in _units(body) if not u.startswith("#")] or _units(body)
        selected = [(u, citation) for u in units[:2]]

    body = " ".join(unit for unit, _ in selected)
    citations = " · ".join(dict.fromkeys(c for _, c in selected))
    return f"Found in the uploaded documents:\n\n{body}\n\nSource: {citations} {DEMO_NOTE}"


class DemoChatModel(BaseChatModel):
    """把每个问题变成一次真实检索 + 聚焦引用，展示完整 RAG 回路。"""

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
            query = last_human[1].content if last_human is not None else ""
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=_answer_from_retrieval(query, last_tool[1].content)
            ))])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=f"I can search the uploaded documents for that. {DEMO_NOTE}"
        ))])
