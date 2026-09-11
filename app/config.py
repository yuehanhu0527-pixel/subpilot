"""SubPilot configuration.

Phase 0: only reads environment variables; no other logic yet.
Later phases build on this (provider selection, paths, demo mode).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(os.environ.get("SUBPILOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


@dataclass(frozen=True)
class Settings:
    app_name: str = "SubPilot"

    # LLM provider（Phase 3 起生效）: "deepseek"（默认，OpenAI-compatible）、
    # "demo"（确定性零 key，Phase 7）或 "fake"（脚本化演示/测试）。
    llm_provider: str = os.environ.get("SUBPILOT_LLM_PROVIDER", "deepseek")

    # DeepSeek-compatible 端点配置：全部来自环境变量，代码不写死；
    # provider=deepseek 时三者必须齐全，缺失由 get_provider() 明确报错。
    llm_model: str | None = os.environ.get("SUBPILOT_LLM_MODEL")
    llm_api_key: str | None = os.environ.get("SUBPILOT_LLM_API_KEY")
    llm_base_url: str | None = os.environ.get("SUBPILOT_LLM_BASE_URL")

    # 后续 Phase 使用（anthropic provider）；当前 Phase 3 未使用。
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")

    # 本地数据目录（Phase 1 起用于 Chroma 与 embedding 模型缓存）
    data_dir: Path = field(default_factory=_default_data_dir)

    # 日程引擎时区（Phase 2）：IANA 名称，如 "America/New_York"。
    # 未设置时 ScheduleEngine 必须显式传 tz，否则报错——绝不回退到服务器本地时区。
    timezone: str | None = os.environ.get("SUBPILOT_TIMEZONE")

    # RAG embedder（Phase 7）: "local"（默认，sentence-transformers + Chroma）
    # 或 "lexical"（轻量词法哈希，无 torch——Render Free 512MB hosted demo）。
    rag_embedder: str = os.environ.get("SUBPILOT_EMBEDDER", "local")


settings = Settings()
