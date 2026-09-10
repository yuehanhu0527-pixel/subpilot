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

    # "anthropic" (default) or "fake" (demo mode). Actually used from Phase 3 on.
    llm_provider: str = os.environ.get("SUBPILOT_LLM_PROVIDER", "anthropic")

    # Set in .env from Phase 3 onward; unset means demo mode.
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")

    # 本地数据目录（Phase 1 起用于 Chroma 与 embedding 模型缓存）
    data_dir: Path = field(default_factory=_default_data_dir)


settings = Settings()
