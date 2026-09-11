"""LLM provider abstraction（Phase 3）。

get_provider() 依据 SUBPILOT_LLM_PROVIDER 返回 provider：
- "deepseek"：OpenAI-compatible 端点，model/api_key/base_url 必须来自环境变量
  （SUBPILOT_LLM_MODEL / SUBPILOT_LLM_API_KEY / SUBPILOT_LLM_BASE_URL），缺失即报错。
- "demo"：确定性 provider，零 API key（Phase 7 Demo mode），生产逻辑不依赖它。
- "fake"：脚本化演示/测试 provider，生产逻辑不依赖它。
"""

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import settings


def get_provider() -> BaseChatModel:
    """按配置返回 chat model；配置缺失时明确报错，绝不静默回退。"""
    if settings.llm_provider == "deepseek":
        values = {
            "SUBPILOT_LLM_MODEL": settings.llm_model,
            "SUBPILOT_LLM_API_KEY": settings.llm_api_key,
            "SUBPILOT_LLM_BASE_URL": settings.llm_base_url,
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                f"Missing environment variable(s): {', '.join(missing)} "
                f"(required when SUBPILOT_LLM_PROVIDER=deepseek)"
            )
        from .deepseek import DeepSeekChatModel

        return DeepSeekChatModel(
            model=values["SUBPILOT_LLM_MODEL"],
            api_key=values["SUBPILOT_LLM_API_KEY"],
            base_url=values["SUBPILOT_LLM_BASE_URL"],
        )
    if settings.llm_provider == "demo":
        from .demo import DemoChatModel  # 惰性导入：仅 Demo mode

        return DemoChatModel()
    if settings.llm_provider == "fake":
        from .fake import FakeChatModel  # 惰性导入：仅演示/测试

        return FakeChatModel()
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
