"""本地 embedding：sentence-transformers。

选本地免费模型的意义：离线可用、零 API 成本，且与 LLM 供应商完全解耦。
"""

MODEL_NAME = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """惰性加载 sentence-transformers 模型，把文本编码为归一化向量。"""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str | None = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            # 延迟导入：sentence-transformers 很重，只在真正需要时加载
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, cache_folder=self._cache_dir)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """返回与输入等长的向量列表（L2 归一化，适配 Chroma 余弦空间）。"""
        self._ensure_model()
        return self._model.encode(texts, normalize_embeddings=True).tolist()
