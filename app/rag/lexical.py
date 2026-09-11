"""轻量词法 embedder（Phase 7 Render 修复）：特征哈希词袋，零 torch。

用途：hosted Demo mode 在 Render Free 512MB 下运行——加载
sentence-transformers / PyTorch 会 OOM，而词法向量对演示语料
（数百词的英文手册）足够做真实检索：同样的 chunker / loader /
Chroma store / retriever / citation 链路，只是向量由哈希生成。

本地 / Live 模式（默认）仍使用 LocalEmbedder（sentence-transformers），
绝不降级——只有显式 SUBPILOT_EMBEDDER=lexical 才启用本实现。
"""

import hashlib
import math
import re

DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9']+")

# 检索信号极弱的常见停用词（demo 语料为英文，仅按需覆盖）
_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "its", "of", "on", "or",
    "say", "says", "that", "the", "their", "they", "this", "to", "what",
    "when", "where", "which", "who", "will", "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        # 朴素单数化（drills → drill），仅去掉普通 s 词尾
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class LexicalEmbedder:
    """token → 有符号特征哈希 → L2 归一化向量（与 Chroma 余弦空间对齐）。"""

    # 词法空间的检索阈值：词法余弦随 chunk 长度衰减（短查询 × 长 chunk），
    # 需比 dense 空间的 0.30 略低；pipeline.query 据此构造 Retriever。
    threshold = 0.25

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * DIM
            for tok in _tokens(text):
                digest = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:2], "big") % DIM
                sign = 1.0 if digest[2] & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(x * x for x in vec))
            if norm:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors
