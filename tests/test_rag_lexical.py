"""LexicalEmbedder：hosted demo 的轻量词法检索（无 torch）行为测试。"""

import math
from pathlib import Path
from types import SimpleNamespace

import app.rag.pipeline as pipeline_mod
from app.rag.lexical import LexicalEmbedder
from app.rag.store import ChromaStore

HANDBOOK = Path(__file__).resolve().parent.parent / "sample_data" / "substitute-handbook.md"


def test_encode_shape_and_normalization():
    embedder = LexicalEmbedder()
    vectors = embedder.encode(["bathroom pass policy", "fire drill"])
    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 256
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=1e-6)


def test_similar_queries_retrieve_handbook_sections(store, collection):
    """真实链路：loader → chunker → lexical 向量 → Chroma → retriever。"""
    embedder = LexicalEmbedder()
    chunks = pipeline_mod.ingest_file(HANDBOOK, collection_name=collection, store=store, embedder=embedder)
    assert chunks >= 1

    for question in (
        "What is the bathroom pass policy?",
        "What does the handbook say about fire drills?",
    ):
        result = pipeline_mod.query(question, collection_name=collection, store=store, embedder=embedder)
        assert result.found, question
        assert result.chunks[0].source_name == "substitute-handbook.md"
        assert "substitute-handbook.md" in result.chunks[0].citation()


def test_no_relevant_query_returns_no_hits(store, collection):
    embedder = LexicalEmbedder()
    pipeline_mod.ingest_file(HANDBOOK, collection_name=collection, store=store, embedder=embedder)
    result = pipeline_mod.query(
        "quantum chromodynamics", collection_name=collection, store=store, embedder=embedder
    )
    assert not result.found


def test_get_embedder_respects_env(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_mod, "_embedder", None)
    monkeypatch.setattr(
        pipeline_mod, "settings", SimpleNamespace(rag_embedder="lexical", data_dir=tmp_path)
    )
    assert isinstance(pipeline_mod.get_embedder(), LexicalEmbedder)
    monkeypatch.setattr(pipeline_mod, "_embedder", None)  # 单例复位，避免串扰其他测试


def test_unknown_embedder_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_mod, "_embedder", None)
    monkeypatch.setattr(
        pipeline_mod, "settings", SimpleNamespace(rag_embedder="nope", data_dir=tmp_path)
    )
    try:
        pipeline_mod.get_embedder()
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "nope" in str(exc)
    finally:
        monkeypatch.setattr(pipeline_mod, "_embedder", None)
