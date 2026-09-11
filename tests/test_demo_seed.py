"""app/demo_seed：hosted demo 启动自动播种（schedule + 示例手册）的测试。"""

import asyncio
import json
from types import SimpleNamespace

import app.demo_seed as seed_mod
import app.main as main_mod
from app.rag import pipeline as pipeline_mod
from app.rag.store import ChromaStore
from app.services.documents import DocumentRegistry
from tests.helpers import FakeEmbedder

VOCAB = {"bathroom": [1, 0, 0, 0, 0, 0]}


def _use_tmp_settings(tmp_path, monkeypatch, modules):
    for mod in modules:
        monkeypatch.setattr(mod, "settings", SimpleNamespace(data_dir=tmp_path))


def test_seed_schedule_writes_five_weekdays(tmp_path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch, [seed_mod])
    seed_mod.seed_schedule()
    data = json.loads((tmp_path / "schedule.json").read_text())
    assert len(data["days"]) == 5
    periods = next(iter(data["days"].values()))
    assert [p["name"] for p in periods] == ["P1", "P2", "P3", "Lunch", "P4", "P5"]


def test_seed_documents_ingests_once(tmp_path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch, [seed_mod])
    monkeypatch.setattr(pipeline_mod, "_store", ChromaStore.ephemeral())
    monkeypatch.setattr(pipeline_mod, "_embedder", FakeEmbedder(VOCAB, dim=6))
    seed_mod.seed_documents()
    first = DocumentRegistry(tmp_path / "documents.json").list()
    assert len(first) == 1 and first[0]["chunks"] >= 1
    seed_mod.seed_documents()  # 幂等：已入库则跳过
    assert len(DocumentRegistry(tmp_path / "documents.json").list()) == 1


def test_lifespan_seeds_when_demo_provider(tmp_path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch, [seed_mod])
    monkeypatch.setattr(
        main_mod, "settings", SimpleNamespace(llm_provider="demo", data_dir=tmp_path)
    )
    monkeypatch.setattr(pipeline_mod, "_store", ChromaStore.ephemeral())
    monkeypatch.setattr(pipeline_mod, "_embedder", FakeEmbedder(VOCAB, dim=6))

    async def run():
        async with main_mod.lifespan(main_mod.app):
            pass

    asyncio.run(run())
    assert (tmp_path / "schedule.json").exists()
    assert len(DocumentRegistry(tmp_path / "documents.json").list()) == 1


def test_lifespan_does_not_seed_otherwise(tmp_path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch, [seed_mod])
    monkeypatch.setattr(
        main_mod, "settings", SimpleNamespace(llm_provider="deepseek", data_dir=tmp_path)
    )

    async def run():
        async with main_mod.lifespan(main_mod.app):
            pass

    asyncio.run(run())
    assert not (tmp_path / "schedule.json").exists()
