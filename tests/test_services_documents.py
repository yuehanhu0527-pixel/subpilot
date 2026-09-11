"""documents.py：已入库文档登记表（JSON 文件，最小持久化）。"""

from datetime import datetime, timezone

from app.services.documents import DocumentRegistry

UTC = timezone.utc


def _dt(hour):
    return datetime(2026, 9, 10, hour, tzinfo=UTC)


def test_list_empty_when_missing(tmp_path):
    assert DocumentRegistry(tmp_path / "documents.json").list() == []


def test_add_and_list_roundtrip(tmp_path):
    reg = DocumentRegistry(tmp_path / "documents.json")
    reg.add("policy.pdf", 3, _dt(9))
    reg.add("lesson.docx", 5, _dt(10))
    docs = reg.list()
    assert [d["name"] for d in docs] == ["policy.pdf", "lesson.docx"]
    assert docs[0]["chunks"] == 3
    assert docs[0]["ingested_at"] == _dt(9).isoformat()


def test_add_same_name_replaces_entry(tmp_path):
    reg = DocumentRegistry(tmp_path / "documents.json")
    reg.add("policy.pdf", 3, _dt(9))
    reg.add("policy.pdf", 7, _dt(11))
    docs = reg.list()
    assert len(docs) == 1
    assert docs[0]["chunks"] == 7


def test_registry_persists_across_instances(tmp_path):
    path = tmp_path / "documents.json"
    DocumentRegistry(path).add("a.txt", 1, _dt(9))
    assert DocumentRegistry(path).list()[0]["name"] == "a.txt"
