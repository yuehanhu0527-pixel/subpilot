"""Demo 数据播种（Phase 7）：schedule.json + 示例手册入库 RAG。

供两处复用（不重复实现）：
- scripts/demo.py：本地一键演示（播种 + 启动 server）；
- app/main.py lifespan：SUBPILOT_LLM_PROVIDER=demo 时启动即自动播种，
  hosted demo（Render）无需手动运行任何脚本。

seed_documents 幂等：已入库（documents.json 有记录）则跳过。
"""

import json
from datetime import date, timedelta
from pathlib import Path

from .config import settings

HANDBOOK = Path(__file__).resolve().parent.parent / "sample_data" / "substitute-handbook.md"

PERIODS = [
    ("P1", "class", "08:00", "08:45", "Math"),
    ("P2", "class", "08:50", "09:35", "ELA"),
    ("P3", "class", "09:40", "10:25", "Science"),
    ("Lunch", "lunch", "11:30", "12:15", None),
    ("P4", "class", "12:20", "13:05", "Social Studies"),
    ("P5", "class", "13:10", "13:55", "Art"),
]


def _upcoming_weekdays(count: int = 5) -> list[date]:
    """含今天在内的最近 count 个工作日——Now 面板当天即有课表可展示。"""
    days, d = [], date.today()
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def seed_schedule() -> None:
    """把未来 5 个学日的典型课表写入 data/schedule.json。"""
    days = {
        d.isoformat(): [
            {k: v for k, v in zip(("name", "kind", "start", "end", "subject"), p) if v is not None}
            for p in PERIODS
        ]
        for d in _upcoming_weekdays()
    }
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "schedule.json").write_text(json.dumps({"days": days}, indent=2))
    print(f"Seeded schedule.json with {len(days)} school days.")


def seed_documents() -> None:
    """把示例手册入库 RAG；已入库则跳过（避免重复 chunk）。"""
    from datetime import datetime, timezone

    from .rag.pipeline import ingest_file
    from .services.documents import DocumentRegistry

    registry = DocumentRegistry(settings.data_dir / "documents.json")
    if any(doc["name"] == HANDBOOK.name for doc in registry.list()):
        print("Handbook already ingested — skipping.")
        return
    chunks = ingest_file(HANDBOOK)
    registry.add(HANDBOOK.name, chunks, datetime.now(timezone.utc))
    print(f"Ingested {HANDBOOK.name} ({chunks} chunks).")
