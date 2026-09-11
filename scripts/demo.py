"""One-command Demo mode（Phase 7）：零 API key 跑通整个 SubPilot。

- 播种 data/schedule.json：本周剩余工作日 × 典型课表；
- 首次运行时把 sample_data/substitute-handbook.md 入库 RAG（真实 embedding，
  首次运行会下载 ~90MB 模型到 data/models）；
- 以 SUBPILOT_LLM_PROVIDER=demo 启动 uvicorn（确定性 provider，无外部调用）。

用法：
    python scripts/demo.py            # 播种 + 启动 http://127.0.0.1:8000
    python scripts/demo.py --seed-only   # 只播种（Docker 构建/启动用）
"""

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = Path(os.environ.get("SUBPILOT_DATA_DIR", ROOT / "data"))
HANDBOOK = ROOT / "sample_data" / "substitute-handbook.md"
TZ = "America/New_York"

PERIODS = [
    ("P1", "class", "08:00", "08:45", "Math"),
    ("P2", "class", "08:50", "09:35", "ELA"),
    ("P3", "class", "09:40", "10:25", "Science"),
    ("Lunch", "lunch", "11:30", "12:15", None),
    ("P4", "class", "12:20", "13:05", "Social Studies"),
    ("P5", "class", "13:10", "13:55", "Art"),
]


def _upcoming_weekdays(count: int = 5) -> list[date]:
    days, d = [], date.today()
    while len(days) < count:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


def seed_schedule() -> None:
    days = {
        d.isoformat(): [
            {k: v for k, v in zip(("name", "kind", "start", "end", "subject"), p) if v is not None}
            for p in PERIODS
        ]
        for d in _upcoming_weekdays()
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "schedule.json").write_text(json.dumps({"days": days}, indent=2))
    print(f"Seeded schedule.json with {len(days)} school days.")


def seed_documents() -> None:
    """把示例手册入库 RAG；已入库则跳过（避免重复 chunk）。"""
    from app.services.documents import DocumentRegistry

    registry = DocumentRegistry(DATA_DIR / "documents.json")
    if any(doc["name"] == HANDBOOK.name for doc in registry.list()):
        print("Handbook already ingested — skipping.")
        return
    from datetime import datetime, timezone

    from app.rag.pipeline import ingest_file

    chunks = ingest_file(HANDBOOK)
    registry.add(HANDBOOK.name, chunks, datetime.now(timezone.utc))
    print(f"Ingested {HANDBOOK.name} ({chunks} chunks).")


def main() -> None:
    seed_only = "--seed-only" in sys.argv
    seed_schedule()
    seed_documents()
    if seed_only:
        return
    env = {
        **os.environ,
        "SUBPILOT_LLM_PROVIDER": "demo",
        "SUBPILOT_TIMEZONE": TZ,
        "SUBPILOT_DATA_DIR": str(DATA_DIR),
    }
    print("Demo mode — open http://127.0.0.1:8000 and ask about the handbook, "
          'e.g. "What is the bathroom pass policy?"')
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
        cwd=ROOT, env=env, check=True,
    )


if __name__ == "__main__":
    main()
