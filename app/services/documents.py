"""已入库文档登记表：JSON 文件最小持久化（Phase 5 文档面板用）。

记录 (文件名, 入库 chunk 数, 入库时间)；同名文件再次入库时替换条目。
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class DocumentRegistry:
    def __init__(self, path: str | Path):
        self._path = Path(path)

    def _read(self) -> dict:
        if not self._path.exists():
            return {"documents": []}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        return list(self._read()["documents"])

    def add(self, name: str, chunks: int, ingested_at: datetime) -> None:
        data = self._read()
        entry = {
            "name": name,
            "chunks": chunks,
            "ingested_at": ingested_at.astimezone(timezone.utc).isoformat(),
        }
        data["documents"] = [d for d in data["documents"] if d["name"] != name]
        data["documents"].append(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
