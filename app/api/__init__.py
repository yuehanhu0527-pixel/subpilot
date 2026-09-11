"""Phase 5 最小 API：Web UI 与既有 Agent / schedule / tracker 之间的薄适配层。

原则：
- 不修改 Agent 核心架构，只做惰性组装与调用。
- 依赖均可 monkeypatch（_get_agent / _load_engine / pipeline 单例）。
- 配置缺失时返回 503 与明确错误，不静默降级。
"""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from ..agent import Agent, load_schedule_engine
from ..config import settings
from ..rag.pipeline import ingest_file
from ..services.documents import DocumentRegistry
from ..services.tracker import (
    BathroomPassRepo,
    ClassroomEventRepo,
    local_hhmm,
    school_day_fn,
)

router = APIRouter(prefix="/api")

_agent: Agent | None = None
_load_engine = load_schedule_engine  # 测试可替换


def _get_agent() -> Agent:
    """惰性单例；LLM/tz 配置缺失时抛出明确 ValueError。"""
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def _tracker_path() -> Path:
    return settings.data_dir / "subpilot.db"


def _registry() -> DocumentRegistry:
    return DocumentRegistry(settings.data_dir / "documents.json")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChatRequest(BaseModel):
    session_id: str
    text: str


class ReportRequest(BaseModel):
    session_id: str


def _period_dict(period) -> dict | None:
    if period is None:
        return None
    return {
        "name": period.name,
        "kind": period.kind,
        "subject": period.subject,
        "start": period.start.strftime("%H:%M"),
        "end": period.end.strftime("%H:%M"),
    }


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    """一轮 Agent 对话；返回最终回答 + 本轮 RAG 来源引用。"""
    try:
        agent = _get_agent()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    try:
        turn = agent.run_turn(req.session_id, req.text, now=_now())
    except Exception as exc:  # LLM/tool 运行时故障：明确 502，不裸抛 500
        raise HTTPException(status_code=502, detail=f"Agent call failed: {exc}")
    return {"answer": turn.answer, "sources": list(turn.sources)}


@router.get("/status")
def status() -> dict:
    """NOW / Bathroom / Events / Documents 四个面板的数据。"""
    now = _now()
    notes: list[str] = []

    schedule_payload = None
    try:
        engine = _load_engine()
        if engine is None:
            notes.append(
                "No schedule file (data/schedule.json) — the Now panel stays quiet until you add one."
            )
        else:
            current = engine.status_at(now)
            schedule_payload = {
                "status": current.status,
                "current_period": _period_dict(current.current_period),
                "minutes_remaining": current.minutes_remaining,
                "next_period": _period_dict(current.next_period),
                "minutes_until_next": current.minutes_until_next,
                "periods_today": [
                    _period_dict(p) for p in engine.periods_on(current.now.date())
                ],
            }
    except ValueError as exc:
        notes.append(str(exc))

    bathroom: list[dict] = []
    events: list[dict] = []
    tz_name = settings.timezone
    try:
        day = school_day_fn(tz_name)(now)
    except ValueError as exc:
        day = None
        notes.append(str(exc))
    if day is not None:
        bathroom = [
            {
                "student": p["student"],
                "minutes_out": p["minutes_out"],
                "left_at": local_hhmm(p["departed_at"], tz_name),
            }
            for p in BathroomPassRepo(_tracker_path()).open_passes(now)
        ]
        events = [
            {
                "time": local_hhmm(e["recorded_at"], tz_name),
                "period": e["period"],
                "description": e["description"],
                "students": e["students"],
            }
            for e in ClassroomEventRepo(_tracker_path()).events_on(day)
        ]

    return {
        "now": now.isoformat(),
        "schedule": schedule_payload,
        "bathroom": bathroom,
        "events": events,
        "documents": _registry().list(),
        "notes": notes,
    }


@router.get("/documents")
def documents() -> list:
    return _registry().list()


@router.post("/documents")
async def upload_documents(file: UploadFile) -> dict:
    """上传 PDF / DOCX / TXT，入库 RAG 并登记。"""
    name = Path(file.filename or "document").name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name")
    dest = settings.data_dir / "documents" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    try:
        chunks = ingest_file(dest)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _registry().add(name, chunks, _now())
    return {"name": name, "chunks": chunks}


@router.post("/report")
def report(req: ReportRequest) -> dict:
    """生成 end-of-day substitute note（仅基于真实记录，提示词约束）。"""
    try:
        agent = _get_agent()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    prompt = (
        "Write the end-of-day substitute note for the regular teacher. "
        "Call list_today_events and base the note only on those events and the "
        "schedule context. If no events were logged, say so plainly — "
        "never invent events."
    )
    try:
        turn = agent.run_turn(req.session_id, prompt, now=_now())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent call failed: {exc}")
    return {"report": turn.answer}
