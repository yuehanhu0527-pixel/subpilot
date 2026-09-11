"""SubPilot — FastAPI entry point (Phase 5: API + static UI)."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router as api_router
from .config import settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Hosted demo：demo provider 下启动即自动播种 schedule + 示例手册，
    # 不依赖手动运行 scripts/demo.py（seed_documents 幂等）。
    if settings.llm_provider == "demo":
        from .demo_seed import seed_documents, seed_schedule

        seed_schedule()
        seed_documents()
    yield


app = FastAPI(title="SubPilot", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    """Liveness check used by the startup smoke test."""
    return {"status": "ok", "app": "subpilot"}


# IMPORTANT: API routers (/api/...) must be registered ABOVE this mount —
# anything registered after a mount at "/" gets shadowed by the static handler.
app.include_router(api_router)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
