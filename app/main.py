"""SubPilot — FastAPI entry point (Phase 0: minimal runnable skeleton)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="SubPilot", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict:
    """Liveness check used by the startup smoke test."""
    return {"status": "ok", "app": "subpilot"}


# IMPORTANT: API routers (/api/...) from later phases must be registered
# ABOVE this mount — anything registered after a mount at "/" gets shadowed
# by the static file handler.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
