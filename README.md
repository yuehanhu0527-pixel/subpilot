# SubPilot

An AI agent that helps substitute teachers get through their day at an unfamiliar school.

**Status: Phase 0 — project scaffold.** The full README (architecture diagram, demo GIF, quickstart) lands in Phase 7.

## Quick start (Phase 0)

```bash
cd subpilot
uv sync                                   # installs dependencies into .venv/
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000> — you should see the SubPilot page skeleton.
<http://127.0.0.1:8000/healthz> should return `{"status":"ok"}`.
