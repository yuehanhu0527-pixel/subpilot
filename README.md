# SubPilot

An AI agent that helps substitute teachers get through their day at an unfamiliar school.

**Status: Phase 2 — schedule engine.** The full README (architecture diagram, demo GIF, quickstart) lands in Phase 7.

## Schedule engine (Phase 2)

Deterministic, no-LLM period / bell schedule state (`app/schedule/`):

- **Models** — `Period` (controlled `kind`: `class` / `lunch` / `prep` / `other`), date-keyed `Schedule` (validated: sorted, no overlaps), `ScheduleStatus` result.
- **Loader** — reads structured course times from JSON (`{"days": {"2026-09-10": [...]}}`) with explicit validation errors.
- **Engine** — given an explicit timezone-aware `now`, reports one of `before_school` / `in_period` / `between_periods` / `after_school` / `no_classes`, with current subject, minutes remaining, and next period. Period starts are inclusive, ends exclusive; minutes round up.

Timezone is always explicit: constructor `tz=` first, then `SUBPILOT_TIMEZONE` — with neither set the engine raises instead of falling back to server-local time. The engine never calls `datetime.now()`; callers inject the current time.

```python
from app.schedule.engine import ScheduleEngine
from app.schedule.loader import load_schedule

schedule = load_schedule("data/schedule.json")
engine = ScheduleEngine(schedule, tz="America/New_York")   # or set SUBPILOT_TIMEZONE

status = engine.status_at(now)          # timezone-aware datetime
status.status                           # "in_period" | "between_periods" | ...
status.current_period.subject           # "Math" when in a class period
status.minutes_remaining                # e.g. 15
status.next_period.name                 # "Lunch" — what's coming up next
```

## RAG pipeline (Phase 1)

Local, offline retrieval over school documents (`app/rag/`):

- **Loader** — PDF (per page, via pypdf), DOCX (per paragraph, heading styles captured), TXT/Markdown. Scanned PDFs and empty files raise explicit errors.
- **Chunker** — rule-based paragraph aggregation (~1000 chars, ~150 overlap), captures Markdown/section headings.
- **Embedder** — local `sentence-transformers` model (`all-MiniLM-L6-v2`), cached under `data/models`.
- **Store** — local persistent Chroma collection (cosine space) under `data/chroma`.
- **Retriever** — dense vector search only, with a similarity threshold (0.30). Hits carry `filename / page / heading` citations; no relevant hits returns `"No relevant information found."` explicitly.

```python
from app.rag.pipeline import ingest_file, query

ingest_file("lesson_plan.pdf")                       # parse → chunk → embed → store
result = query("What is the bathroom pass policy?")  # dense retrieval with citations
result.chunks[0].citation()                          # e.g. "policy.pdf (p.3) · § Bathroom Passes"
result.message                                       # "No relevant information found." when nothing matched
```

Tests: `uv run pytest` (unit) and `uv run pytest -m integration` (end-to-end with the real embedding model; first run downloads the model).

## Quick start (Phase 2)

```bash
cd subpilot
uv sync                                   # installs dependencies into .venv/
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000> — you should see the SubPilot page skeleton.
<http://127.0.0.1:8000/healthz> should return `{"status":"ok"}`.
