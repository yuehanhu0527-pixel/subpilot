# SubPilot

An AI agent that helps substitute teachers get through their day at an unfamiliar school.

**Status: Phase 1 — local RAG pipeline.** The full README (architecture diagram, demo GIF, quickstart) lands in Phase 7.

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

## Quick start (Phase 1)

```bash
cd subpilot
uv sync                                   # installs dependencies into .venv/
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000> — you should see the SubPilot page skeleton.
<http://127.0.0.1:8000/healthz> should return `{"status":"ok"}`.
