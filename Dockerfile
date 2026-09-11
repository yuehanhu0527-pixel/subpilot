# SubPilot — hosted demo deployment (Render free tier / any Docker host).
# Uses the lightweight lexical embedder: no torch / sentence-transformers,
# so the container stays well under the Render Free 512 MB limit.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    SUBPILOT_DATA_DIR=/app/data \
    SUBPILOT_LLM_PROVIDER=demo \
    SUBPILOT_EMBEDDER=lexical \
    SUBPILOT_TIMEZONE=America/New_York

COPY pyproject.toml README.md ./
COPY app ./app
COPY sample_data ./sample_data
COPY scripts ./scripts

RUN pip install --no-cache-dir .

# Seed the schedule + handbook at container start (dates are relative to today),
# then serve. PORT comes from Render.
CMD ["sh", "-c", "python scripts/demo.py --seed-only && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
