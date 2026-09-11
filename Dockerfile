# SubPilot — low-cost demo deployment (Render free tier / any Docker host).
# Embedding model is pre-downloaded at BUILD time so runtime startup is fast
# and never times out on a free-tier cold boot.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    SUBPILOT_DATA_DIR=/app/data \
    SUBPILOT_LLM_PROVIDER=demo \
    SUBPILOT_TIMEZONE=America/New_York

COPY pyproject.toml README.md ./
COPY app ./app
COPY sample_data ./sample_data
COPY scripts ./scripts

RUN pip install --no-cache-dir . && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/data/models')"

# Seed the schedule + handbook at container start (dates are relative to today),
# then serve. PORT comes from Render.
CMD ["sh", "-c", "python scripts/demo.py --seed-only && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
