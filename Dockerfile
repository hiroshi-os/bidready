FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    LLM_PROVIDER=mock \
    EMBEDDING_PROVIDER=hash \
    RERANKER=lexical \
    DATA_DIR=/app/data \
    DATABASE_URL=postgresql+psycopg://bidready:bidready@db:5432/bidready

EXPOSE 8000
CMD ["uvicorn", "bidready.main:app", "--host", "0.0.0.0", "--port", "8000"]
