"""Runtime configuration. Secrets come from the environment only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    database_url: str
    data_dir: Path
    llm_provider: str
    llm_model: str
    openai_api_key: str
    openai_base_url: str
    ollama_base_url: str
    embedding_provider: str
    embedding_model: str
    reranker: str
    reranker_model: str
    prompt_version: str
    webhook_url: str
    smtp_host: str
    smtp_port: int
    smtp_from: str
    smtp_to: str
    tenderlens_base_url: str
    document_store: str
    s3_endpoint_url: str
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str
    heuristic_fallback: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(_env("DATA_DIR", "./data")).resolve()
        port = _env("SMTP_PORT", "587") or "587"
        return cls(
            database_url=_env("DATABASE_URL", "sqlite:///./data/bidready.db"),
            data_dir=data_dir,
            llm_provider=(_env("LLM_PROVIDER", "mock") or "mock").lower(),
            llm_model=_env("LLM_MODEL"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            embedding_provider=(_env("EMBEDDING_PROVIDER", "hash") or "hash").lower(),
            embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            reranker=(_env("RERANKER", "lexical") or "lexical").lower(),
            reranker_model=_env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            prompt_version=(_env("PROMPT_VERSION", "v1") or "v1").lower(),
            webhook_url=_env("WEBHOOK_URL"),
            smtp_host=_env("SMTP_HOST"),
            smtp_port=int(port),
            smtp_from=_env("SMTP_FROM"),
            smtp_to=_env("SMTP_TO"),
            tenderlens_base_url=_env("TENDERLENS_BASE_URL").rstrip("/"),
            document_store=(_env("DOCUMENT_STORE", "local") or "local").lower(),
            s3_endpoint_url=_env("S3_ENDPOINT_URL"),
            s3_bucket=_env("S3_BUCKET"),
            s3_access_key=_env("S3_ACCESS_KEY"),
            s3_secret_key=_env("S3_SECRET_KEY"),
            s3_region=_env("S3_REGION", "us-east-1"),
            heuristic_fallback=_env("HEURISTIC_FALLBACK", "1").lower() not in {"0", "false", "no"},
        )


def ensure_sqlite_parent(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    if ":memory:" in database_url:
        return
    raw = database_url.split("sqlite:///", 1)[-1]
    path = Path(raw)
    if str(path.parent) not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)
