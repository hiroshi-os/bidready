"""Engine and session factory. sqlite is for tests; docker-compose uses Postgres."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bidready.config import Settings, ensure_sqlite_parent
from bidready.models import Base


def make_engine(settings: Settings):
    ensure_sqlite_parent(settings.database_url)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    url = settings.database_url
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(settings: Settings) -> sessionmaker[Session]:
    return sessionmaker(bind=make_engine(settings), expire_on_commit=False)
