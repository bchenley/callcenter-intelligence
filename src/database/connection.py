# callcenter-intelligence
# src/database/connection.py

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base

# Keyed by id(engine), not by URL: tests build several in-memory engines that all
# share the URL ":memory:" but are distinct databases. Caching by URL would hand
# a test the previous test's session factory.
_session_factories: dict[int, sessionmaker[Session]] = {}


def get_engine(db_path: str, encryption_key: str | None = None, echo: bool = False) -> Engine:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    url = "sqlite://" if db_path == ":memory:" else f"sqlite:///{db_path}"
    engine = create_engine(url, echo=echo, future=True)

    if encryption_key:
        # PRAGMA key must run on every new DBAPI connection, before any statement
        # on it, so it is attached to the connect event rather than called once.
        @event.listens_for(engine, "connect")
        def _set_key(dbapi_conn, _record) -> None:
            cur = dbapi_conn.cursor()
            cur.execute(f"PRAGMA key = '{encryption_key}'")
            cur.close()

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def get_session(engine: Engine) -> Session:
    factory = _session_factories.get(id(engine))
    if factory is None:
        factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        _session_factories[id(engine)] = factory
    return factory()


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Commit on success, roll back on any exception, always close. The rollback
    re-raises: a caller that swallowed it would persist a half-written call."""
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping(engine: Engine) -> bool:
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1")).scalar_one() == 1
