"""
LinguaLabel — SQLAlchemy Session & Database Management (Part 11)

Provides thread-safe session factories, engine caching, SQLite foreign-key enforcement,
and schema initialization routines.
"""

from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.config import DATABASE_URL
from db.models import Base

# Module-level engine and sessionmaker cache
_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker[Session]] = None


def get_engine(db_url: Optional[str] = None) -> Engine:
    """
    Returns or creates a cached SQLAlchemy engine.
    Applies SQLite-specific PRAGMA foreign_keys=ON when using SQLite.
    """
    global _ENGINE, _SESSION_FACTORY
    target_url = (db_url or DATABASE_URL).strip()

    if _ENGINE is not None and db_url is None:
        return _ENGINE

    connect_args = {}
    if target_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        target_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    # Enforce SQLite foreign key constraints
    if target_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    if db_url is None:
        _ENGINE = engine
        _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, class_=Session)

    return engine


def get_session_factory(db_url: Optional[str] = None) -> sessionmaker[Session]:
    """Returns a configured sessionmaker."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is not None and db_url is None:
        return _SESSION_FACTORY
    eng = get_engine(db_url)
    factory = sessionmaker(bind=eng, expire_on_commit=False, class_=Session)
    if db_url is None:
        _SESSION_FACTORY = factory
    return factory


@contextmanager
def get_db_session(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """Context manager providing a transactional database session."""
    factory = get_session_factory(db_url)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(db_url: Optional[str] = None) -> None:
    """Initializes tables in the target database using metadata."""
    eng = get_engine(db_url)
    Base.metadata.create_all(bind=eng)

