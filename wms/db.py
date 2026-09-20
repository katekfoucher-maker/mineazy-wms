"""SQLAlchemy engine, session factory and declarative base."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

from wms.config import get_settings

settings = get_settings()
_URL = settings.resolved_database_url

connect_args: dict = {}
engine_kwargs: dict = {}
if _URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    engine_kwargs = {"pool_size": 10, "max_overflow": 20, "pool_recycle": 1800}
    # PyMySQL has no default read/write timeout - a connection that stalls
    # mid-query (a dropped network path, a host-side hiccup) hangs forever
    # instead of raising, which pool_pre_ping can't catch either (the ping
    # query itself would hang the same way). Bounded timeouts turn that into
    # an ordinary retriable error instead of a wedged request/process.
    connect_args = {"connect_timeout": 10, "read_timeout": 30, "write_timeout": 30}
    if settings.db_ssl_ca:
        connect_args["ssl"] = {"ca": settings.db_ssl_ca}

engine = create_engine(_URL, echo=settings.debug, pool_pre_ping=True,
                       connect_args=connect_args, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


class TimestampedBase(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


def get_session():
    """FastAPI dependency / context helper - yields a session, always closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from wms import models  # noqa: F401  (register tables)
    Base.metadata.create_all(bind=engine)
    _apply_light_migrations()


# Columns added after the first release.  create_all() never ALTERs an existing
# table, so add them here (idempotent) instead of forcing a full re-seed.
_ADDED_COLUMNS = {
    "back_orders": {"cycle": "VARCHAR(12) NOT NULL DEFAULT 'WEEKLY'",
                    "period_start": "DATE"},
    "sales_records": {"source_ref": "VARCHAR(50)"},
    "weekly_sales_lines": {"is_simulated": "BOOLEAN NOT NULL DEFAULT 0"},
}

# one-off value fixes after create_all (idempotent)
_DATA_FIXES = [
    "UPDATE back_orders SET stage='OPEN' WHERE stage IN ('SUBMITTED','DISPATCHED')",
    "UPDATE back_orders SET stage='CLOSED', status='CLOSED' WHERE stage='CANCELLED' OR status='CANCELLED'",
]


def _apply_light_migrations() -> None:
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
        if insp.has_table("back_orders"):
            for sql in _DATA_FIXES:
                conn.execute(text(sql))
