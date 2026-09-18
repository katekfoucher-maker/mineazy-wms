"""Shared FastAPI dependencies.

There is no authentication in this system (not in scope). The acting user is
taken from the ``X-Actor`` header (a username) purely for audit-trail tracking,
defaulting to ``admin``.
"""
from __future__ import annotations

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from wms.db import get_session
from wms.models import User
from wms.services.catalog import get_or_create_user


def db_session() -> Session:
    yield from get_session()


def current_user(x_actor: str = Header(default="admin"),
                 db: Session = Depends(db_session)) -> User:
    return get_or_create_user(db, username=x_actor)
