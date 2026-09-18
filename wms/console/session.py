"""Console session: a DB session + the acting user (for audit tracking)."""
from __future__ import annotations

from wms.db import SessionLocal
from wms.services.catalog import get_or_create_user


class Context:
    def __init__(self, actor: str = "admin"):
        self.db = SessionLocal()
        self.user = get_or_create_user(self.db, username=actor)

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass
