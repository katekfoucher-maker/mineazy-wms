"""Web layer: templating, session auth, RBAC, flash messages."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from wms.db import get_session
from wms.models import User
from wms.security import ROLE_LABEL, can

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class Redirect(Exception):
    """Raised inside a dependency to bounce the browser somewhere (e.g. /login)."""
    def __init__(self, url: str):
        self.url = url


# ---- flash messages -------------------------------------------------------
def flash(request: Request, message: str, category: str = "info") -> None:
    request.session.setdefault("_flash", []).append({"m": message, "c": category})


def pop_flashes(request: Request) -> list[dict]:
    return request.session.pop("_flash", [])


# ---- db + current user --------------------------------------------------
def db_session() -> Session:
    yield from get_session()


def current_user(request: Request, db: Session = Depends(db_session)) -> Optional[User]:
    uid = request.session.get("uid")
    if not uid:
        return None
    return db.query(User).filter(User.id == uid, User.is_active.is_(True)).first()


def require_login(request: Request, user: Optional[User] = Depends(current_user)) -> User:
    if user is None:
        request.session["_next"] = str(request.url.path)
        raise Redirect("/login")
    return user


def require_perm(permission: str):
    def _dep(request: Request, user: User = Depends(require_login)) -> User:
        if not can(user.role, permission):
            flash(request, f"Your role ({ROLE_LABEL.get(user.role, user.role)}) "
                           f"cannot perform that action.", "error")
            raise Redirect(request.headers.get("referer") or "/")
        return user
    return _dep


def render(request: Request, name: str, user: User, **ctx):
    return templates.TemplateResponse(name, {
        "request": request, "user": user, "can": can,
        "flashes": pop_flashes(request), "role_label": ROLE_LABEL,
        **ctx,
    })
