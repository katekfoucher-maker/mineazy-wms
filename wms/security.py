"""Password hashing + role-based permissions."""
from __future__ import annotations

import bcrypt

# ---- passwords ----------------------------------------------------------
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except ValueError:
        return False


# ---- roles + permissions ---------------------------------------------------
# "user" is the self-service tier: signs up with a Gmail account (see
# /signup), starts unapproved (User.is_approved=False) until a "users.admin"
# holder approves it on /users. It deliberately sits outside every internal
# staff permission below (backorder.*/products.manage/receiving.enter) so
# those stay hidden with no extra template work - see nav.full/forecast.view.
ROLES = ["admin", "controller", "clerk", "branch", "analyst", "user"]

ROLE_LABEL = {
    "admin": "System Administrator",
    "controller": "Procurement Controller",
    "clerk": "Branch / Order Clerk",
    "branch": "Branch User",
    "analyst": "Reporting Analyst",
    "user": "Standard User",
}

_STAFF_ROLES = {"admin", "controller", "clerk", "branch", "analyst"}

# permission -> which roles hold it
_PERMS: dict[str, set[str]] = {
    "view":             {"admin", "controller", "clerk", "branch", "analyst", "user"},
    "backorder.enter":  {"admin", "controller", "clerk"},   # create back order / delivery note
    "backorder.manage": {"admin", "controller"},            # advance stages / cancel
    "products.manage":  {"admin", "controller", "clerk"},   # add / edit a product catalogue entry
    "receiving.enter":  {"admin", "controller", "clerk"},   # enter/reverse a receiving or Recon dispatch order
    "users.admin":      {"admin"},
    # nav items / page sections a plain "user" doesn't get: Receiving
    # Orders, Recon, Products (nav) and the Model comparison card (Flow
    # Analysis) - every staff role keeps seeing all of these as before.
    "nav.full":         set(_STAFF_ROLES),
    "forecast.compare": set(_STAFF_ROLES),
}


def can(role: str, permission: str) -> bool:
    return role in _PERMS.get(permission, set())
