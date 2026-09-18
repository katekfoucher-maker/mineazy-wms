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
ROLES = ["admin", "controller", "clerk", "branch", "analyst"]

ROLE_LABEL = {
    "admin": "System Administrator",
    "controller": "Procurement Controller",
    "clerk": "Branch / Order Clerk",
    "branch": "Branch User",
    "analyst": "Reporting Analyst",
}

# permission -> which roles hold it
_PERMS: dict[str, set[str]] = {
    "view":             {"admin", "controller", "clerk", "branch", "analyst"},
    "backorder.enter":  {"admin", "controller", "clerk"},   # create back order / delivery note
    "backorder.manage": {"admin", "controller"},            # advance stages / cancel
    "products.manage":  {"admin", "controller", "clerk"},   # add / edit a product catalogue entry
    "receiving.enter":  {"admin", "controller", "clerk"},   # enter/reverse a receiving or Recon dispatch order
    "users.admin":      {"admin"},
}


def can(role: str, permission: str) -> bool:
    return role in _PERMS.get(permission, set())
