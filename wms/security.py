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
# holder approves it on /users. It gets Warehouse (nav.full/receiving.enter,
# below) but stays outside products.manage/products.view/users.admin/
# forecast.compare - see products.view/forecast.compare below.
#
# "member" is an admin-created general-access tier: sees/uses everything a
# staff role does (Warehouse, Recon, Allocation, Branches, back orders) EXCEPT
# the Products and Users nav items and the Model comparison card on Flow
# Analysis - see products.view/users.admin/forecast.compare below.
ROLES = ["admin", "controller", "clerk", "branch", "analyst", "member", "user"]

ROLE_LABEL = {
    "admin": "System Administrator",
    "controller": "Procurement Controller",
    "clerk": "Branch / Order Clerk",
    "branch": "Branch User",
    "analyst": "Reporting Analyst",
    "member": "Team Member",
    "user": "Standard User",
}

_STAFF_ROLES = {"admin", "controller", "clerk", "branch", "analyst"}

# permission -> which roles hold it
_PERMS: dict[str, set[str]] = {
    "view":             {"admin", "controller", "clerk", "branch", "analyst", "member", "user"},
    "backorder.enter":  {"admin", "controller", "clerk", "member"},   # create back order / delivery note
    "backorder.manage": {"admin", "controller"},            # advance stages / cancel
    "products.manage":  {"admin", "controller", "clerk"},   # add / edit a product catalogue entry
    "receiving.enter":  {"admin", "controller", "clerk", "member", "user"},  # enter/reverse a receiving or Recon dispatch order
    "users.admin":      {"admin"},
    # Warehouse/Recon nav + page: every staff role, "member" and "user" all
    # get this now. Products/Users nav and the Model comparison card stay
    # gated separately below (products.view/users.admin/forecast.compare).
    "nav.full":         set(_STAFF_ROLES) | {"member", "user"},
    # Split by predicted sales (Allocation plan tab): everyone with nav.full
    # can use it - it's a core Allocation feature, not the data-upload cards
    # gated by backorder.enter above ("user" stays out of those).
    "allocation.split": set(_STAFF_ROLES) | {"member", "user"},
    # Products (nav + page): every staff role keeps it; "member" doesn't.
    "products.view":    set(_STAFF_ROLES),
    "forecast.compare": set(_STAFF_ROLES),
}


def can(role: str, permission: str) -> bool:
    return role in _PERMS.get(permission, set())
