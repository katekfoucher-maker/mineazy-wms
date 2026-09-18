"""Bootstrap a brand-new (empty) database with a real first admin login and
the warehouse branch - safe to run against real/live data, unlike
``seed.py`` (which starts with ``Base.metadata.drop_all()`` and loads
fabricated demo data).

    python -m wms.scripts.bootstrap

Creates tables if they don't exist yet (the same idempotent ``init_db()``
every app startup already runs - it never drops anything), then
interactively prompts for:
  - a first admin login (username, full name, password)
  - the warehouse branch (code "DC" - Receiving Orders / Recon require it)
  - any further real branches, one at a time

Everything else (products, receiving orders, recon, back orders) can be
entered through the web app once you can log in.
"""
from __future__ import annotations

import getpass

from wms.db import SessionLocal, init_db
from wms.errors import WMSError
from wms.models import Branch, User
from wms.security import ROLES, hash_password
from wms.services.catalog import create_branch


def _prompt_role() -> str:
    while True:
        role = input(f"Role {ROLES} [admin]: ").strip() or "admin"
        if role in ROLES:
            return role
        print(f"  Not a valid role - pick one of {ROLES}.")


def _create_user(db) -> None:
    print("\n-- First admin login --")
    username = input("Username: ").strip()
    if not username:
        print("  Skipped (no username entered).")
        return
    if db.query(User).filter(User.username == username).first():
        print(f"  '{username}' already exists - skipped.")
        return
    full_name = input("Full name: ").strip() or username
    role = _prompt_role()
    while True:
        pw = getpass.getpass("Password: ")
        pw2 = getpass.getpass("Confirm password: ")
        if pw and pw == pw2:
            break
        print("  Passwords empty or didn't match - try again.")
    db.add(User(username=username, full_name=full_name, role=role,
                password_hash=hash_password(pw)))
    db.commit()
    print(f"  Created user '{username}' ({role}).")


def _create_branches(db) -> None:
    print("\n-- Branches --")
    if not db.query(Branch).filter(Branch.code == "DC").first():
        print("Receiving Orders / Recon need a warehouse branch with code 'DC'.")
        name = (input("DC branch display name [DISTRIBUTION CENTER]: ").strip()
                or "DISTRIBUTION CENTER")
        create_branch(db, code="DC", name=name)
        print("  Created DC.")
    else:
        print("  DC already exists - skipped.")

    print("Add any other real branches now (blank code to stop).")
    while True:
        code = input("Branch code: ").strip().upper()
        if not code:
            break
        if db.query(Branch).filter(Branch.code == code).first():
            print(f"  '{code}' already exists - skipped.")
            continue
        name = input(f"  Name for {code}: ").strip() or code
        try:
            create_branch(db, code=code, name=name)
            print(f"  Created {code}.")
        except WMSError as e:
            print(f"  {e}")


def run() -> None:
    init_db()
    db = SessionLocal()
    try:
        _create_user(db)
        _create_branches(db)
    finally:
        db.close()
    print("\nDone. Log in at /login, then use Products (in the app) to build "
         "your real catalogue.")


if __name__ == "__main__":
    run()
