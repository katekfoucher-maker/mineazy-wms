"""Copy REAL business data from a local SQLite database into the database
configured via DATABASE_URL (e.g. your Aiven MySQL) - branches, products,
receiving orders, dispatch orders, delivery notes, back orders, stock on
hand, sales history, and the audit log.

Deliberately EXCLUDES the ``users`` table. The local dev database's demo
logins (admin/wms1234 etc - see wms/scripts/seed.py) are public knowledge,
documented in this repo's own README, so copying them over would hand out a
known password on your live, internet-facing site. Create your own real
admin login separately with ``python -m wms.scripts.bootstrap``. Every
copied row's "created_by" / "user_id" reference is set to NULL instead of
carrying over a demo user id that won't exist in the target - the business
record itself is kept, just without old attribution.

    python -m wms.scripts.migrate_to_mysql [path/to/wms.db]

Defaults to the project's wms.db if no path is given. The target is
whatever DATABASE_URL (+ DB_SSL_CA) your environment already points to -
the same variables you used for wms.scripts.bootstrap.

Safe to re-run: skips any row whose id already exists in the target, and
reports (without aborting the rest) any row that fails to copy - e.g. text
too long for a column, or a duplicate unique value.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, insert, select, text

from wms import models  # noqa: F401  (register every table)
from wms.db import Base, engine as target_engine, init_db

# not migrated - see module docstring
_SKIP_TABLES = {"users"}
# nullable FK columns that reference users.id - reset instead of carried over
_USER_FK_COLUMNS = {
    "receiving_orders": "created_by",
    "dispatch_orders": "created_by",
    "delivery_notes": "created_by",
    "back_orders": "created_by",
    "back_order_events": "user_id",
    "audit_logs": "user_id",
}


def run(sqlite_path: str) -> None:
    src_path = Path(sqlite_path).resolve()
    if not src_path.exists():
        raise SystemExit(f"Source SQLite file not found: {src_path}")

    print(f"Source: {src_path}")
    print(f"Target: {target_engine.url.render_as_string(hide_password=True)}")
    print(f"Skipping table: 'users' (create your own real login with "
         "wms.scripts.bootstrap instead)")
    if input("Copy all other data from source into target? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    source_engine = create_engine(f"sqlite:///{src_path}")
    init_db()   # make sure the target's tables exist

    with source_engine.connect() as src:
        for table in Base.metadata.sorted_tables:
            if table.name in _SKIP_TABLES:
                continue
            rows = [dict(r._mapping) for r in src.execute(select(table))]
            if not rows:
                print(f"  {table.name}: 0 rows")
                continue

            fk_col = _USER_FK_COLUMNS.get(table.name)
            if fk_col:
                for r in rows:
                    r[fk_col] = None

            with target_engine.begin() as dst:
                existing_ids = {r[0] for r in dst.execute(select(table.c.id))}
            new_rows = [r for r in rows if r["id"] not in existing_ids]

            copied = failed = 0
            for r in new_rows:
                try:
                    with target_engine.begin() as dst:
                        dst.execute(insert(table), [r])
                    copied += 1
                except Exception as e:                       # noqa: BLE001
                    failed += 1
                    print(f"    ! {table.name} id={r['id']} failed: {e}")

            already = len(rows) - len(new_rows)
            msg = f"  {table.name}: {copied} copied"
            if already:
                msg += f", {already} already present"
            if failed:
                msg += f", {failed} FAILED (see above)"
            print(msg)

    # MySQL's auto_increment counter doesn't know about the ids just inserted
    # directly - bump each table past the highest id so new rows the app
    # creates from here don't collide with a migrated id.
    if target_engine.url.get_backend_name() == "mysql":
        with target_engine.begin() as dst:
            for table in Base.metadata.sorted_tables:
                max_id = dst.execute(
                    select(table.c.id).order_by(table.c.id.desc()).limit(1)).scalar()
                if max_id:
                    dst.execute(text(
                        f"ALTER TABLE {table.name} AUTO_INCREMENT = {max_id + 1}"))

    print("Done.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "wms.db")
