"""One-time import of the file-based analytics data (monthly sales history,
weekly sales, weekly stock snapshots) into the database, so it survives on a
host with no persistent disk (see wms/models.py's MonthlySalesLine /
WeeklySalesLine / WeeklyStockSnapshotLine, and the "prefer files if present,
else DB" loaders in monthly_sales.py / weekly_forecast.py).

    python -m wms.scripts.import_file_analytics [--yes]

Reads from the project's configured sales_history_dir, weekly_sales_dir and
weekly_inventory_dir (local Excel files) and writes into the matching
database tables via the same save_month / save_week / save_inventory_week
functions the web upload routes use - so nothing here bypasses their
column/branch handling. Safe to re-run: each (branch, period/week) is
replaced wholesale by whatever the source file currently has, same as a
fresh upload would do.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

from wms.analytics import monthly_sales, weekly_forecast as weekly_fc
from wms.db import init_db

_RETRIES = 3


def _with_retries(fn, *args):
    """Retry a flaky remote-DB write a few times with a short backoff - a
    stalled connection over a real network is common enough (and now fails
    fast instead of hanging, see wms/db.py's read/write/connect timeouts)
    that one file failing shouldn't abort an otherwise-fine import run."""
    for attempt in range(1, _RETRIES + 1):
        try:
            return fn(*args)
        except Exception as e:                           # noqa: BLE001
            if attempt == _RETRIES:
                raise
            print(f"    ! attempt {attempt} failed ({e}); retrying...")
            time.sleep(2 * attempt)


def _import_sales_history() -> None:
    d = monthly_sales.history_dir()
    files = sorted(glob.glob(str(d / "*.xls*")))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    print(f"sales_history: {len(files)} file(s) in {d}")
    n_files = n_rows = 0
    failed = []
    for f in files:
        panel = monthly_sales._finish_panel([monthly_sales._read_file(f)])
        if panel.empty:
            print(f"  ! skipped (unparseable name or empty): {os.path.basename(f)}")
            continue
        bc, period = panel["branch_code"].iloc[0], panel["period"].iloc[0]
        try:
            saved = _with_retries(monthly_sales.save_month, bc, period, panel)
        except Exception as e:                            # noqa: BLE001
            print(f"  ! FAILED after {_RETRIES} attempts: {os.path.basename(f)}: {e}")
            failed.append(os.path.basename(f))
            continue
        n_files += 1
        n_rows += saved
        print(f"  {os.path.basename(f)}: {bc} {period.strftime('%b %Y')} - {saved} row(s)")
    print(f"sales_history: {n_files} file(s), {n_rows} row(s) saved"
         + (f", {len(failed)} FAILED: {failed}" if failed else ""))


def _import_weekly_sales() -> None:
    d = weekly_fc.weekly_dir()
    files = sorted(glob.glob(str(d / "**" / "*.xls*"), recursive=True))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    print(f"weekly_sales: {len(files)} file(s) in {d}")
    n_files = n_rows = 0
    failed = []
    for f in files:
        code, ws = weekly_fc.parse_name(f)
        if not code or ws is None:
            print(f"  ! skipped (unparseable name): {os.path.basename(f)}")
            continue
        rows = weekly_fc._items(f)
        try:
            saved = _with_retries(weekly_fc.save_week, code, ws, rows)
        except Exception as e:                            # noqa: BLE001
            print(f"  ! FAILED after {_RETRIES} attempts: {os.path.basename(f)}: {e}")
            failed.append(os.path.basename(f))
            continue
        n_files += 1
        n_rows += saved
        print(f"  {os.path.basename(f)}: {code} {ws.date()} - {saved} row(s)")
    print(f"weekly_sales: {n_files} file(s), {n_rows} row(s) saved"
         + (f", {len(failed)} FAILED: {failed}" if failed else ""))


def _import_weekly_inventory() -> None:
    d = weekly_fc.weekly_inventory_dir()
    files = sorted(glob.glob(str(d / "**" / "*.xls*"), recursive=True))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    print(f"weekly_inventory: {len(files)} file(s) in {d}")
    n_files = n_rows = 0
    failed = []
    for f in files:
        with open(f, "rb") as fh:
            raw = fh.read()
        parsed = weekly_fc.parse_upload_inventory(raw, os.path.basename(f))
        if not parsed:
            print(f"  ! skipped (unparseable name or no qty column): {os.path.basename(f)}")
            continue
        code, ws, rows = parsed
        try:
            saved = _with_retries(weekly_fc.save_inventory_week, code, ws, rows)
        except Exception as e:                            # noqa: BLE001
            print(f"  ! FAILED after {_RETRIES} attempts: {os.path.basename(f)}: {e}")
            failed.append(os.path.basename(f))
            continue
        n_files += 1
        n_rows += saved
        print(f"  {os.path.basename(f)}: {code} {ws.date()} - {saved} row(s)")
    print(f"weekly_inventory: {n_files} file(s), {n_rows} row(s) saved"
         + (f", {len(failed)} FAILED: {failed}" if failed else ""))


def run(assume_yes: bool = False) -> None:
    from wms.db import engine
    print(f"Target: {engine.url.render_as_string(hide_password=True)}")
    if not assume_yes:
        if input("Import all local sales_history/weekly_sales/weekly_inventory "
                "files into this database? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return
    init_db()
    _import_sales_history()
    _import_weekly_sales()
    _import_weekly_inventory()
    print("Done.")


if __name__ == "__main__":
    yes = "--yes" in sys.argv[1:] or os.environ.get("MIGRATE_YES") == "1"
    run(assume_yes=yes)
