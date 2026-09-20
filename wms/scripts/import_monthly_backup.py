"""Import a folder of monthly "Item Statistics" sales exports (a Hansa
backup, one file per month) into MonthlySalesLine - the database-backed
source Sales & Forecasting's monthly demand model reads.

    python -m wms.scripts.import_monthly_backup "C:\\path\\to\\BELMONT" [--yes]

Filenames must carry a month and branch, e.g. "JULY 2025 BELMONT SALES.xlsx"
- and should carry a YEAR too: older uploads that omit it default to 2026,
but any file naming its year is read as that real year (a backup spanning
2025-2026 is exactly what this was added for). Safe to re-run: each
(branch, month) is replaced wholesale by whatever the file currently has,
same as a fresh upload through the web app would do.
"""
from __future__ import annotations

import glob
import os
import sys

from wms.analytics import monthly_sales
from wms.db import init_db


def run(directory: str, assume_yes: bool = False) -> None:
    files = sorted(glob.glob(os.path.join(directory, "*.xls*")))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    if not files:
        raise SystemExit(f"No .xlsx/.xls files found in {directory}")

    print(f"Source: {directory}")
    print(f"{len(files)} file(s) found:")
    for f in files:
        month, branch, year = monthly_sales._parse_name(f)
        tag = "OK" if month and branch else "! unreadable name"
        yr = year if year else "(defaults to 2026 - no year in filename)"
        print(f"  {os.path.basename(f):45} {tag:20} year={yr}")

    if not assume_yes:
        if input("Import all of these into MonthlySalesLine? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return

    init_db()
    n_files = n_rows = 0
    failed = []
    for f in files:
        panel = monthly_sales._finish_panel([monthly_sales._read_file(f)])
        if panel.empty:
            print(f"  ! skipped (unparseable name or empty): {os.path.basename(f)}")
            continue
        bc, period = panel["branch_code"].iloc[0], panel["period"].iloc[0]
        try:
            saved = monthly_sales.save_month(bc, period, panel)
        except Exception as e:                                # noqa: BLE001
            print(f"  ! FAILED: {os.path.basename(f)}: {e}")
            failed.append(os.path.basename(f))
            continue
        n_files += 1
        n_rows += saved
        print(f"  {os.path.basename(f)}: {bc} {period.strftime('%b %Y')} - {saved} row(s)")

    print(f"\n{n_files} file(s), {n_rows} row(s) saved"
         + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    print("Done. The monthly demand model picks this up on its next request.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--yes"]
    yes = "--yes" in sys.argv[1:] or os.environ.get("MIGRATE_YES") == "1"
    if not args:
        raise SystemExit("Usage: python -m wms.scripts.import_monthly_backup <directory> [--yes]")
    run(args[0], assume_yes=yes)
