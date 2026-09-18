"""Refresh each branch's stock-on-hand from the latest weekly Hansa stock file.

    python -m wms.scripts.refresh_onhand           # apply
    python -m wms.scripts.refresh_onhand --dry-run # show what would change

For every branch that has files in ``data/weekly_inventory/`` it takes the most
recent week's ``... Stock List`` export, reads Item No / Balance (negative
balances -> 0), and REPLACES that branch's balance in the ``stock_on_hand``
table - the same table the Allocation plan and the Flow Analysis low-sales
table read as authoritative "On hand". A copy of that file is also written to
``data/inventory/<CODE>.xlsx`` so the spreadsheet-snapshot fallback matches.

Run this after loading a new batch of weekly stock files so the whole site
shows current on-hand alongside the retrained forecast.
"""
from __future__ import annotations

import argparse
import shutil
import sys

import pandas as pd


def _latest_per_branch(files):
    """{branch_code: (path, week_start)} keeping the newest week per branch."""
    from wms.analytics import weekly_forecast as wf
    best: dict = {}
    for f in files:
        code, ws = wf.parse_name(f)
        if not code or ws is None:
            continue
        if code not in best or ws > best[code][1]:
            best[code] = (f, ws)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="refresh_onhand", description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the files/line counts, write nothing")
    a = ap.parse_args(argv)

    import glob
    import os
    from wms.analytics import weekly_forecast as wf
    from wms.analytics import inventory as inv_mod
    from wms.db import SessionLocal
    from wms.models import Branch
    from wms.services import stock as stock_svc

    d = wf.weekly_inventory_dir()
    files = [f for f in sorted(glob.glob(str(d / "**" / "*.xls*"), recursive=True))
             if not os.path.basename(f).startswith("~$")]
    if not files:
        print(f"no weekly stock files in {d}", file=sys.stderr)
        return 1

    latest = _latest_per_branch(files)
    if not latest:
        print("no branch/week could be read from the file names", file=sys.stderr)
        return 1

    snap_dir = inv_mod.inventory_dir()
    snap_dir.mkdir(parents=True, exist_ok=True)
    db = SessionLocal()
    total = 0
    try:
        for code, (path, ws) in sorted(latest.items()):
            raw = pd.read_excel(path, header=0, dtype=str)
            sc = inv_mod._pick(raw.columns, inv_mod._SKU_KEYS) or raw.columns[0]
            qc = inv_mod._pick(raw.columns, inv_mod._QTY_KEYS)
            if qc is None:
                print(f"  {code}: no quantity column in {os.path.basename(path)} - skipped")
                continue
            raw = raw[raw[sc].notna()]
            qty = pd.to_numeric(raw[qc], errors="coerce").fillna(0.0)
            items, nonzero = [], 0
            for sku, q in zip(raw[sc].astype(str).str.strip(), qty):
                if not sku or sku.lower() == "nan":
                    continue
                v = max(0, int(round(float(q))))         # negatives -> 0
                items.append({"sku": sku, "qty": v})
                nonzero += v > 0
            br = db.query(Branch).filter(Branch.code == code).first()
            tag = "would set" if a.dry_run else "set"
            if br is None:
                print(f"  {code}: unknown branch code - skipped "
                      f"({os.path.basename(path)})")
                continue
            print(f"  {code}  {ws.date()}  {os.path.basename(path)}  "
                  f"{tag} {len(items):,} lines ({nonzero:,} in stock)")
            if not a.dry_run:
                stock_svc.set_branch_stock(db, branch_id=br.id, items=items,
                                           user_id=None)
                ext = os.path.splitext(path)[1].lower()
                ext = ext if ext in (".xlsx", ".xls", ".csv") else ".xlsx"
                shutil.copyfile(path, snap_dir / f"{code}{ext}")
            total += len(items)
    finally:
        db.close()

    print(f"\n{'dry run - nothing written' if a.dry_run else 'done'}: "
          f"{len(latest)} branch(es), {total:,} lines. "
          f"The Allocation plan and Flow Analysis now read this on-hand.")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
