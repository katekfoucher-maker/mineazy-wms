"""Bulk-import (or refresh) the Product catalogue from a full stock-list
export - e.g. "Item No / Name / Group / Qty" columns, one row per SKU.

    python -m wms.scripts.import_product_catalog "C:\\path\\to\\STOCK LIST.xlsx" [--yes]

The Product table only exists as a name/category reference for SKUs a weekly
sales/inventory upload doesn't otherwise resolve (see products_page's
docstring in wms/web/routes.py) - it was seeded from a single ~90-line demo
document, not the real catalogue, which is why branch-inventory lines for
SKUs outside that small set show up with no product name. This script fills
that gap from the real, full list.

New SKUs are inserted; SKUs that already exist have their name/category
updated only if the file's value actually differs (an unchanged row is left
alone, including any unit_price/uom set by hand elsewhere). One bulk
operation each way, not one row at a time - this file is thousands of rows.
"""
from __future__ import annotations

import sys

import pandas as pd

from wms.db import init_db


def run(path: str, assume_yes: bool = False) -> None:
    df = pd.read_excel(path, sheet_name="Stock List")
    df = df[["Item No", "Name", "Group"]].copy()
    df["Item No"] = df["Item No"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Group"] = df["Group"].astype(str).str.strip()
    df = df[(df["Item No"] != "") & (df["Item No"].str.lower() != "nan")]
    df = df.drop_duplicates(subset="Item No", keep="last")

    print(f"Source: {path}")
    print(f"{len(df)} product row(s) found.")

    init_db()
    from wms.db import SessionLocal
    from wms.models import Product

    db = SessionLocal()
    try:
        existing = {p.sku: p for p in db.query(Product).all()}
        to_insert, to_update = [], []
        for r in df.itertuples(index=False):
            sku, name, group = r[0], r[1], (r[2] or None)
            p = existing.get(sku)
            if p is None:
                to_insert.append({"sku": sku, "name": name[:200], "category": group,
                                  "uom": "EA", "unit_price": None, "is_active": True})
            elif p.name != name or p.category != group:
                to_update.append({"sku": sku, "name": name[:200], "category": group})

        print(f"  new products to add:      {len(to_insert)}")
        print(f"  existing products to update: {len(to_update)}")
        print(f"  unchanged:                 {len(df) - len(to_insert) - len(to_update)}")

        if not assume_yes:
            if input("Apply these changes to the Product catalogue? [y/N] ").strip().lower() != "y":
                print("Aborted.")
                return

        from sqlalchemy import insert, update as sa_update
        if to_insert:
            db.execute(insert(Product), to_insert)
        for row in to_update:
            db.execute(sa_update(Product).where(Product.sku == row["sku"])
                       .values(name=row["name"], category=row["category"]))
        db.commit()
        print(f"Done: {len(to_insert)} added, {len(to_update)} updated.")
    finally:
        db.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--yes"]
    yes = "--yes" in sys.argv[1:]
    if not args:
        raise SystemExit("Usage: python -m wms.scripts.import_product_catalog <path.xlsx> [--yes]")
    run(args[0], assume_yes=yes)
