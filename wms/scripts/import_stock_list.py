"""Import a product catalogue from an Excel stock list.

    python -m wms.scripts.import_stock_list "C:\\path\\STOCK LIST.xlsx"
    python -m wms.scripts.import_stock_list list.xlsx --sheet "Stock List" --fresh

Expected columns (case-insensitive, spaces / dots ignored):
    Item No | Name | Group        (Group -> category, expanded to a readable label)

Products are upserted by ``Item No``.  ``--fresh`` first wipes products and all
downstream data (back orders, delivery notes, sales).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from wms.db import SessionLocal, init_db
from wms.models import Product

# Group code -> readable category. Unknown codes are kept as-is.
GROUPS = {
    "MACH": "Machinery", "PUMP": "Pumps", "WELD": "Welding", "DRIL": "Drilling",
    "GALV": "Galvanised Fittings", "BEAR": "Bearings", "ELEC": "Electrical",
    "GENE": "General", "PPE": "PPE", "HDPE": "HDPE Piping", "POWE": "Power Tools",
    "CRUS": "Crushing Spares", "HAND": "Hand Tools", "PVC": "PVC Piping",
    "LUBR": "Lubricants", "BOLT": "Bolts & Fasteners", "CLOT": "Clothing",
    "LIFT": "Lifting Equipment", "PULL": "Pulleys", "STAN": "Stands",
    "HARD": "Hardware", "AUTO": "Automotive", "CHEM": "Chemicals",
    "E/MO": "Electric Motors", "PAIN": "Paint", "FENC": "Fencing", "CLAM": "Clamps",
    "STEE": "Steel", "SCAL": "Scales", "ABBR": "Abrasives", "HOSE": "Hoses",
    "FARM": "Farming", "BEAT": "Hammermill Beaters", "SEAL": "Seals",
    "PLAS": "Plastics", "SHEL": "Shelving", "DRIN": "Consumables",
    "LIGH": "Lighting", "SOLA": "Solar", "TANK": "Tanks", "GOLD": "Gold Processing",
    "SUND": "Sundries",
}


def _norm(col: str) -> str:
    return str(col).strip().lower().replace(" ", "").replace(".", "")


def run(path: str, sheet: str | None, fresh: bool, dry_run: bool) -> None:
    init_db()
    f = Path(path)
    if not f.exists():
        sys.exit(f"File not found: {path}")

    xl = pd.ExcelFile(f)
    sheet = sheet or ("Stock List" if "Stock List" in xl.sheet_names else xl.sheet_names[0])
    raw = xl.parse(sheet)
    cols = {_norm(c): c for c in raw.columns}
    sku_col = cols.get("itemno") or cols.get("sku") or cols.get("code")
    name_col = cols.get("name") or cols.get("description")
    group_col = cols.get("group") or cols.get("category")
    if not sku_col or not name_col:
        sys.exit(f"Sheet '{sheet}' needs 'Item No' and 'Name' columns; found {list(raw.columns)}")

    df = pd.DataFrame({
        "sku": raw[sku_col].astype(str).str.strip(),
        "name": raw[name_col].astype(str).str.strip(),
        "group": raw[group_col].astype(str).str.strip() if group_col else "",
    })
    df = df[(df["sku"] != "") & (df["sku"].str.lower() != "nan")]
    df = df.drop_duplicates(subset="sku", keep="first").reset_index(drop=True)

    print(f"Sheet '{sheet}': {len(df)} unique items")
    print("Groups:", ", ".join(f"{g}({n})" for g, n in df.group.value_counts().head(12).items()), "...")
    if dry_run:
        print("(dry run - nothing written)")
        return

    db = SessionLocal()
    try:
        if fresh:
            from wms.models import (
                BackOrderEvent, BackOrderItem, BackOrder, DeliveryNoteLine,
                DeliveryNote, SalesRecord,
            )
            for m in (BackOrderEvent, BackOrderItem, BackOrder, DeliveryNoteLine,
                      DeliveryNote, SalesRecord, Product):
                db.query(m).delete()
            db.commit()
            print("  --fresh: cleared products + back orders + delivery notes + sales")

        existing = {p.sku: p for p in db.query(Product).all()}
        created = updated = 0
        for r in df.itertuples(index=False):
            cat = GROUPS.get(r.group, r.group or None)
            p = existing.get(r.sku)
            if p:
                p.name, p.category = r.name, cat
                updated += 1
            else:
                db.add(Product(sku=r.sku, name=r.name, category=cat, uom="EA"))
                created += 1
        db.commit()
        print(f"Done: {created} created, {updated} updated, "
              f"total products = {db.query(Product).count()}")
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Import a product catalogue from Excel")
    ap.add_argument("path")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="wipe products + back orders + delivery notes + sales first")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.path, a.sheet, a.fresh, a.dry_run)
