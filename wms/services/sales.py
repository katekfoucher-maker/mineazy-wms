"""Sales-history ingest (feeds forecasting + allocation)."""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.enums import SalesSource
from wms.errors import WMSError
from wms.models import Branch, Product, SalesRecord


def record_sale(db: Session, *, branch_id: int, product_id: int, sale_date: date,
                qty: int, unit_price: Optional[float] = None,
                source: str = SalesSource.POS.value) -> SalesRecord:
    if qty <= 0:
        raise WMSError("Sale quantity must be positive.")
    row = SalesRecord(branch_id=branch_id, product_id=product_id, sale_date=sale_date,
                      qty=qty, unit_price=unit_price, source=source)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def import_csv(db: Session, path: str, *, user_id: Optional[int] = None) -> int:
    """CSV columns: branch_code, sku, sale_date (YYYY-MM-DD), qty[, unit_price]."""
    p = Path(path)
    if not p.exists():
        raise WMSError(f"File not found: {path}", 404)
    branches = {b.code: b.id for b in db.query(Branch).all()}
    products = {pr.sku: pr for pr in db.query(Product).all()}
    n = 0
    with p.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
            bid = branches.get(row.get("branch_code") or row.get("branch"))
            prod = products.get(row.get("sku") or row.get("item_no"))
            if not bid or not prod:
                continue
            d = datetime.strptime(row["sale_date"], "%Y-%m-%d").date()
            qty = int(float(row["qty"]))
            price = float(row["unit_price"]) if row.get("unit_price") else prod.unit_price
            db.add(SalesRecord(branch_id=bid, product_id=prod.id, sale_date=d, qty=qty,
                               unit_price=price, source=SalesSource.IMPORT.value))
            n += 1
    write_audit(db, entity_type="SalesRecord", action="IMPORT_CSV",
                detail={"rows": n, "file": p.name}, user_id=user_id)
    db.commit()
    return n
