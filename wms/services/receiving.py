"""Receiving-order entry - stock arriving into the warehouse.

A receiving order is the mirror image of a delivery note: instead of
recording what left the warehouse for a branch, it records what arrived at
one (normally the distribution centre) and adds it onto that location's
running stock-on-hand balance. There is no requested-vs-sent shortfall
concept here - what's on the line is what came in.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.errors import WMSError
from wms.models import Branch, ReceivingOrder, ReceivingLine
from wms.services.catalogue import resolve_product as _product


def _next_no(db: Session) -> str:
    return f"RO-{(db.query(func.count(ReceivingOrder.id)).scalar() or 0) + 1:06d}"


def enter_receiving_order(
    db: Session, *,
    branch_id: int,
    lines: list[dict],
    doc_no: Optional[str] = None,
    doc_date: Optional[date] = None,
    supplier: Optional[str] = None,
    comment: Optional[str] = None,
    commit: bool = True,
    user_id: Optional[int] = None,
) -> ReceivingOrder:
    """lines: [{sku|product_id, received_qty}]"""
    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise WMSError("Location not found.")
    if not lines:
        raise WMSError("A receiving order needs at least one line.")

    ro_no = (doc_no or "").strip() or _next_no(db)
    clash = db.query(ReceivingOrder).filter(ReceivingOrder.ro_no == ro_no).first()
    if clash:
        raise WMSError(
            f"Receiving order '{ro_no}' has already been entered"
            f"{f' for {clash.branch.name}' if clash.branch else ''}"
            f" on {clash.doc_date}. Use a different reference to file this "
            f"as a separate receipt.")

    d = doc_date or date.today()
    ro = ReceivingOrder(ro_no=ro_no, branch_id=branch_id, doc_date=d,
                        supplier=(supplier or None), comment=comment,
                        created_by=user_id)
    db.add(ro)
    db.flush()

    items = []
    for raw in lines:
        p = _product(db, raw.get("product_id") or raw.get("sku"),
                     name=raw.get("description") or raw.get("name"), user_id=user_id)
        qty = int(raw.get("received_qty") or raw.get("qty") or 0)
        if qty <= 0:
            raise WMSError(f"'{p.sku}': received qty must be positive.")
        db.add(ReceivingLine(ro_id=ro.id, product_id=p.id, received_qty=qty))
        items.append({"sku": p.sku, "qty": qty})
    db.flush()

    from wms.services import stock as stock_svc
    stock_svc.add_stock(db, branch_id=branch_id, items=items,
                        user_id=user_id, commit=False)

    write_audit(db, entity_type="ReceivingOrder", entity_id=ro.id, action="ENTER",
               detail={"ro_no": ro_no, "lines": len(items),
                       "units": sum(i["qty"] for i in items)},
               user_id=user_id)
    if commit:
        db.commit()
    return ro


def reverse_receiving_order(db: Session, ro_no: str, *,
                            user_id: Optional[int] = None) -> dict:
    """Pull a receiving order's units back out of stock, then delete it."""
    ro = db.query(ReceivingOrder).filter(ReceivingOrder.ro_no == ro_no).first()
    if not ro:
        raise WMSError(f"Receiving order '{ro_no}' not found.")

    units_pulled = ro.total_received
    branch_name = ro.branch.name if ro.branch else ""
    from wms.services import stock as stock_svc
    stock_svc.add_stock(
        db, branch_id=ro.branch_id,
        items=[{"sku": l.product.sku, "qty": -l.received_qty} for l in ro.lines],
        user_id=user_id, commit=False)

    write_audit(db, entity_type="ReceivingOrder", entity_id=ro.id, action="REVERSE",
               detail={"ro_no": ro_no, "units_pulled": units_pulled}, user_id=user_id)
    db.delete(ro)
    db.commit()
    return {"units_pulled": units_pulled, "branch": branch_name}
