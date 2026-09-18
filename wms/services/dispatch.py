"""Dispatch-order entry (Recon) - stock leaving the warehouse for a branch.

A dispatch order is the mirror image of a receiving order: instead of
recording what arrived at the warehouse, it records what left it for a
branch, subtracting the dispatched quantity from the warehouse's running
stock-on-hand balance and adding it onto the destination branch's. There is
no requested-vs-sent shortfall and no back order raised here - this is
purely a stock movement, for an accurate warehouse balance. (The older
DeliveryNote/BackOrder flow still exists separately and is untouched.)
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.errors import WMSError
from wms.models import Branch, DispatchOrder, DispatchLine
from wms.services.catalogue import resolve_product as _product


def _next_no(db: Session) -> str:
    return f"DO-{(db.query(func.count(DispatchOrder.id)).scalar() or 0) + 1:06d}"


def _warehouse_branch(db: Session) -> Branch:
    wh = db.query(Branch).filter(Branch.code == "DC").first()
    if not wh:
        raise WMSError("No DISTRIBUTION CENTER (warehouse) branch is set up.")
    return wh


def enter_dispatch_order(
    db: Session, *,
    branch_id: int,
    lines: list[dict],
    doc_no: Optional[str] = None,
    doc_date: Optional[date] = None,
    comment: Optional[str] = None,
    commit: bool = True,
    user_id: Optional[int] = None,
) -> tuple[DispatchOrder, list[str]]:
    """lines: [{sku|product_id, dispatched_qty}]

    Each line is capped at what the warehouse actually has on hand - a
    dispatch can never push the warehouse balance below zero, since that
    would stop it from being an accurate record of what's really there. A
    capped line comes back as a warning, not a hard failure, so the rest of
    the document still goes through.
    """
    from wms.services import stock as stock_svc

    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise WMSError("Branch not found.")
    if not lines:
        raise WMSError("A dispatch order needs at least one line.")
    wh = _warehouse_branch(db)
    if wh.id == branch_id:
        raise WMSError("The destination branch can't be the warehouse itself.")

    do_no = (doc_no or "").strip() or _next_no(db)
    clash = db.query(DispatchOrder).filter(DispatchOrder.do_no == do_no).first()
    if clash:
        raise WMSError(
            f"Dispatch order '{do_no}' has already been entered"
            f"{f' for {clash.branch.name}' if clash.branch else ''}"
            f" on {clash.doc_date}. Use a different reference to file this "
            f"as a separate dispatch.")

    lv = stock_svc.levels_df(db)
    wh_stock: dict = {}
    if not lv.empty:
        sub = lv[lv["branch_code"] == wh.code]
        wh_stock = {str(r.sku).upper(): int(r.on_hand) for r in sub.itertuples()}

    d = doc_date or date.today()
    do = DispatchOrder(do_no=do_no, branch_id=branch_id, doc_date=d,
                       comment=comment, created_by=user_id)
    db.add(do)
    db.flush()

    warnings: list[str] = []
    items = []
    for raw in lines:
        p = _product(db, raw.get("product_id") or raw.get("sku"),
                     name=raw.get("description") or raw.get("name"), user_id=user_id)
        qty = int(raw.get("dispatched_qty") or raw.get("qty") or 0)
        if qty <= 0:
            raise WMSError(f"'{p.sku}': dispatched qty must be positive.")
        avail = wh_stock.get(p.sku.upper(), 0)
        sent = min(qty, avail)
        if sent < qty:
            warnings.append(f"{p.sku}: only {avail:,} on hand at the warehouse - "
                            f"capped from {qty:,} to {sent:,}.")
        if sent <= 0:
            warnings.append(f"{p.sku}: none available at the warehouse - line skipped.")
            continue
        db.add(DispatchLine(do_id=do.id, product_id=p.id, dispatched_qty=sent))
        items.append({"sku": p.sku, "qty": sent})
    db.flush()

    if not items:
        db.rollback()
        raise WMSError("Nothing could be dispatched - none of these lines have any "
                       "stock on hand at the warehouse.")

    stock_svc.add_stock(db, branch_id=wh.id,
                        items=[{"sku": i["sku"], "qty": -i["qty"]} for i in items],
                        user_id=user_id, commit=False)
    stock_svc.add_stock(db, branch_id=branch_id, items=items,
                        user_id=user_id, commit=False)

    write_audit(db, entity_type="DispatchOrder", entity_id=do.id, action="ENTER",
               detail={"do_no": do_no, "lines": len(items),
                       "units": sum(i["qty"] for i in items)},
               user_id=user_id)
    if commit:
        db.commit()
    return do, warnings


def reverse_dispatch_order(db: Session, do_no: str, *,
                           user_id: Optional[int] = None) -> dict:
    """Pull a dispatch order's units back out of the branch and return them
    to the warehouse, then delete it."""
    do = db.query(DispatchOrder).filter(DispatchOrder.do_no == do_no).first()
    if not do:
        raise WMSError(f"Dispatch order '{do_no}' not found.")

    from wms.services import stock as stock_svc
    wh = _warehouse_branch(db)
    units_pulled = do.total_dispatched
    branch_name = do.branch.name if do.branch else ""
    items = [{"sku": l.product.sku, "qty": l.dispatched_qty} for l in do.lines]

    stock_svc.add_stock(db, branch_id=do.branch_id,
                        items=[{"sku": i["sku"], "qty": -i["qty"]} for i in items],
                        user_id=user_id, commit=False)
    stock_svc.add_stock(db, branch_id=wh.id, items=items, user_id=user_id, commit=False)

    write_audit(db, entity_type="DispatchOrder", entity_id=do.id, action="REVERSE",
               detail={"do_no": do_no, "units_pulled": units_pulled}, user_id=user_id)
    db.delete(do)
    db.commit()
    return {"units_pulled": units_pulled, "branch": branch_name}
