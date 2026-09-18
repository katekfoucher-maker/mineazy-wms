"""Delivery-note entry.

A delivery note is the requested-vs-sent source document (like HansaWorld
Stock-Movement doc 26503244).  On save:
  * the *sent* quantity is recorded as branch sales (demand signal), and
  * every line where ``requested > sent`` raises one back order at SUBMITTED
    (grouped into a single ``BackOrder`` per note).

    backorder_qty = requested_qty - sent_qty      (a blank sent qty counts as 0)
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.enums import SalesSource
from wms.errors import WMSError
from wms.models import Branch, DeliveryNote, DeliveryNoteLine, Product, SalesRecord
from wms.services.catalogue import resolve_product as _product


def _next_no(db: Session) -> str:
    return f"DN-{(db.query(func.count(DeliveryNote.id)).scalar() or 0) + 1:06d}"


def enter_delivery_note(
    db: Session, *,
    branch_id: int,
    lines: list[dict],
    from_location_label: str = "DC",
    doc_no: Optional[str] = None,
    doc_date: Optional[date] = None,
    comment: Optional[str] = None,
    record_sales: bool = True,
    raise_backorder: bool = True,
    commit: bool = True,
    user_id: Optional[int] = None,
) -> DeliveryNote:
    """lines: [{sku|product_id, requested_qty, sent_qty(optional)}]

    ``raise_backorder`` groups this note's shortfall into a per-note back order
    (the classic behaviour). The weekly dispatch flow passes ``raise_backorder=
    False`` and folds the shortfall into a weekly bucket itself.
    ``commit=False`` leaves the transaction open for the caller.
    """
    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise WMSError("Branch not found.")
    if not lines:
        raise WMSError("A delivery note needs at least one line.")

    dn_no = (doc_no or "").strip() or _next_no(db)
    clash = db.query(DeliveryNote).filter(DeliveryNote.dn_no == dn_no).first()
    if clash:
        raise WMSError(
            f"Stock Movement '{dn_no}' has already been entered"
            f"{f' for {clash.branch.name}' if clash.branch else ''}"
            f" on {clash.doc_date}. Open that back order instead, or clear the "
            f"Stock Movement ID to file this as a separate document.")

    d = doc_date or date.today()
    dn = DeliveryNote(dn_no=dn_no, branch_id=branch_id,
                      from_location_label=from_location_label, doc_date=d,
                      comment=comment, created_by=user_id)
    db.add(dn)
    db.flush()

    for raw in lines:
        p = _product(db, raw.get("product_id") or raw.get("sku"),
                     name=raw.get("description") or raw.get("name"), user_id=user_id)
        req = int(raw["requested_qty"])
        sent = int(raw.get("sent_qty") or 0)
        if req <= 0:
            raise WMSError(f"'{p.sku}': requested qty must be positive.")
        if sent < 0:
            raise WMSError(f"'{p.sku}': sent qty cannot be negative.")
        if sent > req:
            raise WMSError(f"'{p.sku}': sent ({sent}) exceeds requested ({req}).")
        db.add(DeliveryNoteLine(dn_id=dn.id, product_id=p.id,
                                requested_qty=req, sent_qty=sent))
        if record_sales and sent > 0:
            db.add(SalesRecord(branch_id=branch_id, product_id=p.id, sale_date=d,
                               qty=sent, unit_price=p.unit_price,
                               source=SalesSource.DELIVERY.value, source_ref=dn_no))
    db.flush()

    bo = None
    if raise_backorder:
        from wms.services import backorder_entry
        bo = backorder_entry.from_delivery_note(db, dn, user_id=user_id)
    write_audit(db, entity_type="DeliveryNote", entity_id=dn.id, action="ENTER",
                detail={"dn_no": dn.dn_no, "branch": branch.name, "lines": len(lines),
                        "back_order": bo.bo_no if bo else None}, user_id=user_id)
    if commit:
        db.commit()
        db.refresh(dn)
    else:
        db.flush()
    return dn


def serialize(db: Session, dn: DeliveryNote) -> dict:
    return {
        "id": dn.id, "dn_no": dn.dn_no, "branch_id": dn.branch_id,
        "branch": dn.branch.name if dn.branch else "",
        "from_location": dn.from_location_label, "doc_date": dn.doc_date,
        "comment": dn.comment,
        "total_requested": dn.total_requested, "total_sent": dn.total_sent,
        "lines": [{
            "line_id": l.id, "sku": l.product.sku, "description": l.product.name,
            "requested_qty": l.requested_qty, "sent_qty": l.sent_qty,
            "backorder_qty": l.backorder_qty,
        } for l in dn.lines],
    }
