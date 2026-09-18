"""Backorder entry module.

A back order is a per-branch accumulator: while it is **OPEN** every dispatch
note for that branch loads its shortfall (``requested - sent``) onto it; closing
it stops that. ``attach_dispatch(dn)`` is the entry point for the dispatch flow.

Also here:
  * ``create_back_order(...)`` - manual entry
  * ``from_delivery_note(dn)`` - legacy per-note back order (API / seed)
  * ``reverse_dispatch(dn_no)`` - undo every effect of one dispatch
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.config import get_settings
from wms.enums import (
    BackOrderPriority, BackOrderSource, BackOrderStage, BackOrderStatus,
)
from wms.errors import WMSError
from wms.models import (
    BackOrder, BackOrderEvent, BackOrderItem, Branch, DeliveryNote,
    DeliveryNoteLine, Product, SalesRecord,
)
from wms.services.catalogue import resolve_product as _product

settings = get_settings()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def next_bo_no(db: Session) -> str:
    return f"BO-{(db.query(func.count(BackOrder.id)).scalar() or 0) + 1:06d}"


def _new_header(db, *, branch_id, source, source_ref, priority, notes,
                sla_days, user_id) -> BackOrder:
    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise WMSError("Branch not found.")
    bo = BackOrder(
        bo_no=next_bo_no(db), branch_id=branch_id, source=source, source_ref=source_ref,
        priority=priority or BackOrderPriority.NORMAL.value,
        stage=BackOrderStage.OPEN.value, status=BackOrderStatus.OPEN.value,
        sla_days=sla_days if sla_days is not None else settings.backorder_sla_days,
        notes=notes, created_by=user_id, submitted_at=_now(),
    )
    db.add(bo)
    db.flush()
    db.add(BackOrderEvent(back_order_id=bo.id, from_stage=None,
                          to_stage=BackOrderStage.OPEN.value, at=bo.submitted_at,
                          user_id=user_id, note="created"))
    return bo


def _add_item(db, bo: BackOrder, product: Product, qty: int) -> BackOrderItem:
    if qty <= 0:
        raise WMSError(f"'{product.sku}': quantity must be positive.")
    item = BackOrderItem(
        back_order_id=bo.id, product_id=product.id, sku=product.sku,
        description=product.name, category=product.category,
        unit_price=product.unit_price, qty_ordered=qty,
    )
    db.add(item)
    return item


def create_back_order(
    db: Session, *,
    branch_id: int,
    items: list[dict],
    priority: str = BackOrderPriority.NORMAL.value,
    notes: Optional[str] = None,
    source: str = BackOrderSource.MANUAL.value,
    source_ref: Optional[str] = None,
    sla_days: Optional[int] = None,
    user_id: Optional[int] = None,
) -> BackOrder:
    """items: [{sku|product_id, qty}]"""
    if not items:
        raise WMSError("A back order needs at least one item.")
    bo = _new_header(db, branch_id=branch_id, source=source, source_ref=source_ref,
                     priority=priority, notes=notes, sla_days=sla_days, user_id=user_id)
    for raw in items:
        p = _product(db, raw.get("product_id") or raw.get("sku"),
                     name=raw.get("description") or raw.get("name"), user_id=user_id)
        _add_item(db, bo, p, int(raw.get("qty") or raw.get("qty_ordered")))
    write_audit(db, entity_type="BackOrder", entity_id=bo.id, action="CREATE",
                detail={"bo_no": bo.bo_no, "branch_id": branch_id,
                        "items": len(items), "source": source}, user_id=user_id)
    db.commit()
    db.refresh(bo)
    return bo


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def weekly_bo_no(branch_code: str, week_start: date) -> str:
    iso = week_start.isocalendar()
    return f"BO-{branch_code}-{iso[0]}W{iso[1]:02d}"


def open_dispatch_bo(db: Session, branch_id: int) -> Optional[BackOrder]:
    """The branch's back order that is currently accepting dispatches, if any."""
    return (db.query(BackOrder)
            .filter(BackOrder.branch_id == branch_id,
                    BackOrder.source == BackOrderSource.DISPATCH.value,
                    BackOrder.status == BackOrderStatus.OPEN.value)
            .order_by(BackOrder.id)
            .first())


def _rebuild_dispatch_items(db: Session, bo: BackOrder) -> None:
    """Recompute a dispatch back order's item lines as the summed shortfall
    (``requested - sent``) over every dispatch note still attached to it."""
    refs = [r for r in (bo.source_ref or "").split(",") if r]
    short: dict[int, int] = {}
    if refs:
        rows = (db.query(DeliveryNoteLine)
                .join(DeliveryNote, DeliveryNote.id == DeliveryNoteLine.dn_id)
                .filter(DeliveryNote.dn_no.in_(refs)).all())
        for l in rows:
            s = l.requested_qty - l.sent_qty
            if s > 0:
                short[l.product_id] = short.get(l.product_id, 0) + s
    by_product = {i.product_id: i for i in bo.items}
    for pid, qty in short.items():
        it = by_product.get(pid)
        if it:
            it.qty_ordered = qty
        else:
            _add_item(db, bo, db.query(Product).get(pid), qty)
    for pid, it in by_product.items():
        if pid not in short:
            db.delete(it)
    db.flush()


def attach_dispatch(db: Session, dn: DeliveryNote, *,
                    user_id: Optional[int] = None) -> BackOrder:
    """Load this dispatch note's shortfall onto the branch's OPEN back order.

    The branch has at most one OPEN dispatch back order; every dispatch loads
    onto it until it is closed, after which the next dispatch opens a new one.
    A dispatch with no shortfall still attaches (so it shows in the record).
    """
    branch = dn.branch or db.query(Branch).filter(Branch.id == dn.branch_id).first()
    if not branch:
        raise WMSError("Branch not found.")

    bo = open_dispatch_bo(db, dn.branch_id)
    created = bo is None
    now = _now()
    if created:
        wk = _monday(dn.doc_date or date.today())
        base_no = weekly_bo_no(branch.code, wk)
        bo_no, n = base_no, 1
        while db.query(BackOrder.id).filter(BackOrder.bo_no == bo_no).first():
            n += 1
            bo_no = f"{base_no}-{n}"
        bo = BackOrder(
            bo_no=bo_no, branch_id=dn.branch_id,
            source=BackOrderSource.DISPATCH.value, source_ref=dn.dn_no,
            priority=BackOrderPriority.NORMAL.value,
            stage=BackOrderStage.OPEN.value, status=BackOrderStatus.OPEN.value,
            period_start=wk, sla_days=settings.backorder_sla_days,
            notes=f"Back order for {branch.name} — opened {wk.isoformat()}",
            created_by=user_id, submitted_at=now,
        )
        db.add(bo)
        db.flush()
        db.add(BackOrderEvent(back_order_id=bo.id, from_stage=None,
                              to_stage=BackOrderStage.OPEN.value, at=now,
                              user_id=user_id, note=f"opened by dispatch {dn.dn_no}"))
    else:
        refs = [r for r in (bo.source_ref or "").split(",") if r]
        if dn.dn_no not in refs:
            refs.append(dn.dn_no)
        bo.source_ref = ",".join(refs)

    by_product = {i.product_id: i for i in bo.items}
    short_lines = 0
    for line in dn.lines:
        short = line.requested_qty - line.sent_qty
        if short <= 0:
            continue
        it = by_product.get(line.product_id)
        if it:
            it.qty_ordered += short
        else:
            by_product[line.product_id] = _add_item(db, bo, line.product, short)
        short_lines += 1

    if not created:
        db.add(BackOrderEvent(back_order_id=bo.id, from_stage=bo.stage,
                              to_stage=bo.stage, at=now, user_id=user_id,
                              note=f"loaded dispatch {dn.dn_no} ({short_lines} short line(s))"))
    write_audit(db, entity_type="BackOrder", entity_id=bo.id,
                action="OPEN_BACKORDER" if created else "LOAD_DISPATCH",
                detail={"bo_no": bo.bo_no, "dn_no": dn.dn_no, "short_lines": short_lines},
                user_id=user_id)
    db.flush()
    return bo


def reverse_dispatch(db: Session, dn_no: str, *, user_id: Optional[int] = None) -> dict:
    """Undo every effect of one dispatch note: pull its *sent* units back out of
    branch stock, delete the sales it recorded, unload its shortfall from the
    branch's back order (deleting the back order if it becomes empty and this
    dispatch opened it), then delete the dispatch note itself.

    Refused once the back order it fed has been closed.
    """
    from wms.services import stock as stock_svc

    dn = db.query(DeliveryNote).filter(DeliveryNote.dn_no == dn_no).first()
    if not dn:
        raise WMSError("Dispatch not found.", 404)

    bo = (db.query(BackOrder)
          .filter(BackOrder.source == BackOrderSource.DISPATCH.value,
                  BackOrder.branch_id == dn.branch_id)
          .filter(BackOrder.source_ref.like(f"%{dn_no}%"))
          .order_by(BackOrder.id).all())
    bo = next((b for b in bo if dn_no in (b.source_ref or "").split(",")), None)
    if bo and bo.status != BackOrderStatus.OPEN.value:
        raise WMSError(f"Back order {bo.bo_no} is closed — this dispatch can no "
                       f"longer be reversed.")

    branch = dn.branch
    lines = [(l.product_id, l.product.sku if l.product else None,
              l.requested_qty, l.sent_qty) for l in dn.lines]

    # 1. stock: subtract the sent quantities
    pulled = 0
    for _pid, sku, _req, sent in lines:
        if sku and sent > 0:
            stock_svc.add_stock(db, branch_id=dn.branch_id,
                                items=[{"sku": sku, "qty": -int(sent)}],
                                user_id=user_id, commit=False)
            pulled += sent

    # 2. sales recorded from this dispatch
    sales_deleted = (db.query(SalesRecord)
                     .filter(SalesRecord.source_ref == dn_no)
                     .delete(synchronize_session=False))

    # 3. back order: detach this note and rebuild / drop
    bo_result = None
    if bo:
        refs = [r for r in (bo.source_ref or "").split(",") if r and r != dn_no]
        opened_by_this = not refs or (bo.source_ref or "").split(",")[0] == dn_no
        if not refs:
            db.delete(bo)
            bo_result = f"deleted {bo.bo_no}"
        else:
            bo.source_ref = ",".join(refs)
            _rebuild_dispatch_items(db, bo)
            db.add(BackOrderEvent(back_order_id=bo.id, from_stage=bo.stage,
                                  to_stage=bo.stage, at=_now(), user_id=user_id,
                                  note=f"reversed dispatch {dn_no}"))
            bo_result = f"updated {bo.bo_no}"
        _ = opened_by_this

    # 4. the dispatch note itself (+ its lines, cascade)
    db.delete(dn)
    write_audit(db, entity_type="DeliveryNote", entity_id=dn_no, action="REVERSE",
                detail={"dn_no": dn_no, "branch": branch.name if branch else None,
                        "units_pulled": pulled, "sales_deleted": sales_deleted,
                        "back_order": bo_result}, user_id=user_id)
    db.commit()
    return {"dn_no": dn_no, "units_pulled": pulled, "sales_deleted": sales_deleted,
            "back_order": bo_result}


def from_delivery_note(db: Session, dn: DeliveryNote, *,
                       user_id: Optional[int] = None) -> Optional[BackOrder]:
    """Group every shortfall line on ``dn`` into one OPEN back order.
    Returns None when the note was fully filled."""
    shorts = [(l, l.requested_qty - l.sent_qty) for l in dn.lines
              if l.requested_qty - l.sent_qty > 0]
    if not shorts:
        return None

    # replace any prior auto back order for this DN that is still open
    prior = (db.query(BackOrder)
             .filter(BackOrder.source == BackOrderSource.DELIVERY_NOTE.value,
                     BackOrder.source_ref == dn.dn_no,
                     BackOrder.stage == BackOrderStage.OPEN.value)
             .first())
    if prior:
        db.delete(prior)
        db.flush()

    bo = _new_header(db, branch_id=dn.branch_id,
                     source=BackOrderSource.DELIVERY_NOTE.value, source_ref=dn.dn_no,
                     priority=BackOrderPriority.NORMAL.value,
                     notes=f"Auto from delivery note {dn.dn_no}",
                     sla_days=None, user_id=user_id)
    for line, short in shorts:
        _add_item(db, bo, line.product, short)
    write_audit(db, entity_type="BackOrder", entity_id=bo.id, action="CREATE_FROM_DN",
                detail={"bo_no": bo.bo_no, "dn_no": dn.dn_no, "items": len(shorts)},
                user_id=user_id)
    db.flush()
    return bo


def add_item(db: Session, *, bo_no: str, sku: str, qty: int,
             user_id: Optional[int] = None) -> BackOrder:
    bo = get(db, bo_no)
    if bo.stage != BackOrderStage.OPEN.value:
        raise WMSError("Items can only be added while the back order is Open.")
    _add_item(db, bo, _product(db, sku, user_id=user_id), qty)
    write_audit(db, entity_type="BackOrder", entity_id=bo.id, action="ADD_ITEM",
                detail={"sku": sku, "qty": qty}, user_id=user_id)
    db.commit()
    db.refresh(bo)
    return bo


def get(db: Session, bo_no: str) -> BackOrder:
    bo = db.query(BackOrder).filter(BackOrder.bo_no == bo_no).first()
    if not bo:
        raise WMSError("Back order not found.", 404)
    return bo


def list_back_orders(
    db: Session, *,
    branch_id: Optional[int] = None,
    stage: Optional[str] = None,
    status: Optional[str] = None,
    open_only: bool = False,
    q: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[BackOrder]:
    query = db.query(BackOrder)
    if branch_id:
        query = query.filter(BackOrder.branch_id == branch_id)
    if stage:
        query = query.filter(BackOrder.stage == stage)
    if status:
        query = query.filter(BackOrder.status == status)
    if open_only:
        query = query.filter(BackOrder.status == BackOrderStatus.OPEN.value)
    if q:
        like = f"%{q}%"
        query = query.filter(
            BackOrder.bo_no.ilike(like) | BackOrder.po_no.ilike(like) |
            BackOrder.requisition_no.ilike(like) | BackOrder.source_ref.ilike(like))
    if date_from:
        query = query.filter(func.date(BackOrder.submitted_at) >= date_from)
    if date_to:
        query = query.filter(func.date(BackOrder.submitted_at) <= date_to)
    return query.order_by(BackOrder.id.desc()).all()


def serialize(bo: BackOrder) -> dict:
    refs = [r for r in (bo.source_ref or "").split(",") if r]
    period_label = ""
    if bo.period_start:
        iso = bo.period_start.isocalendar()
        period_label = f"week {iso[1]} · {bo.period_start.isoformat()}"
    return {
        "bo_no": bo.bo_no, "branch": bo.branch.name if bo.branch else bo.branch_id,
        "branch_id": bo.branch_id, "stage": bo.stage, "status": bo.status,
        "priority": bo.priority, "cycle": bo.cycle or "WEEKLY",
        "source": bo.source, "source_ref": bo.source_ref, "source_refs": refs,
        "period_start": bo.period_start, "period_label": period_label,
        "line_count": len(bo.items),
        "requisition_no": bo.requisition_no, "po_no": bo.po_no, "supplier": bo.supplier,
        "expected_date": bo.expected_date, "notes": bo.notes,
        "submitted_at": bo.submitted_at, "closed_at": bo.closed_at,
        "qty_ordered": bo.qty_ordered, "qty_fulfilled": bo.qty_fulfilled,
        "fulfil_pct": bo.fulfil_pct, "value_outstanding": round(bo.value_outstanding, 2),
        "items": [{
            "item_id": i.id, "product_id": i.product_id,
            "sku": i.sku, "description": i.description,
            "category": i.category, "unit_price": float(i.unit_price or 0),
            "qty_ordered": i.qty_ordered, "qty_approved": i.qty_approved,
            "qty_on_po": i.qty_on_po, "qty_received": i.qty_received,
            "qty_allocated": i.qty_allocated, "qty_dispatched": i.qty_dispatched,
            "qty_fulfilled": i.qty_fulfilled, "outstanding_qty": i.outstanding_qty,
            "fill_status": i.fill_status,
        } for i in bo.items],
        "events": [{
            "from": e.from_stage, "to": e.to_stage, "at": e.at, "note": e.note,
        } for e in bo.events],
    }
