"""SQL -> pandas DataFrame loaders."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from wms.enums import STAGE_LABEL, BackOrderStage
from wms.models import (
    Branch, BackOrder, BackOrderItem, BackOrderEvent, DeliveryNote,
    DeliveryNoteLine, Product, SalesRecord,
)


def dn_lines_df(db: Session, *, branch_id: Optional[int] = None,
                date_from: Optional[date] = None,
                date_to: Optional[date] = None) -> pd.DataFrame:
    """Every line of every delivery note, with requested / sent / backorder."""
    q = (db.query(DeliveryNote, DeliveryNoteLine, Product, Branch)
           .join(DeliveryNoteLine, DeliveryNoteLine.dn_id == DeliveryNote.id)
           .join(Product, Product.id == DeliveryNoteLine.product_id)
           .join(Branch, Branch.id == DeliveryNote.branch_id))
    if branch_id:
        q = q.filter(DeliveryNote.branch_id == branch_id)
    if date_from:
        q = q.filter(DeliveryNote.doc_date >= date_from)
    if date_to:
        q = q.filter(DeliveryNote.doc_date <= date_to)

    today = date.today()
    rows = []
    for dn, l, p, b in q.all():
        req, sent = l.requested_qty or 0, l.sent_qty or 0
        bo = max(0, req - sent)
        price = float(p.unit_price) if p.unit_price is not None else None
        d = dn.doc_date or (dn.created_at.date() if dn.created_at else today)
        rows.append({
            "dn_id": dn.id, "dn_no": dn.dn_no, "doc_date": pd.Timestamp(d),
            "from_location": dn.from_location_label,
            "branch_id": b.id, "branch": b.name,
            "sku": p.sku, "product_id": p.id, "description": p.name, "category": p.category,
            "requested_qty": req, "sent_qty": sent, "backorder_qty": bo,
            "fill_status": "FULL" if bo == 0 else ("NIL" if sent == 0 else "PARTIAL"),
            "unit_price": price,
            "backorder_value": (bo * price) if price is not None else None,
            "age_days": (today - d).days,
        })
    cols = ["dn_id", "dn_no", "doc_date", "from_location", "branch_id", "branch", "sku",
            "product_id", "description", "category", "requested_qty", "sent_qty",
            "backorder_qty", "fill_status", "unit_price", "backorder_value", "age_days"]
    return pd.DataFrame(rows, columns=cols)


def back_orders_df(db: Session, *, branch_id: Optional[int] = None,
                   stage: Optional[str] = None, status: Optional[str] = None) -> pd.DataFrame:
    """One row per back order (header) with fulfilment + timing fields."""
    q = db.query(BackOrder)
    if branch_id:
        q = q.filter(BackOrder.branch_id == branch_id)
    if stage:
        q = q.filter(BackOrder.stage == stage)
    if status:
        q = q.filter(BackOrder.status == status)
    br = {b.id: b.name for b in db.query(Branch).all()}
    now = pd.Timestamp(datetime.utcnow())
    rows = []
    for bo in q.all():
        sub = pd.Timestamp(bo.submitted_at) if bo.submitted_at else pd.NaT
        end = (pd.Timestamp(bo.closed_at) if bo.closed_at else
               pd.Timestamp(bo.cancelled_at) if bo.cancelled_at else pd.NaT)
        age = (now - sub).days if pd.notna(sub) else None
        lead = (end - sub).days if pd.notna(sub) and pd.notna(end) else None
        ordered = bo.qty_ordered
        rows.append({
            "bo_no": bo.bo_no, "branch_id": bo.branch_id,
            "branch": br.get(bo.branch_id, str(bo.branch_id)),
            "stage": bo.stage, "stage_label": STAGE_LABEL.get(BackOrderStage(bo.stage), bo.stage),
            "status": bo.status, "priority": bo.priority, "source": bo.source,
            "source_ref": bo.source_ref, "po_no": bo.po_no, "supplier": bo.supplier,
            "submitted_at": sub, "closed_at": end,
            "items": len(bo.items),
            "qty_ordered": ordered, "qty_fulfilled": bo.qty_fulfilled,
            "outstanding_qty": ordered - bo.qty_fulfilled,
            "fulfil_pct": bo.fulfil_pct,
            "value_outstanding": round(bo.value_outstanding, 2),
            "age_days": age, "lead_time_days": lead, "sla_days": bo.sla_days,
            "is_overdue": bool(bo.status == "OPEN" and bo.sla_days is not None
                               and age is not None and age > bo.sla_days),
        })
    return pd.DataFrame(rows)


def back_order_items_df(db: Session, *, branch_id: Optional[int] = None) -> pd.DataFrame:
    q = (db.query(BackOrder, BackOrderItem)
           .join(BackOrderItem, BackOrderItem.back_order_id == BackOrder.id))
    if branch_id:
        q = q.filter(BackOrder.branch_id == branch_id)
    br = {b.id: b.name for b in db.query(Branch).all()}
    rows = []
    for bo, it in q.all():
        price = float(it.unit_price) if it.unit_price is not None else None
        rows.append({
            "bo_no": bo.bo_no, "branch_id": bo.branch_id,
            "branch": br.get(bo.branch_id, str(bo.branch_id)),
            "stage": bo.stage, "status": bo.status,
            "sku": it.sku, "description": it.description, "category": it.category,
            "qty_ordered": it.qty_ordered, "qty_approved": it.qty_approved,
            "qty_on_po": it.qty_on_po, "qty_received": it.qty_received,
            "qty_allocated": it.qty_allocated, "qty_dispatched": it.qty_dispatched,
            "qty_fulfilled": it.qty_fulfilled, "outstanding_qty": it.outstanding_qty,
            "fill_status": it.fill_status, "unit_price": price,
            "value_ordered": (it.qty_ordered * price) if price is not None else None,
            "value_outstanding": (it.outstanding_qty * price) if price is not None else None,
        })
    return pd.DataFrame(rows)


def back_order_events_df(db: Session) -> pd.DataFrame:
    q = (db.query(BackOrder, BackOrderEvent)
           .join(BackOrderEvent, BackOrderEvent.back_order_id == BackOrder.id))
    br = {b.id: b.name for b in db.query(Branch).all()}
    rows = []
    for bo, e in q.all():
        rows.append({
            "bo_no": bo.bo_no, "branch_id": bo.branch_id,
            "branch": br.get(bo.branch_id, str(bo.branch_id)),
            "from_stage": e.from_stage, "to_stage": e.to_stage,
            "at": pd.Timestamp(e.at), "note": e.note,
        })
    df = pd.DataFrame(rows, columns=["bo_no", "branch_id", "branch", "from_stage",
                                     "to_stage", "at", "note"])
    return df.sort_values(["bo_no", "at"]) if not df.empty else df


def demand_df(db: Session, *, days: int = 90) -> pd.DataFrame:
    """Branch demand history for forecasting / branch-sales analysis:
    delivery-note *requested* qty + explicit sales records (which include the
    delivery-note *sent* qty, source DELIVERY)."""
    cutoff = date.today() - timedelta(days=days)
    rows = []
    dl = dn_lines_df(db, date_from=cutoff)
    for _, r in dl.iterrows():
        rows.append({"date": r["doc_date"], "branch_id": r["branch_id"], "branch": r["branch"],
                     "product_id": r["product_id"], "sku": r["sku"],
                     "qty": r["requested_qty"], "source": "DN_REQUEST"})
    br = {b.id: b.name for b in db.query(Branch).all()}
    pr = {p.id: p.sku for p in db.query(Product).all()}
    for s in db.query(SalesRecord).filter(SalesRecord.sale_date >= cutoff).all():
        rows.append({"date": pd.Timestamp(s.sale_date), "branch_id": s.branch_id,
                     "branch": br.get(s.branch_id), "product_id": s.product_id,
                     "sku": pr.get(s.product_id), "qty": s.qty, "source": s.source})
    return pd.DataFrame(rows, columns=["date", "branch_id", "branch", "product_id",
                                       "sku", "qty", "source"])


def sales_df(db: Session, *, days: int = 180, branch_id: Optional[int] = None) -> pd.DataFrame:
    """Explicit branch sales only (excludes the DN_REQUEST demand signal)."""
    cutoff = date.today() - timedelta(days=days)
    q = db.query(SalesRecord).filter(SalesRecord.sale_date >= cutoff)
    if branch_id:
        q = q.filter(SalesRecord.branch_id == branch_id)
    br = {b.id: b.name for b in db.query(Branch).all()}
    pr = {p.id: p for p in db.query(Product).all()}
    rows = []
    for s in q.all():
        p = pr.get(s.product_id)
        price = float(s.unit_price) if s.unit_price is not None else (
            float(p.unit_price) if p and p.unit_price is not None else None)
        rows.append({
            "date": pd.Timestamp(s.sale_date), "branch_id": s.branch_id,
            "branch": br.get(s.branch_id), "product_id": s.product_id,
            "sku": p.sku if p else str(s.product_id),
            "category": p.category if p else None,
            "qty": s.qty, "unit_price": price,
            "value": (s.qty * price) if price is not None else None,
            "source": s.source,
        })
    return pd.DataFrame(rows, columns=["date", "branch_id", "branch", "product_id", "sku",
                                       "category", "qty", "unit_price", "value", "source"])
