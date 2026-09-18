"""Delivery notes + the backorder processing flow (entry, stages, analysis)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from wms.api.deps import current_user, db_session
from wms.api.schemas import DNIn, BackOrderIn, BackOrderAdvanceIn
from wms.analytics import backorder_flow as bof
from wms.analytics import loaders
from wms.models import DeliveryNote, User
from wms.services import backorders as dn_svc
from wms.services import backorder_entry as bo_entry
from wms.services import backorder_stages as bo_stage

router = APIRouter(prefix="/api/outbound", tags=["outbound / backorders"])


import numpy as np


def _recs(df):
    if df is None or df.empty:
        return []
    d = df.replace([np.inf, -np.inf], np.nan)
    return d.astype(object).where(d.notna(), None).to_dict("records")


# ---- Delivery notes (requested-vs-sent source document) ----------------
@router.post("/delivery-notes", status_code=201)
def create_dn(body: DNIn, db: Session = Depends(db_session), u: User = Depends(current_user)):
    dn = dn_svc.enter_delivery_note(
        db, branch_id=body.branch_id, from_location_label=body.from_location_label,
        doc_no=body.doc_no, doc_date=body.doc_date, comment=body.comment,
        lines=[l.model_dump() for l in body.lines], user_id=u.id)
    return dn_svc.serialize(db, dn)


@router.get("/delivery-notes/{dn_no}")
def get_dn(dn_no: str, db: Session = Depends(db_session), _: User = Depends(current_user)):
    dn = db.query(DeliveryNote).filter(DeliveryNote.dn_no == dn_no).first()
    if not dn:
        raise HTTPException(404, "Delivery note not found.")
    return dn_svc.serialize(db, dn)


# ---- Back orders: entry + flow -----------------------------------------
@router.post("/back-orders", status_code=201)
def create_back_order(body: BackOrderIn, db: Session = Depends(db_session),
                      u: User = Depends(current_user)):
    bo = bo_entry.create_back_order(
        db, branch_id=body.branch_id, items=[i.model_dump() for i in body.items],
        priority=body.priority, notes=body.notes, sla_days=body.sla_days, user_id=u.id)
    return bo_entry.serialize(bo)


@router.get("/back-orders")
def list_back_orders(branch_id: int | None = None, stage: str | None = None,
                     status: str | None = None, open_only: bool = False,
                     q: str | None = None, db: Session = Depends(db_session),
                     _: User = Depends(current_user)):
    rows = bo_entry.list_back_orders(db, branch_id=branch_id, stage=stage,
                                     status=status, open_only=open_only, q=q)
    return [bo_entry.serialize(b) for b in rows]


@router.get("/back-orders/{bo_no}")
def get_back_order(bo_no: str, db: Session = Depends(db_session),
                   _: User = Depends(current_user)):
    return bo_entry.serialize(bo_entry.get(db, bo_no))


@router.get("/back-orders/{bo_no}/next-stages")
def next_stages(bo_no: str, db: Session = Depends(db_session),
                _: User = Depends(current_user)):
    return {"stage": bo_entry.get(db, bo_no).stage,
            "allowed": bo_stage.allowed_next(bo_entry.get(db, bo_no).stage)}


@router.post("/back-orders/{bo_no}/advance")
def advance(bo_no: str, body: BackOrderAdvanceIn, db: Session = Depends(db_session),
            u: User = Depends(current_user)):
    bo = bo_stage.advance(
        db, bo_no=bo_no, to_stage=body.to_stage, items=body.items,
        requisition_no=body.requisition_no, po_no=body.po_no, supplier=body.supplier,
        expected_date=body.expected_date, note=body.note, user_id=u.id)
    return bo_entry.serialize(bo)


# ---- Back-order flow analysis ----------------------------------------------
@router.get("/back-orders-analysis")
def analysis(branch_id: int | None = None, db: Session = Depends(db_session),
             _: User = Depends(current_user)):
    bo = loaders.back_orders_df(db, branch_id=branch_id)
    items = loaders.back_order_items_df(db, branch_id=branch_id)
    events = loaders.back_order_events_df(db)
    return {
        "fulfilment": bof.fulfilment_metrics(bo, items),
        "bottleneck_stage": bof.bottleneck_stage(events),
        "stage_funnel": _recs(bof.stage_funnel(bo)),
        "cycle_times": _recs(bof.cycle_times(events)),
        "aging": _recs(bof.aging(bo)),
        "aging_by_stage": _recs(bof.aging_by_stage(bo)),
        "by_branch": _recs(bof.by_branch(bo, items)),
        "top_items": _recs(bof.top_items(items)),
        "branch_vs_sales": _recs(bof.branch_backorders_vs_sales(db)),
        "trend": _recs(bof.trend(bo, events)),
    }
