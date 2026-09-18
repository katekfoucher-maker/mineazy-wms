"""Backorder processing flow - the stage machine.

    OPEN -> CLOSED

Workflow-only: transitions record quantities + timestamps + an event row; they
do NOT post stock movements (that stays a separate, manual step).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.enums import (
    STAGE_ITEM_QTY, STAGE_NEXT, STAGE_TIMESTAMP, BackOrderStage, BackOrderStatus,
)
from wms.errors import WMSError
from wms.models import BackOrder

# quantity carried forward into a stage when the caller does not specify one
_PREV_QTY = {
    BackOrderStage.CLOSED: "qty_ordered",
}

_TERMINAL = {BackOrderStage.CLOSED.value}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def allowed_next(stage: str) -> list[str]:
    if stage in _TERMINAL:
        return []
    if stage not in BackOrderStage._value2member_map_:      # legacy stage value
        return [BackOrderStage.CLOSED.value]
    return [s.value for s in STAGE_NEXT.get(BackOrderStage(stage), set())]


def advance(
    db: Session, *,
    bo_no: str,
    to_stage: str,
    items: Optional[dict[int, int]] = None,   # {item_id: qty} for the stage's qty field
    requisition_no: Optional[str] = None,
    po_no: Optional[str] = None,
    supplier: Optional[str] = None,
    expected_date: Optional[date] = None,
    note: Optional[str] = None,
    user_id: Optional[int] = None,
) -> BackOrder:
    bo = db.query(BackOrder).filter(BackOrder.bo_no == bo_no).first()
    if not bo:
        raise WMSError("Back order not found.", 404)
    if bo.stage in _TERMINAL:
        raise WMSError(f"Back order {bo_no} is already {bo.stage}.")

    target = BackOrderStage(to_stage) if to_stage in BackOrderStage._value2member_map_ \
        else None
    if target is None:
        raise WMSError(f"Unknown stage '{to_stage}'.")
    if to_stage not in allowed_next(bo.stage):
        raise WMSError(f"Cannot move {bo_no} from {bo.stage} to {to_stage}. "
                       f"Allowed: {', '.join(allowed_next(bo.stage))}.")

    # optional references (kept for reporting; no stage requires them now)
    bo.requisition_no = requisition_no or bo.requisition_no
    bo.po_no = po_no or bo.po_no
    bo.supplier = supplier or bo.supplier
    bo.expected_date = expected_date or bo.expected_date

    # record the quantity this stage tracks
    qty_field = STAGE_ITEM_QTY.get(target)
    if qty_field:
        prev_field = _PREV_QTY[target]
        for it in bo.items:
            if items and it.id in items:
                v = int(items[it.id])
                if v < 0:
                    raise WMSError("Quantity cannot be negative.")
                cap = getattr(it, prev_field)
                if v > cap:
                    raise WMSError(f"{it.sku}: {qty_field} ({v}) exceeds {prev_field} ({cap}).")
                setattr(it, qty_field, v)
            else:
                setattr(it, qty_field, getattr(it, prev_field))

    frm = bo.stage
    bo.stage = to_stage
    ts = _now()
    setattr(bo, STAGE_TIMESTAMP[target], ts)
    if target is BackOrderStage.CLOSED:
        bo.status = BackOrderStatus.CLOSED.value

    from wms.models import BackOrderEvent
    db.add(BackOrderEvent(back_order_id=bo.id, from_stage=frm, to_stage=to_stage,
                          at=ts, user_id=user_id, note=note))
    write_audit(db, entity_type="BackOrder", entity_id=bo.id, action="ADVANCE",
                detail={"bo_no": bo.bo_no, "from": frm, "to": to_stage},
                reason=note, user_id=user_id)
    db.commit()
    db.refresh(bo)
    return bo
