"""Pydantic request bodies for the API."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class DNLineIn(BaseModel):
    sku: Optional[str] = None
    product_id: Optional[int] = None
    requested_qty: int = Field(gt=0)
    sent_qty: int = Field(default=0, ge=0)


class DNIn(BaseModel):
    branch_id: int
    from_location_label: str = "DC"
    doc_no: Optional[str] = None
    doc_date: Optional[date] = None
    comment: Optional[str] = None
    lines: list[DNLineIn]


class BackOrderItemIn(BaseModel):
    sku: Optional[str] = None
    product_id: Optional[int] = None
    qty: int = Field(gt=0)


class BackOrderIn(BaseModel):
    branch_id: int
    items: list[BackOrderItemIn]
    priority: str = "NORMAL"
    notes: Optional[str] = None
    sla_days: Optional[int] = None


class BackOrderAdvanceIn(BaseModel):
    to_stage: str
    items: Optional[dict[int, int]] = None       # back_order_item_id -> qty
    requisition_no: Optional[str] = None
    po_no: Optional[str] = None
    supplier: Optional[str] = None
    expected_date: Optional[date] = None
    note: Optional[str] = None


class BackOrderCancelIn(BaseModel):
    reason: str = Field(min_length=2)
