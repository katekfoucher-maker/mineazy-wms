"""Enumerations shared across models, services and analytics."""
from __future__ import annotations

from enum import Enum


class FillStatus(str, Enum):
    FULL = "FULL"        # sent >= requested
    PARTIAL = "PARTIAL"  # 0 < sent < requested
    NIL = "NIL"          # sent == 0


# ---- Backorder processing flow -------------------------------------------
# A back order tracks how the branch's requirement is *being entered*: it stays
# OPEN while dispatch notes are still being loaded onto it, then CLOSED.
class BackOrderStage(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


# order shown in the "stage funnel" and used for aging
STAGE_SEQUENCE = [BackOrderStage.OPEN, BackOrderStage.CLOSED]

STAGE_LABEL = {
    BackOrderStage.OPEN: "Open",
    BackOrderStage.CLOSED: "Closed",
}

# timestamp column stamped when a back order enters each stage
STAGE_TIMESTAMP = {
    BackOrderStage.OPEN: "submitted_at",
    BackOrderStage.CLOSED: "closed_at",
}

# allowed forward transitions
STAGE_NEXT = {
    BackOrderStage.OPEN: {BackOrderStage.CLOSED},
    BackOrderStage.CLOSED: set(),
}

# which item quantity a stage records (default carries the previous one forward)
STAGE_ITEM_QTY = {
    BackOrderStage.CLOSED: "qty_fulfilled",
}


class BackOrderStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class BackOrderPriority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class BackOrderSource(str, Enum):
    MANUAL = "MANUAL"
    DELIVERY_NOTE = "DELIVERY_NOTE"
    DISPATCH = "DISPATCH"           # weekly bucket fed by dispatch confirmations


class BackOrderCycle(str, Enum):
    """How often the branch expects to re-raise this requirement."""
    WEEKLY = "WEEKLY"
    FORTNIGHTLY = "FORTNIGHTLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"


CYCLE_DEFAULT = BackOrderCycle.WEEKLY.value

CYCLE_LABEL = {
    BackOrderCycle.WEEKLY: "Weekly",
    BackOrderCycle.FORTNIGHTLY: "Fortnightly",
    BackOrderCycle.MONTHLY: "Monthly",
    BackOrderCycle.QUARTERLY: "Quarterly",
}


class SalesSource(str, Enum):
    POS = "POS"
    DELIVERY = "DELIVERY"     # the sent qty on a delivery note
    IMPORT = "IMPORT"
