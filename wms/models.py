"""SQLAlchemy ORM models.

Domains: reference data (users, branches, products) · delivery notes (the
requested-vs-sent source document) · backorder processing flow · sales history ·
audit.  There is no stock ledger - see README ("remove inventory module").
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean, Column, Date, DateTime, ForeignKey, Index, Integer, Numeric,
    String, Text, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from wms.db import TimestampedBase


# ======================================================================
# REFERENCE DATA
# ======================================================================
class User(TimestampedBase):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("username", name="uq_user_username"),)

    username = Column(String(60), nullable=False)
    full_name = Column(String(150), nullable=False)
    role = Column(String(40), nullable=False, default="clerk")
    password_hash = Column(String(255), nullable=True)
    branch_id = Column(Integer, ForeignKey("branches.id", ondelete="SET NULL"), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)


class Branch(TimestampedBase):
    __tablename__ = "branches"
    __table_args__ = (UniqueConstraint("code", name="uq_branch_code"),)

    code = Column(String(20), nullable=False)
    name = Column(String(120), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)


class Product(TimestampedBase):
    __tablename__ = "products"
    __table_args__ = (
        UniqueConstraint("sku", name="uq_product_sku"),
        Index("ix_product_category", "category"),
    )

    sku = Column(String(50), nullable=False)
    name = Column(String(200), nullable=False)
    category = Column(String(80), nullable=True)
    uom = Column(String(20), nullable=False, default="EA")
    unit_price = Column(Numeric(14, 2), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)


# ======================================================================
# DELIVERY NOTE  -  the requested-vs-sent source document
#   sent qty  -> recorded as branch sales
#   shortfall -> raises a BackOrder
# ======================================================================
class DeliveryNote(TimestampedBase):
    __tablename__ = "delivery_notes"
    __table_args__ = (
        UniqueConstraint("dn_no", name="uq_dn_no"),
        Index("ix_dn_branch", "branch_id"),
        Index("ix_dn_date", "doc_date"),
    )

    dn_no = Column(String(50), nullable=False)
    branch_id = Column(Integer, ForeignKey("branches.id", ondelete="RESTRICT"), nullable=False)
    from_location_label = Column(String(60), nullable=True)   # free text e.g. "DC"
    doc_date = Column(Date, nullable=True)
    comment = Column(String(255), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    branch = relationship("Branch", lazy="joined")
    lines = relationship("DeliveryNoteLine", back_populates="dn",
                         cascade="all, delete-orphan", lazy="selectin")

    @property
    def total_requested(self) -> int:
        return sum(l.requested_qty for l in self.lines)

    @property
    def total_sent(self) -> int:
        return sum(l.sent_qty for l in self.lines)


class DeliveryNoteLine(TimestampedBase):
    __tablename__ = "delivery_note_lines"
    __table_args__ = (Index("ix_dnl_dn", "dn_id"),)

    dn_id = Column(Integer, ForeignKey("delivery_notes.id", ondelete="CASCADE"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False)
    requested_qty = Column(Integer, nullable=False)
    sent_qty = Column(Integer, nullable=False, default=0)      # blank on paper == 0

    dn = relationship("DeliveryNote", back_populates="lines")
    product = relationship("Product", lazy="joined")

    @property
    def backorder_qty(self) -> int:
        return max(0, self.requested_qty - self.sent_qty)


# ======================================================================
# RECEIVING ORDER  -  stock arriving into the warehouse (usually the
# distribution centre). Confirming one adds every line's received qty onto
# that location's stock_on_hand balance (see wms.services.receiving) -
# the mirror image of a dispatch, which subtracts nothing but records what
# left the warehouse; receiving only ever adds.
# ======================================================================
class ReceivingOrder(TimestampedBase):
    __tablename__ = "receiving_orders"
    __table_args__ = (
        UniqueConstraint("ro_no", name="uq_ro_no"),
        Index("ix_ro_branch", "branch_id"),
        Index("ix_ro_date", "doc_date"),
    )

    ro_no = Column(String(50), nullable=False)
    branch_id = Column(Integer, ForeignKey("branches.id", ondelete="RESTRICT"), nullable=False)
    supplier = Column(String(120), nullable=True)
    doc_date = Column(Date, nullable=True)
    comment = Column(String(255), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    branch = relationship("Branch", lazy="joined")
    lines = relationship("ReceivingLine", back_populates="ro",
                         cascade="all, delete-orphan", lazy="selectin")

    @property
    def total_received(self) -> int:
        return sum(l.received_qty for l in self.lines)


class ReceivingLine(TimestampedBase):
    __tablename__ = "receiving_lines"
    __table_args__ = (Index("ix_rol_ro", "ro_id"),)

    ro_id = Column(Integer, ForeignKey("receiving_orders.id", ondelete="CASCADE"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False)
    received_qty = Column(Integer, nullable=False)

    ro = relationship("ReceivingOrder", back_populates="lines")
    product = relationship("Product", lazy="joined")


# ======================================================================
# DISPATCH ORDER (Recon)  -  stock leaving the warehouse for a branch. The
# mirror image of a receiving order: confirming one subtracts every line's
# dispatched qty from the warehouse's stock_on_hand balance and adds it onto
# the destination branch's (see wms.services.dispatch) - purely a stock
# movement record for an accurate warehouse balance. No requested-vs-sent
# shortfall and no back order - see DeliveryNote/BackOrder for that older,
# separate flow, which this does not touch.
# ======================================================================
class DispatchOrder(TimestampedBase):
    __tablename__ = "dispatch_orders"
    __table_args__ = (
        UniqueConstraint("do_no", name="uq_do_no"),
        Index("ix_do_branch", "branch_id"),
        Index("ix_do_date", "doc_date"),
    )

    do_no = Column(String(50), nullable=False)
    branch_id = Column(Integer, ForeignKey("branches.id", ondelete="RESTRICT"), nullable=False)
    doc_date = Column(Date, nullable=True)
    comment = Column(String(255), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    branch = relationship("Branch", lazy="joined")
    lines = relationship("DispatchLine", back_populates="do",
                         cascade="all, delete-orphan", lazy="selectin")

    @property
    def total_dispatched(self) -> int:
        return sum(l.dispatched_qty for l in self.lines)


class DispatchLine(TimestampedBase):
    __tablename__ = "dispatch_lines"
    __table_args__ = (Index("ix_dol_do", "do_id"),)

    do_id = Column(Integer, ForeignKey("dispatch_orders.id", ondelete="CASCADE"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False)
    dispatched_qty = Column(Integer, nullable=False)

    do = relationship("DispatchOrder", back_populates="lines")
    product = relationship("Product", lazy="joined")


# ======================================================================
# BACKORDER PROCESSING FLOW
#   BackOrder (header) --< BackOrderItem (lines)
#   BackOrderEvent logs every stage transition (raw material for cycle times)
# ======================================================================
class BackOrder(TimestampedBase):
    __tablename__ = "back_orders"
    __table_args__ = (
        UniqueConstraint("bo_no", name="uq_bo_no"),
        Index("ix_bo_branch_stage", "branch_id", "stage"),
        Index("ix_bo_status", "status"),
        Index("ix_bo_submitted", "submitted_at"),
    )

    bo_no = Column(String(40), nullable=False)
    branch_id = Column(Integer, ForeignKey("branches.id", ondelete="RESTRICT"), nullable=False)
    source = Column(String(16), nullable=False, default="MANUAL")       # BackOrderSource
    source_ref = Column(String(255), nullable=True)                     # originating DN no(s), comma-sep
    period_start = Column(Date, nullable=True)                          # Monday of the ISO week (weekly bucket)
    priority = Column(String(10), nullable=False, default="NORMAL")     # BackOrderPriority
    cycle = Column(String(12), nullable=False, default="WEEKLY")       # BackOrderCycle
    stage = Column(String(24), nullable=False, default="OPEN")          # BackOrderStage
    status = Column(String(12), nullable=False, default="OPEN")         # BackOrderStatus
    sla_days = Column(Integer, nullable=True)

    # procurement references captured as the flow progresses
    requisition_no = Column(String(50), nullable=True)
    po_no = Column(String(50), nullable=True)
    supplier = Column(String(150), nullable=True)
    expected_date = Column(Date, nullable=True)
    notes = Column(String(500), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # one timestamp per stage entered
    submitted_at = Column(DateTime, nullable=True)
    review_at = Column(DateTime, nullable=True)
    procurement_at = Column(DateTime, nullable=True)
    requisition_at = Column(DateTime, nullable=True)
    po_at = Column(DateTime, nullable=True)
    goods_received_at = Column(DateTime, nullable=True)
    ready_at = Column(DateTime, nullable=True)
    dispatched_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    branch = relationship("Branch", lazy="joined")
    items = relationship("BackOrderItem", back_populates="back_order",
                         cascade="all, delete-orphan", lazy="selectin")
    events = relationship("BackOrderEvent", back_populates="back_order",
                          cascade="all, delete-orphan", lazy="selectin",
                          order_by="BackOrderEvent.id")

    # ---- derived ----
    @property
    def qty_ordered(self) -> int:
        return sum(i.qty_ordered for i in self.items)

    @property
    def qty_fulfilled(self) -> int:
        return sum(i.qty_fulfilled for i in self.items)

    @property
    def fulfil_pct(self) -> float:
        o = self.qty_ordered
        return round(self.qty_fulfilled / o, 4) if o else 0.0

    @property
    def value_outstanding(self):
        return sum((i.qty_ordered - i.qty_fulfilled) * float(i.unit_price or 0)
                   for i in self.items)


class BackOrderItem(TimestampedBase):
    __tablename__ = "back_order_items"
    __table_args__ = (Index("ix_boi_bo", "back_order_id"),
                      Index("ix_boi_product", "product_id"))

    back_order_id = Column(Integer, ForeignKey("back_orders.id", ondelete="CASCADE"),
                           nullable=False)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False)
    sku = Column(String(50), nullable=False)
    description = Column(String(200), nullable=True)
    category = Column(String(80), nullable=True)
    unit_price = Column(Numeric(14, 2), nullable=True)

    qty_ordered = Column(Integer, nullable=False)       # backordered demand
    qty_approved = Column(Integer, nullable=False, default=0)
    qty_on_po = Column(Integer, nullable=False, default=0)
    qty_received = Column(Integer, nullable=False, default=0)
    qty_allocated = Column(Integer, nullable=False, default=0)
    qty_dispatched = Column(Integer, nullable=False, default=0)
    qty_fulfilled = Column(Integer, nullable=False, default=0)

    back_order = relationship("BackOrder", back_populates="items")
    product = relationship("Product", lazy="joined")

    @property
    def outstanding_qty(self) -> int:
        return max(0, self.qty_ordered - self.qty_fulfilled)

    @property
    def fill_status(self) -> str:
        if self.qty_fulfilled >= self.qty_ordered:
            return "FULL"
        return "NIL" if self.qty_fulfilled == 0 else "PARTIAL"


class BackOrderEvent(TimestampedBase):
    __tablename__ = "back_order_events"
    __table_args__ = (Index("ix_boe_bo", "back_order_id"),)

    back_order_id = Column(Integer, ForeignKey("back_orders.id", ondelete="CASCADE"),
                           nullable=False)
    from_stage = Column(String(24), nullable=True)
    to_stage = Column(String(24), nullable=False)
    at = Column(DateTime, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    note = Column(String(255), nullable=True)

    back_order = relationship("BackOrder", back_populates="events")


# ======================================================================
# STOCK ON HAND  -  running per-branch balance
#   a dispatch confirmation ADDS the dispatched qty here; an inventory
#   upload REPLACES a branch's rows.  The Allocation plan reads this.
# ======================================================================
class StockOnHand(TimestampedBase):
    __tablename__ = "stock_on_hand"
    __table_args__ = (
        UniqueConstraint("branch_id", "sku", name="uq_soh_branch_sku"),
        Index("ix_soh_branch", "branch_id"),
    )

    branch_id = Column(Integer, ForeignKey("branches.id", ondelete="CASCADE"), nullable=False)
    sku = Column(String(50), nullable=False)          # free text - matches forecast/upload SKUs
    qty_on_hand = Column(Integer, nullable=False, default=0)

    branch = relationship("Branch", lazy="joined")


# ======================================================================
# SALES HISTORY  (feeds branch analysis + forecasting)
# ======================================================================
class SalesRecord(TimestampedBase):
    __tablename__ = "sales_records"
    __table_args__ = (
        Index("ix_sales_branch_product_date", "branch_id", "product_id", "sale_date"),
        Index("ix_sales_date", "sale_date"),
    )

    branch_id = Column(Integer, ForeignKey("branches.id", ondelete="RESTRICT"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False)
    sale_date = Column(Date, nullable=False)
    qty = Column(Integer, nullable=False)
    unit_price = Column(Numeric(14, 2), nullable=True)
    source = Column(String(16), nullable=False, default="IMPORT")   # SalesSource
    source_ref = Column(String(50), nullable=True)                  # dispatch note no. for DELIVERY sales

    branch = relationship("Branch", lazy="joined")
    product = relationship("Product", lazy="joined")


# ======================================================================
# FILE-UPLOAD-DERIVED ANALYTICS DATA
#   These three used to live purely as uploaded Excel files under data/ -
#   fine on a machine with a real disk, but Render's free-tier filesystem is
#   ephemeral (wiped on every restart/redeploy), so anything that only lived
#   on disk was invisible again a few minutes after being uploaded. Each
#   upload now parses straight into one of these tables instead; branch_code
#   and sku are free text (not FKs), same as StockOnHand above, since a
#   HansaWorld export's codes don't always exactly match the curated
#   Branch/Product tables and shouldn't be rejected for that.
# ======================================================================
class MonthlySalesLine(TimestampedBase):
    """One row per (branch, sku, month) - the 'Item Statistics' monthly
    exports that feed the Sales & Forecasting page's demand model."""
    __tablename__ = "monthly_sales_lines"
    __table_args__ = (
        UniqueConstraint("branch_code", "sku", "period",
                         name="uq_msl_branch_sku_period"),
        Index("ix_msl_branch_period", "branch_code", "period"),
    )

    branch_code = Column(String(20), nullable=False)
    sku = Column(String(50), nullable=False)
    item = Column(String(200), nullable=True)
    period = Column(Date, nullable=False)              # month-end date
    qty = Column(Numeric(14, 2), nullable=False, default=0)
    turnover = Column(Numeric(14, 2), nullable=True)
    profit = Column(Numeric(14, 2), nullable=True)
    gp_pct = Column(Numeric(7, 3), nullable=True)
    day_from = Column(Integer, nullable=True)
    day_to = Column(Integer, nullable=True)


class WeeklySalesLine(TimestampedBase):
    """One row per (branch, sku, week) - feeds Flow Analysis's KPIs and the
    weekly per-SKU demand-forecast models. ``is_simulated`` rows are the
    monthly-history-derived gap-filler weekly_simulate.py builds for a
    branch/week with no real weekly upload yet - never both real and
    simulated for the same (branch, week) at once in practice (simulate
    explicitly skips days a real upload already covers), but the constraint
    allows it rather than assumes it, since a real upload's own save doesn't
    reach into simulated rows to clean them up (regenerate() does, wholesale,
    right after)."""
    __tablename__ = "weekly_sales_lines"
    __table_args__ = (
        UniqueConstraint("branch_code", "sku", "week_start", "is_simulated",
                         name="uq_wsl_branch_sku_week"),
        Index("ix_wsl_branch_week", "branch_code", "week_start"),
    )

    branch_code = Column(String(20), nullable=False)
    sku = Column(String(50), nullable=False)
    is_simulated = Column(Boolean, nullable=False, default=False)
    item = Column(String(200), nullable=True)
    week_start = Column(Date, nullable=False)
    qty = Column(Numeric(14, 2), nullable=False, default=0)
    profit = Column(Numeric(14, 2), nullable=True)
    revenue = Column(Numeric(14, 2), nullable=True)


class WeeklyStockSnapshotLine(TimestampedBase):
    """One row per (branch, sku, week) - the weekly Hansa on-hand exports used
    to spot stockout weeks in the weekly demand model (see weekly_forecast.py's
    'stockout unconstraining')."""
    __tablename__ = "weekly_stock_snapshot_lines"
    __table_args__ = (
        UniqueConstraint("branch_code", "sku", "week_start",
                         name="uq_wssl_branch_sku_week"),
        Index("ix_wssl_branch_week", "branch_code", "week_start"),
    )

    branch_code = Column(String(20), nullable=False)
    sku = Column(String(50), nullable=False)
    week_start = Column(Date, nullable=False)
    qty_on_hand = Column(Numeric(14, 2), nullable=False, default=0)


# ======================================================================
# AUDIT  (every mutating action, with the acting user)
# ======================================================================
class AuditLog(TimestampedBase):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_entity", "entity_type", "entity_id"),
        Index("ix_audit_user", "user_id"),
        Index("ix_audit_ts", "created_at"),
    )

    entity_type = Column(String(60), nullable=False)
    entity_id = Column(String(50), nullable=True)
    action = Column(String(40), nullable=False)
    detail = Column(Text, nullable=True)
    reason = Column(String(255), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    user = relationship("User", lazy="joined")
