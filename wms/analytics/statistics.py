"""Overall and per-branch statistics (sales + backorders; no stock on-hand)."""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from wms.enums import BackOrderStatus
from wms.models import Branch, Product, BackOrder
from wms.analytics import loaders
from wms.analytics import backorders as bo_an
from wms.analytics import backorder_flow as bof
from wms.analytics import monthly_sales


def overall_kpis(db: Session) -> dict:
    dl = loaders.dn_lines_df(db)
    dn_sum = bo_an.overall_summary(dl) if not dl.empty else {}
    hdr = loaders.back_orders_df(db)
    items = loaders.back_order_items_df(db)
    fm = bof.fulfilment_metrics(hdr, items) if not hdr.empty else {}

    # real monthly Hansa exports (data/sales_history), not the SalesRecord
    # table - most of it is fabricated seed history, see monthly_sales.py
    trend = monthly_sales.sales_trend(months=3)
    sales_qty = int(trend["qty"].sum()) if not trend.empty else 0
    sales_val = round(float(trend["value"].sum()), 2) if not trend.empty else 0.0

    open_bo = db.query(func.count(BackOrder.id)).filter(
        BackOrder.status == BackOrderStatus.OPEN.value).scalar()

    return {
        "as_of": date.today().isoformat(),
        "products": db.query(func.count(Product.id)).scalar(),
        "branches": db.query(func.count(Branch.id)).scalar(),
        "sales_qty_90d": sales_qty,
        "sales_value_90d": sales_val,
        "back_orders": fm.get("back_orders", 0),
        "open_back_orders": int(open_bo or 0),
        "backorder_fill_rate_qty": fm.get("fill_rate_qty"),
        "mean_lead_time_days": fm.get("mean_lead_time_days"),
        "bottleneck_stage": bof.bottleneck_stage(loaders.back_order_events_df(db)),
        "value_outstanding": fm.get("value_outstanding"),
        "dn_fill_rate_qty": dn_sum.get("fill_rate_qty"),
        "total_dn_backorder_qty": dn_sum.get("total_backorder_qty", 0),
    }


def branch_kpis(db: Session, branch_id: int) -> dict:
    b = db.query(Branch).filter(Branch.id == branch_id).first()
    if not b:
        return {"error": "branch not found"}

    # real monthly Hansa exports for this branch, not the SalesRecord table
    bs = monthly_sales.branch_sales_summary(months=3, branch_code=b.code)
    row = bs.iloc[0] if not bs.empty else None
    sales_qty = int(row["sales_qty"]) if row is not None else 0
    daily = round(sales_qty / 90, 2) if sales_qty else 0.0

    hdr = loaders.back_orders_df(db, branch_id=branch_id)
    items = loaders.back_order_items_df(db, branch_id=branch_id)
    fm = bof.fulfilment_metrics(hdr, items) if not hdr.empty else {}
    dl = loaders.dn_lines_df(db, branch_id=branch_id)
    dn_sum = bo_an.overall_summary(dl) if not dl.empty else {}

    return {
        "branch": b.name, "branch_id": branch_id,
        "sales_qty_90d": sales_qty,
        "sales_value_90d": float(row["sales_value"]) if row is not None else 0.0,
        "avg_daily_demand": daily,
        "distinct_skus_sold": int(row["distinct_skus"]) if row is not None else 0,
        "back_orders": fm.get("back_orders", 0),
        "open_back_orders": fm.get("open", 0),
        "backorder_fill_rate_qty": fm.get("fill_rate_qty"),
        "mean_lead_time_days": fm.get("mean_lead_time_days"),
        "overdue_open": fm.get("overdue_open", 0),
        "value_outstanding": fm.get("value_outstanding"),
        "dn_fill_rate_qty": dn_sum.get("fill_rate_qty"),
    }


def branch_comparison(db: Session) -> pd.DataFrame:
    return pd.DataFrame([branch_kpis(db, b.id)
                         for b in db.query(Branch).order_by(Branch.name).all()])


def abc_classification(db: Session, days: int = 90) -> pd.DataFrame:
    """A/B/C SKUs by sales value share (80 / 15 / 5), from real monthly Hansa
    exports (see wms.analytics.monthly_sales) - not the SalesRecord table,
    whose sales history is almost entirely fabricated demo/seed data."""
    return monthly_sales.abc_classification(months=max(1, days // 30))


def sales_trend(db: Session, *, branch_id: int | None = None, freq: str = "W") -> pd.DataFrame:
    """Monthly qty/value trend, from real monthly Hansa exports. ``freq`` is
    accepted for backward compatibility but ignored: the source data is
    monthly, so the trend is reported per month, not resampled to weeks."""
    bcode = ""
    if branch_id:
        b = db.query(Branch).filter(Branch.id == branch_id).first()
        bcode = b.code if b else ""
    return monthly_sales.sales_trend(months=6, branch_code=bcode)


def branch_sales_summary(db: Session, days: int = 90) -> pd.DataFrame:
    """Per-branch sales totals + trend, from real monthly Hansa exports (see
    wms.analytics.monthly_sales) - not the SalesRecord table, whose sales
    history is almost entirely fabricated demo/seed data (see seed.py)."""
    return monthly_sales.branch_sales_summary(months=max(1, days // 30))
