"""Demand forecasting -> suggested branch orders ("anticipate branch orders").

Order-up-to-target: with no stock-on-hand tracking, the suggested order for a
(branch, product) each review cycle is the target level itself:

    avg_daily_demand   = recency-weighted mean of daily demand over the window
    safety_stock       = z(service_level) * demand_std * sqrt(lead_time)
    reorder_point      = avg * lead_time + safety_stock
    target_level       = avg * (lead_time + review_period) + safety_stock
    suggested_order_qty = ceil(target_level)
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from wms.config import get_settings
from wms.models import Branch, Product
from wms.analytics import loaders

settings = get_settings()
_Z = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def _z(sl: float) -> float:
    return _Z.get(round(sl, 3), 1.6449)


def _weighted_daily(day_series: pd.Series, days: int) -> tuple[float, float]:
    idx = pd.date_range(end=date.today(), periods=days, freq="D")
    daily = day_series.reindex(idx, fill_value=0).astype(float)
    w = np.linspace(0.5, 1.5, len(daily))
    return float(np.average(daily.values, weights=w)), float(daily.std(ddof=0))


def forecast_table(
    db: Session, *,
    branch_id: Optional[int] = None,
    history_days: Optional[int] = None,
    lead_time_days: Optional[int] = None,
    service_level: Optional[float] = None,
    review_period_days: Optional[int] = None,
) -> pd.DataFrame:
    history_days = history_days or settings.sales_history_days
    lead_time_days = lead_time_days or settings.lead_time_days
    service_level = service_level or settings.service_level
    review_period_days = review_period_days or settings.review_period_days

    dem = loaders.demand_df(db, days=history_days)
    if dem.empty:
        return pd.DataFrame()
    branches = {b.id: b.name for b in db.query(Branch).all()}
    products = {p.id: p for p in db.query(Product).all()}

    if branch_id:
        dem = dem[dem.branch_id == branch_id]
    z = _z(service_level)
    rows = []
    for (bid, pid), grp in dem.groupby(["branch_id", "product_id"]):
        s = grp.groupby("date")["qty"].sum()
        mean, std = _weighted_daily(s, history_days)
        if mean <= 0:
            continue
        safety = z * std * math.sqrt(max(lead_time_days, 1))
        rop = mean * lead_time_days + safety
        target = mean * (lead_time_days + review_period_days) + safety
        rows.append({
            "branch_id": int(bid), "branch": branches.get(bid, str(bid)),
            "sku": products[pid].sku if pid in products else str(pid),
            "product_id": int(pid),
            "description": products[pid].name if pid in products else "",
            "avg_daily_demand": round(mean, 3), "demand_std": round(std, 3),
            "reorder_point": round(rop, 1),
            "target_level": round(target, 1),
            "suggested_order_qty": max(0, math.ceil(target)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("suggested_order_qty", ascending=False)


def suggested_orders(db: Session, *, branch_id: Optional[int] = None) -> pd.DataFrame:
    df = forecast_table(db, branch_id=branch_id)
    return df[df.suggested_order_qty > 0].reset_index(drop=True) if not df.empty else df
