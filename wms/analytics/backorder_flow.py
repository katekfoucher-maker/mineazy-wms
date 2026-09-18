"""Backorder processing-flow analytics.

Comprehensive analysis of the back-order lifecycle:
fulfilment metrics · stage funnel · cycle times / lead time · aging ·
by branch · **backorders vs sales by branch** · weekly trend.

All functions take frames from ``wms.analytics.loaders``:
``back_orders_df`` (headers), ``back_order_items_df`` (lines),
``back_order_events_df`` (stage transitions).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from wms.enums import STAGE_LABEL, STAGE_SEQUENCE, BackOrderStage
from wms.analytics import loaders


def _rate(n, d) -> float:
    return round(float(n) / float(d), 4) if d else 0.0


def _label(stage: str) -> str:
    return (STAGE_LABEL.get(BackOrderStage(stage), stage)
            if stage in BackOrderStage._value2member_map_ else stage)


# ----------------------------------------------------------------------
# 1. Fulfilment + headline metrics
# ----------------------------------------------------------------------
def fulfilment_metrics(bo: pd.DataFrame, items: pd.DataFrame) -> dict:
    if bo.empty:
        return {"back_orders": 0}
    closed = bo[bo.status == "CLOSED"]
    open_ = bo[bo.status == "OPEN"]
    ordered = int(items.qty_ordered.sum()) if not items.empty else 0
    fulfilled = int(items.qty_fulfilled.sum()) if not items.empty else 0
    lead = closed["lead_time_days"].dropna()
    line_full = int((items.fill_status == "FULL").sum()) if not items.empty else 0
    on_time = ((closed["lead_time_days"] <= closed["sla_days"]).sum()
               if not closed.empty else 0)
    return {
        "back_orders": len(bo),
        "open": len(open_), "closed": len(closed),
        "branches_affected": int(bo.branch.nunique()),
        "qty_ordered": ordered, "qty_fulfilled": fulfilled,
        "qty_outstanding": ordered - fulfilled,
        "fill_rate_qty": _rate(fulfilled, ordered),
        "item_lines": len(items),
        "lines_fully_fulfilled": line_full,
        "line_fill_rate": _rate(line_full, len(items)) if not items.empty else 0.0,
        "mean_fulfil_pct_closed": round(float(closed.fulfil_pct.mean()), 4)
            if not closed.empty else None,
        "value_outstanding": round(float(bo.value_outstanding.sum()), 2),
        "mean_lead_time_days": round(float(lead.mean()), 1) if not lead.empty else None,
        "median_lead_time_days": round(float(lead.median()), 1) if not lead.empty else None,
        "overdue_open": int(open_.is_overdue.sum()),
        "on_time_close_rate": _rate(on_time, len(closed)) if not closed.empty else None,
    }


# ----------------------------------------------------------------------
# 2. Stage funnel - where do back orders sit right now
# ----------------------------------------------------------------------
def stage_funnel(bo: pd.DataFrame) -> pd.DataFrame:
    order = [s.value for s in STAGE_SEQUENCE]
    labels = [_label(s) for s in order]
    if bo.empty:
        return pd.DataFrame({"stage": order, "label": labels, "back_orders": 0,
                             "qty_outstanding": 0, "value_outstanding": 0.0})
    g = (bo.groupby("stage")
           .agg(back_orders=("bo_no", "count"),
                qty_outstanding=("outstanding_qty", "sum"),
                value_outstanding=("value_outstanding", "sum"))
           .reindex(order).fillna(0))
    g["back_orders"] = g["back_orders"].astype(int)
    g["qty_outstanding"] = g["qty_outstanding"].astype(int)
    g = g.reset_index()
    g["label"] = g["stage"].map(_label)
    return g[["stage", "label", "back_orders", "qty_outstanding", "value_outstanding"]]


# ----------------------------------------------------------------------
# 3. Cycle times - days spent in each stage, and the bottleneck
# ----------------------------------------------------------------------
def _segments(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.sort_values(["bo_no", "at"]).copy()
    ev["next_at"] = ev.groupby("bo_no")["at"].shift(-1)
    ev["days_in_stage"] = (ev["next_at"] - ev["at"]).dt.total_seconds() / 86400
    return ev.dropna(subset=["days_in_stage"])


def cycle_times(events: pd.DataFrame) -> pd.DataFrame:
    cols = ["stage", "label", "transitions", "mean_days", "median_days", "p90_days"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    seg = _segments(events)
    if seg.empty:
        return pd.DataFrame(columns=cols)
    g = (seg.groupby("to_stage")
            .agg(transitions=("bo_no", "count"),
                 mean_days=("days_in_stage", "mean"),
                 median_days=("days_in_stage", "median"),
                 p90_days=("days_in_stage", lambda s: s.quantile(0.9)))
            .reset_index().rename(columns={"to_stage": "stage"}))
    pos = {s.value: i for i, s in enumerate(STAGE_SEQUENCE)}
    g = g.assign(_o=g["stage"].map(lambda s: pos.get(s, 99))).sort_values("_o").drop(columns="_o")
    g["label"] = g["stage"].map(_label)
    for c in ("mean_days", "median_days", "p90_days"):
        g[c] = g[c].round(2)
    return g[cols]


def bottleneck_stage(events: pd.DataFrame) -> Optional[str]:
    ct = cycle_times(events)
    if ct.empty:
        return None
    return str(ct.loc[ct["mean_days"].idxmax(), "label"])


# ----------------------------------------------------------------------
# 4. Aging of open back orders
# ----------------------------------------------------------------------
def aging(bo: pd.DataFrame) -> pd.DataFrame:
    buckets = ["0-7", "8-14", "15-30", "30+"]
    open_ = bo[bo.status == "OPEN"].copy() if not bo.empty else bo
    if open_.empty:
        return pd.DataFrame({"bucket": buckets, "back_orders": 0,
                             "qty_outstanding": 0, "value_outstanding": 0.0})
    open_["bucket"] = pd.cut(open_.age_days.fillna(0),
                             bins=[-1, 7, 14, 30, np.inf], labels=buckets)
    g = (open_.groupby("bucket", observed=False)
              .agg(back_orders=("bo_no", "count"),
                   qty_outstanding=("outstanding_qty", "sum"),
                   value_outstanding=("value_outstanding", "sum"))
              .reset_index())
    return g


def aging_by_stage(bo: pd.DataFrame) -> pd.DataFrame:
    """Open back-order count per (current stage x age bucket) - the heatmap source."""
    open_ = bo[bo.status == "OPEN"].copy() if not bo.empty else bo
    if open_.empty:
        return pd.DataFrame()
    open_["bucket"] = pd.cut(open_.age_days.fillna(0),
                             bins=[-1, 7, 14, 30, np.inf],
                             labels=["0-7", "8-14", "15-30", "30+"])
    piv = (open_.pivot_table(index="stage", columns="bucket", values="bo_no",
                             aggfunc="count", observed=False, fill_value=0))
    pos = {s.value: i for i, s in enumerate(STAGE_SEQUENCE)}
    piv = piv.reindex(sorted(piv.index, key=lambda s: pos.get(s, 99)))
    piv.index = [_label(s) for s in piv.index]
    return piv.reset_index().rename(columns={"index": "stage"})


# ----------------------------------------------------------------------
# 5. By branch
# ----------------------------------------------------------------------
def by_branch(bo: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    if bo.empty:
        return pd.DataFrame(columns=["branch", "back_orders", "open", "closed",
                                     "qty_ordered", "qty_fulfilled", "fill_rate_qty",
                                     "mean_lead_time_days", "overdue_open",
                                     "value_outstanding"])
    hg = bo.groupby("branch")
    out = hg.agg(
        back_orders=("bo_no", "count"),
        open=("status", lambda s: int((s == "OPEN").sum())),
        closed=("status", lambda s: int((s == "CLOSED").sum())),
        mean_lead_time_days=("lead_time_days", "mean"),
        overdue_open=("is_overdue", "sum"),
        value_outstanding=("value_outstanding", "sum"),
    )
    ig = (items.groupby("branch").agg(qty_ordered=("qty_ordered", "sum"),
                                      qty_fulfilled=("qty_fulfilled", "sum"))
          if not items.empty else pd.DataFrame(columns=["qty_ordered", "qty_fulfilled"]))
    out = out.join(ig).fillna(0)
    out["fill_rate_qty"] = (out["qty_fulfilled"] / out["qty_ordered"]).replace([np.inf, np.nan], 0).round(4)
    out["mean_lead_time_days"] = out["mean_lead_time_days"].round(1)
    for c in ("qty_ordered", "qty_fulfilled", "overdue_open"):
        out[c] = out[c].astype(int)
    return out.reset_index().sort_values("fill_rate_qty")


def top_items(items: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame(columns=["sku", "description", "back_orders", "branches",
                                     "qty_ordered", "qty_fulfilled", "outstanding_qty",
                                     "fill_rate_qty", "value_outstanding"])
    g = (items.groupby(["sku", "description"])
              .agg(back_orders=("bo_no", "nunique"), branches=("branch", "nunique"),
                   qty_ordered=("qty_ordered", "sum"), qty_fulfilled=("qty_fulfilled", "sum"),
                   outstanding_qty=("outstanding_qty", "sum"),
                   value_outstanding=("value_outstanding", "sum"))
              .reset_index())
    g["fill_rate_qty"] = (g.qty_fulfilled / g.qty_ordered).round(4)
    return g.sort_values("outstanding_qty", ascending=False).head(top)


# ----------------------------------------------------------------------
# 6. Back orders vs sales, by branch
# ----------------------------------------------------------------------
def branch_backorders_vs_sales(db: Session, *, weeks: int = 13) -> pd.DataFrame:
    """Per branch, as a **weekly rate**: sales demand vs backordered demand,
    with a 'demand met' proxy = sales / (sales + backordered). The sales rate
    comes from :func:`wms.analytics.weekly_forecast.weekly_demand_estimate`
    - real weekly sales history for branches that have uploaded it, else a
    real-monthly-data-derived estimate (never the SalesRecord table, whose
    sales history is almost entirely fabricated demo/seed data)."""
    from wms.analytics import weekly_forecast as wfc
    days = weeks * 7
    cutoff = date.today() - timedelta(days=days)
    wd = wfc.weekly_demand_estimate()
    sales = (wd.groupby("branch")["weekly_demand"].sum() if not wd.empty
             else pd.Series(dtype=float))
    bo = loaders.back_orders_df(db)
    items = loaders.back_order_items_df(db)
    if not bo.empty:
        recent = bo[bo.submitted_at >= pd.Timestamp(cutoff)]
        bo_qty = (items[items.bo_no.isin(recent.bo_no)].groupby("branch")["qty_ordered"].sum()
                  if not items.empty else pd.Series(dtype=float))
        bo_cnt = recent.groupby("branch")["bo_no"].count()
        fill = recent.groupby("branch")["fulfil_pct"].mean()
    else:
        bo_qty = bo_cnt = fill = pd.Series(dtype=float)

    branches = sorted(set(sales.index) | set(bo_qty.index))
    wk = max(weeks, 1)
    rows = []
    for b in branches:
        s = float(sales.get(b, 0.0))          # already a weekly rate
        q_total = float(bo_qty.get(b, 0.0))   # a window total
        q = q_total / wk                      # -> weekly rate, comparable to s
        rows.append({
            "branch": b,
            "sales_per_week": round(s, 1),
            "backordered_per_week": round(q, 1),
            "back_orders_per_week": round(int(bo_cnt.get(b, 0)) / wk, 2),
            "mean_fulfil_pct": round(float(fill.get(b, 0.0)), 4),
            "demand_met_pct": round(s / (s + q), 4) if (s + q) else None,
            "backorder_to_sales_ratio": round(q / s, 4) if s else None,
            # window totals kept for charts / exports
            "sales_qty": int(round(s * wk)),
            "backordered_qty": int(q_total),
            "back_orders": int(bo_cnt.get(b, 0)),
        })
    return pd.DataFrame(rows).sort_values("backorder_to_sales_ratio",
                                          ascending=False, na_position="last")


# ----------------------------------------------------------------------
# 7. Weekly trend
# ----------------------------------------------------------------------
def trend(bo: pd.DataFrame, events: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    if bo.empty:
        return pd.DataFrame(columns=["period", "raised", "closed", "outstanding",
                                     "fill_rate_qty"])
    raised = (bo.dropna(subset=["submitted_at"])
                .set_index("submitted_at").groupby(pd.Grouper(freq=freq))
                .agg(raised=("bo_no", "count"),
                     ordered=("qty_ordered", "sum"),
                     fulfilled=("qty_fulfilled", "sum")))
    closed = (bo[bo.status == "CLOSED"].dropna(subset=["closed_at"])
                .set_index("closed_at").groupby(pd.Grouper(freq=freq))
                .agg(closed=("bo_no", "count")))
    g = raised.join(closed, how="outer").fillna(0)
    g["outstanding"] = (g["raised"].cumsum() - g["closed"].cumsum()).astype(int)
    g["fill_rate_qty"] = (g["fulfilled"] / g["ordered"]).replace([np.inf, np.nan], 0).round(4)
    g = g.reset_index().rename(columns={"index": "period"})
    g.columns = ["period" if c not in ("raised", "closed", "outstanding", "ordered",
                                       "fulfilled", "fill_rate_qty") else c
                 for c in g.columns]
    for c in ("raised", "closed"):
        g[c] = g[c].astype(int)
    return g[["period", "raised", "closed", "outstanding", "fill_rate_qty"]]


# ----------------------------------------------------------------------
# 8. Product performance - was a weak seller under-stocked, or is demand just low?
# ----------------------------------------------------------------------
_PP_COLS = ["sku", "description", "category", "sold_qty", "sold_value",
            "requested_qty", "sent_qty", "fill_rate", "unmet_demand_ratio",
            "stockout_line_pct", "bo_incidents", "bo_qty", "bo_outstanding_qty",
            "recovery_rate", "avg_wait_days", "demand_captured_pct",
            "lost_sales_value", "verdict"]


def _pp_verdict(r) -> str:
    """Reads a product row and says whether poor sales look like a supply
    problem (under-stocked / unavailable) or genuinely weak demand."""
    req = r.get("requested_qty") or 0
    sold = r.get("sold_qty") or 0
    inc = r.get("bo_incidents") or 0
    fr, umd, so = r.get("fill_rate"), r.get("unmet_demand_ratio"), r.get("stockout_line_pct")

    def bad(v, thr):
        return v is not None and v == v and v > thr           # not NaN and over threshold

    supply_short = ((fr is not None and fr == fr and fr < 0.6)
                    or bad(umd, 0.4) or bad(so, 0.3))
    if req == 0 and inc == 0:
        return "No demand signal"                             # branches aren't asking
    if supply_short:
        return "Chronic shortage" if inc >= 3 else "Supply-constrained"
    if (fr is None or fr != fr or fr >= 0.9) and inc == 0 and sold <= 5:
        return "Genuinely low demand"
    return "Healthy"


def product_performance(db: Session, *, days: int = 90,
                        branch_id: Optional[int] = None) -> pd.DataFrame:
    """Per SKU over the window, joining three signals:

    * **sales** - units and value actually sold
    * **delivery-note fill** - requested vs sent, and how often a request got
      *nothing* (a stockout line)
    * **back orders** - how much demand was deferred, how much came back, the wait

    plus a ``verdict`` that separates "sold little because it was unavailable"
    from "sold little because nobody wanted it".

    Sales come from real monthly Hansa exports (see
    ``wms.analytics.monthly_sales``), not the SalesRecord table, whose sales
    history is almost entirely fabricated demo/seed data.
    """
    from wms.analytics import monthly_sales
    cutoff = date.today() - timedelta(days=days)
    bcode = ""
    if branch_id:
        from wms.models import Branch as _Branch
        b = db.query(_Branch).filter(_Branch.id == branch_id).first()
        bcode = b.code if b else ""
    sales = monthly_sales.recent_panel(monthly_sales.cached_panel(),
                                       months=max(1, days // 30), branch_code=bcode)
    dl = loaders.dn_lines_df(db, branch_id=branch_id, date_from=cutoff)
    items = loaders.back_order_items_df(db, branch_id=branch_id)
    bo = loaders.back_orders_df(db, branch_id=branch_id)

    s = (sales.groupby("sku").agg(sold_qty=("qty", "sum"), sold_value=("turnover", "sum"),
                                  selling_branches=("branch", "nunique"))
         if not sales.empty else pd.DataFrame())

    if not dl.empty:
        f = dl.groupby("sku").agg(
            requested_qty=("requested_qty", "sum"), sent_qty=("sent_qty", "sum"),
            dn_lines=("dn_no", "count"),
            nil_lines=("fill_status", lambda x: int((x == "NIL").sum())),
            partial_lines=("fill_status", lambda x: int((x == "PARTIAL").sum())),
            description=("description", "last"), category=("category", "last"))
    else:
        f = pd.DataFrame()

    if not items.empty:
        b = items.groupby("sku").agg(
            bo_incidents=("bo_no", "nunique"), bo_branches=("branch", "nunique"),
            bo_qty=("qty_ordered", "sum"), bo_fulfilled=("qty_fulfilled", "sum"),
            bo_outstanding_qty=("outstanding_qty", "sum"),
            bo_outstanding_value=("value_outstanding", "sum"))
        if not bo.empty:
            wait = (items[["bo_no", "sku"]].merge(bo[["bo_no", "lead_time_days"]], on="bo_no")
                    .groupby("sku")["lead_time_days"].mean().rename("avg_wait_days"))
            b = b.join(wait)
    else:
        b = pd.DataFrame()

    df = pd.concat([f, s, b], axis=1).reset_index().rename(columns={"index": "sku"})
    if df.empty:
        return pd.DataFrame(columns=_PP_COLS)
    if "sku" not in df.columns:                        # concat produced an unnamed index
        df = df.rename(columns={df.columns[0]: "sku"})

    add0 = ["sold_qty", "sold_value", "selling_branches", "requested_qty", "sent_qty",
            "dn_lines", "nil_lines", "partial_lines", "bo_incidents", "bo_branches",
            "bo_qty", "bo_fulfilled", "bo_outstanding_qty", "bo_outstanding_value"]
    for c in add0:
        df[c] = df[c].fillna(0) if c in df.columns else 0

    req = df["requested_qty"].replace(0, np.nan)
    df["fill_rate"] = (df["sent_qty"] / req).round(3)
    df["unmet_demand_ratio"] = ((df["requested_qty"] - df["sent_qty"]).clip(lower=0) / req).round(3)
    df["stockout_line_pct"] = (df["nil_lines"] / df["dn_lines"].replace(0, np.nan)).round(3)
    df["recovery_rate"] = (df["bo_fulfilled"] / df["bo_qty"].replace(0, np.nan)).round(3)
    known = df["sold_qty"] + df["bo_qty"]
    df["demand_captured_pct"] = (df["sold_qty"] / known.replace(0, np.nan)).round(3)
    df["lost_sales_value"] = df["bo_outstanding_value"].round(2)
    df["avg_wait_days"] = df["avg_wait_days"].round(1) if "avg_wait_days" in df.columns else np.nan
    df["description"] = (df["description"].fillna(df["sku"])
                         if "description" in df.columns else df["sku"])
    if "category" not in df.columns:
        df["category"] = None
    df["verdict"] = df.apply(_pp_verdict, axis=1)

    for c in _PP_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return (df[_PP_COLS]
            .sort_values(["lost_sales_value", "unmet_demand_ratio", "sold_value"],
                         ascending=False, na_position="last")
            .reset_index(drop=True))


def product_performance_kpis(pp: pd.DataFrame) -> dict:
    if pp.empty:
        return {"products": 0, "supply_constrained": 0, "chronic_shortage": 0,
                "genuinely_low_demand": 0, "products_with_stockout": 0,
                "lost_sales_value": 0.0, "median_fill_rate": None}
    v = pp["verdict"]
    fr = pp["fill_rate"].dropna()
    return {
        "products": int(len(pp)),
        "supply_constrained": int((v == "Supply-constrained").sum()),
        "chronic_shortage": int((v == "Chronic shortage").sum()),
        "genuinely_low_demand": int((v == "Genuinely low demand").sum()),
        "products_with_stockout": int((pp["stockout_line_pct"].fillna(0) > 0).sum()),
        "lost_sales_value": round(float(pp["lost_sales_value"].fillna(0).sum()), 2),
        "median_fill_rate": round(float(fr.median()), 3) if not fr.empty else None,
    }


# ----------------------------------------------------------------------
# Low sales: demand problem or supply problem?
# ----------------------------------------------------------------------
def low_sales_diagnosis(db: Session, *, bcode: str = "",
                        min_predicted: int = 3, gap_frac: float = 0.4) -> pd.DataFrame:
    """For every (branch, SKU) selling well below its weekly forecast, say WHY:

      * ``Stockout, back ordered`` : a branch raised a back order for it, so the
        low sales are because it was not on the shelf (tracked via back orders).
      * ``No stock, no back order`` : nothing on hand and nobody back-ordered it;
        a likely supply gap the back-order process missed.
      * ``Low demand, stock available`` : stock is on hand and no back order:
        this is genuine low demand, NOT a supply problem (no false alarm).

    ``gap_frac`` is how far below forecast counts as "low" (0.4 = sold <60% of
    predicted). ``min_predicted`` skips tiny/noisy forecasts.
    """
    from wms.analytics import weekly_forecast as wfc
    from wms.services import stock as stock_svc
    from wms.models import BackOrder, BackOrderItem, Branch

    cols = ["branch", "sku", "product", "predicted_wk", "actual_wk",
            "shortfall_wk", "backordered", "on_hand", "cause"]
    st = wfc.cached_run()["state"]
    if st is None or st.empty or "recent_sales" not in st.columns:
        return pd.DataFrame(columns=cols)

    # open back-order demand per (branch code, sku)
    bq = (db.query(Branch.code, BackOrderItem.sku,
                   func.sum(BackOrderItem.qty_ordered - BackOrderItem.qty_fulfilled))
          .join(BackOrder, BackOrder.id == BackOrderItem.back_order_id)
          .join(Branch, Branch.id == BackOrder.branch_id)
          .filter(BackOrder.status == "OPEN")
          .group_by(Branch.code, BackOrderItem.sku).all())
    bo_by = {(c, s): int(v or 0) for c, s, v in bq}

    lv = stock_svc.levels_df(db)
    oh_by = {(r.branch_code, r.sku): int(r.on_hand) for r in lv.itertuples()} \
        if not lv.empty else {}

    want = bcode.strip().lower()
    rows = []
    for r in st.itertuples():
        pred, act = int(r.weekly_demand), int(r.recent_sales)
        if pred < min_predicted or act >= pred * (1 - gap_frac):
            continue
        if want and r.branch.lower() != want and not r.branch_name.lower().startswith(want):
            continue
        boq = bo_by.get((r.branch, r.sku), 0)
        oh = oh_by.get((r.branch, r.sku), 0)
        if boq > 0:
            cause = "Stockout, back ordered"
        elif oh <= 0:
            cause = "No stock, no back order"
        else:
            cause = "Low demand, stock available"
        rows.append({"branch": r.branch_name, "sku": r.sku, "product": r.item,
                     "predicted_wk": pred, "actual_wk": act,
                     "shortfall_wk": pred - act, "backordered": boq,
                     "on_hand": oh, "cause": cause})
    if not rows:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(rows)[cols]
            .sort_values(["shortfall_wk", "backordered"], ascending=False)
            .reset_index(drop=True))


# ----------------------------------------------------------------------
# convenience: everything at once
# ----------------------------------------------------------------------
def full_report(db: Session) -> dict:
    bo = loaders.back_orders_df(db)
    items = loaders.back_order_items_df(db)
    events = loaders.back_order_events_df(db)
    return {
        "fulfilment": fulfilment_metrics(bo, items),
        "bottleneck_stage": bottleneck_stage(events),
        "stage_funnel": stage_funnel(bo),
        "cycle_times": cycle_times(events),
        "aging": aging(bo),
        "aging_by_stage": aging_by_stage(bo),
        "by_branch": by_branch(bo, items),
        "top_items": top_items(items),
        "branch_vs_sales": branch_backorders_vs_sales(db),
        "trend": trend(bo, events),
    }
