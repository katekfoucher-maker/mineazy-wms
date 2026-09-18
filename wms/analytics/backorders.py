"""Backorder analytics - overall and by branch.

backorder = requested_qty - sent_qty (blank sent == 0) per delivery-note line.
All functions take the ``dn_lines_df`` frame from ``wms.analytics.loaders``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rate(num, den) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


def overall_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"lines": 0}
    bo = df[df.backorder_qty > 0]
    n = len(df)
    val = df["backorder_value"].dropna()
    return {
        "documents": int(df.dn_no.nunique()),
        "branches": int(df.branch.nunique()),
        "lines": n,
        "distinct_items": int(df.sku.nunique()),
        "total_requested": int(df.requested_qty.sum()),
        "total_sent": int(df.sent_qty.sum()),
        "total_backorder_qty": int(df.backorder_qty.sum()),
        "fill_rate_qty": _rate(df.sent_qty.sum(), df.requested_qty.sum()),
        "lines_full": int((df.fill_status == "FULL").sum()),
        "lines_partial": int((df.fill_status == "PARTIAL").sum()),
        "lines_nil": int((df.fill_status == "NIL").sum()),
        "line_fill_rate": _rate((df.fill_status == "FULL").sum(), n),
        "backordered_lines": int(len(bo)),
        "backordered_lines_pct": _rate(len(bo), n),
        "backorder_value": round(float(val.sum()), 2) if not val.empty else None,
    }


def by_fill_status(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["fill_status", "lines", "requested", "sent", "backorder_qty"])
    g = (df.groupby("fill_status")
           .agg(lines=("sku", "count"), requested=("requested_qty", "sum"),
                sent=("sent_qty", "sum"), backorder_qty=("backorder_qty", "sum"))
           .reindex(["FULL", "PARTIAL", "NIL"]).fillna(0).astype(int))
    g["pct_of_lines"] = (g.lines / g.lines.sum() * 100).round(1)
    return g.reset_index()


def by_branch(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["branch", "lines", "backordered_lines", "requested",
                                     "sent", "backorder_qty", "fill_rate_qty",
                                     "line_fill_rate", "nil_lines", "backorder_value"])
    grp = df.groupby("branch")
    out = grp.agg(
        lines=("sku", "count"),
        backordered_lines=("backorder_qty", lambda s: int((s > 0).sum())),
        requested=("requested_qty", "sum"),
        sent=("sent_qty", "sum"),
        backorder_qty=("backorder_qty", "sum"),
        backorder_value=("backorder_value", "sum"),
        oldest_age_days=("age_days", "max"),
    )
    full = df[df.backorder_qty == 0].groupby("branch").size().reindex(out.index).fillna(0)
    nil = df[df.fill_status == "NIL"].groupby("branch").size().reindex(out.index).fillna(0)
    out["fill_rate_qty"] = (out["sent"] / out["requested"]).round(4)
    out["line_fill_rate"] = (full / out["lines"]).round(4)
    out["nil_lines"] = nil.astype(int)
    return out.reset_index().sort_values("fill_rate_qty")


def by_item(df: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sku", "description", "occurrences", "branches_affected",
                                     "requested", "sent", "backorder_qty", "fill_rate_qty",
                                     "backorder_value"])
    g = (df.groupby(["sku", "description"])
           .agg(occurrences=("dn_no", "nunique"), branches_affected=("branch", "nunique"),
                requested=("requested_qty", "sum"), sent=("sent_qty", "sum"),
                backorder_qty=("backorder_qty", "sum"),
                backorder_value=("backorder_value", "sum"))
           .reset_index())
    g["fill_rate_qty"] = (g.sent / g.requested).round(4)
    g = g.sort_values("backorder_qty", ascending=False)
    return g.head(top) if top else g


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df["category"].isna().all():
        return pd.DataFrame(columns=["category", "lines", "requested", "sent",
                                     "backorder_qty", "fill_rate_qty"])
    g = (df.assign(category=df["category"].fillna("(uncategorised)"))
           .groupby("category")
           .agg(lines=("sku", "count"), requested=("requested_qty", "sum"),
                sent=("sent_qty", "sum"), backorder_qty=("backorder_qty", "sum"))
           .reset_index())
    g["fill_rate_qty"] = (g.sent / g.requested).round(4)
    return g.sort_values("backorder_qty", ascending=False)


def ageing(df: pd.DataFrame) -> pd.DataFrame:
    labels = ["0-7", "8-14", "15-30", "30+"]
    bo = df[df.backorder_qty > 0].copy()
    if bo.empty:
        return pd.DataFrame({"bucket": labels, "lines": [0] * 4, "backorder_qty": [0] * 4})
    bo["bucket"] = pd.cut(bo.age_days, bins=[-1, 7, 14, 30, np.inf], labels=labels)
    return (bo.groupby("bucket", observed=False)
              .agg(lines=("sku", "count"), backorder_qty=("backorder_qty", "sum"))
              .reset_index())


def trend(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "requested", "sent", "backorder_qty", "fill_rate_qty"])
    g = (df.set_index("doc_date").groupby(pd.Grouper(freq=freq))
           .agg(requested=("requested_qty", "sum"), sent=("sent_qty", "sum"),
                backorder_qty=("backorder_qty", "sum"))
           .reset_index().rename(columns={"doc_date": "period"}))
    g["fill_rate_qty"] = (g.sent / g.requested).round(4)
    return g[g.requested > 0]


def recurring(df: pd.DataFrame, min_occurrences: int = 2) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["branch", "sku", "description", "occurrences",
                                     "total_backorder_qty"])
    bo = df[df.backorder_qty > 0]
    g = (bo.groupby(["branch", "sku", "description"])
           .agg(occurrences=("dn_no", "nunique"),
                total_backorder_qty=("backorder_qty", "sum"))
           .reset_index())
    return g[g.occurrences >= min_occurrences].sort_values(
        ["occurrences", "total_backorder_qty"], ascending=False)
