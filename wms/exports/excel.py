"""Excel workbook builders (xlsxwriter)."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from wms.config import get_settings
from wms.models import Branch
from wms.analytics import loaders, statistics, allocation
from wms.analytics import backorders as bo_an
from wms.analytics import backorder_flow as bof

settings = get_settings()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_workbook(sheets: dict[str, pd.DataFrame], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        book = xw.book
        head = book.add_format({"bold": True, "bg_color": "#1F3864",
                                "font_color": "white", "border": 1})
        for name, df in sheets.items():
            df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            sn = name[:31]
            df.to_excel(xw, sheet_name=sn, index=False, startrow=1, header=False)
            ws = xw.sheets[sn]
            if df.empty:
                ws.write(0, 0, f"{name} - no data")
                continue
            for c, col in enumerate(df.columns):
                ws.write(0, c, str(col), head)
                mx = df[col].map(lambda v: len(str(v)) if v is not None else 0).max()
                width = max(12, min(48, (int(mx) if pd.notna(mx) else 12) + 2))
                ws.set_column(c, c, width)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
    return path


def backorder_workbook(db, *, branch_id: Optional[int] = None,
                       date_from: Optional[date] = None, date_to: Optional[date] = None,
                       out_dir: Optional[Path] = None) -> Path:
    dl = loaders.dn_lines_df(db, branch_id=branch_id, date_from=date_from, date_to=date_to)
    summary = pd.DataFrame([{"metric": k, "value": v}
                            for k, v in bo_an.overall_summary(dl).items()])
    detail = dl[dl.backorder_qty > 0][
        ["dn_no", "doc_date", "from_location", "branch", "sku", "description",
         "requested_qty", "sent_qty", "backorder_qty", "fill_status", "unit_price",
         "backorder_value", "age_days"]
    ].sort_values(["branch", "backorder_qty"], ascending=[True, False])
    scope = f"branch{branch_id}" if branch_id else "all"
    path = Path(out_dir or settings.out) / f"backorders_{scope}_{_ts()}.xlsx"
    return write_workbook({
        "Summary": summary,
        "By Fill Status": bo_an.by_fill_status(dl),
        "By Branch": bo_an.by_branch(dl),
        "By Item": bo_an.by_item(dl),
        "By Category": bo_an.by_category(dl),
        "Ageing": bo_an.ageing(dl),
        "Trend": bo_an.trend(dl),
        "Recurring": bo_an.recurring(dl, min_occurrences=1),
        "Line Detail": detail,
        "Open Back Orders": loaders.back_orders_df(db, branch_id=branch_id, status="OPEN"),
        "Source Lines": dl,
    }, path)


def backorder_flow_workbook(db, *, branch_id: Optional[int] = None,
                            out_dir: Optional[Path] = None) -> Path:
    """The backorder *processing flow* workbook (stages, fulfilment, cycle times,
    aging, by branch, backorders vs sales, trend, events)."""
    bo = loaders.back_orders_df(db, branch_id=branch_id)
    items = loaders.back_order_items_df(db, branch_id=branch_id)
    events = loaders.back_order_events_df(db)
    summary = pd.DataFrame([{"metric": k, "value": v}
                            for k, v in bof.fulfilment_metrics(bo, items).items()])
    scope = f"branch{branch_id}" if branch_id else "all"
    path = Path(out_dir or settings.out) / f"backorder_flow_{scope}_{_ts()}.xlsx"
    return write_workbook({
        "Summary": summary,
        "Product Performance": bof.product_performance(db, branch_id=branch_id),
        "Active Back Orders": bo,
        "Items": items,
        "Stage Funnel": bof.stage_funnel(bo),
        "Cycle Times": bof.cycle_times(events),
        "Aging": bof.aging(bo),
        "Aging by Stage": bof.aging_by_stage(bo),
        "By Branch": bof.by_branch(bo, items),
        "Top Items Outstanding": bof.top_items(items, top=50),
        "Branch BO vs Sales": bof.branch_backorders_vs_sales(db),
        "Trend": bof.trend(bo, events),
        "Events": events,
    }, path)


def sales_workbook(db, *, out_dir: Optional[Path] = None) -> Path:
    """Branch sales analysis - real monthly Hansa "Item Statistics" exports
    (see wms.analytics.monthly_sales), not the SalesRecord table (whose sales
    history is almost entirely fabricated demo/seed data, see seed.py)."""
    from wms.analytics import monthly_sales
    path = Path(out_dir or settings.out) / f"branch_sales_{_ts()}.xlsx"
    return write_workbook({
        "By Branch": statistics.branch_sales_summary(db),
        "Monthly Trend": statistics.sales_trend(db),
        "ABC": statistics.abc_classification(db),
        "Sales Lines (monthly)": monthly_sales.cached_panel(),
    }, path)


def suggested_orders_workbook(db, *, branch_id: Optional[int] = None,
                              out_dir: Optional[Path] = None) -> Path:
    """Real reorder-point suggestions from actual weekly sales history and
    current on-hand stock (see weekly_forecast.reorder_points) - replaces the
    old forecast.forecast_table(), which derived its reorder-point statistics
    from the SalesRecord table's fabricated seed history."""
    from wms.analytics import weekly_forecast as wfc
    bcode = ""
    if branch_id:
        b = db.query(Branch).filter(Branch.id == branch_id).first()
        bcode = b.code if b else ""
    out = wfc.reorder_points(db, bcode=bcode, limit=100000)
    df = pd.DataFrame(out["rows"])
    sheets = {"All": df}
    if not df.empty:
        sheets["Suggested"] = df[df.status == "reorder_now"]
        for name, grp in df.groupby("branch"):
            sheets[str(name)[:31]] = grp
    path = Path(out_dir or settings.out) / f"suggested_orders_{_ts()}.xlsx"
    return write_workbook(sheets, path)


def demand_forecast_workbook(*, bcode: str = "", q: str = "",
                             out_dir: Optional[Path] = None) -> Path:
    """One sheet, exactly the columns shown on the Sales & Forecasting page,
    every row, respecting the same branch / search filters."""
    from wms.analytics import weekly_forecast as wfc
    table = wfc.display_frame(bcode=bcode, q=q)
    path = Path(out_dir or settings.out) / f"demand_forecast_{_ts()}.xlsx"
    return write_workbook({"Forecast by product": table}, path)


def allocation_workbook(db, *, branch_code: str = "", q: str = "",
                        out_dir: Optional[Path] = None) -> Path:
    path = Path(out_dir or settings.out) / f"allocation_plan_{_ts()}.xlsx"
    return write_workbook({
        "Weekly Allocation": allocation.weekly_allocation_plan(
            db, branch_code=branch_code, q=q),
        "Suggested Orders": allocation.allocation_plan(db),
    }, path)


def split_allocation_workbook(db, *, sku: str, qty: int, branch_codes=None,
                              out_dir: Optional[Path] = None) -> Path:
    """The 'split a quantity by predicted sales' result: how ``qty`` units of one
    product are shared across the chosen branches (or every forecast branch)."""
    res = allocation.allocate_by_forecast(db, sku=sku, qty=int(qty or 0),
                                          branch_codes=branch_codes)
    rows = pd.DataFrame(res.get("allocations", []))
    if not rows.empty:
        rows = rows.rename(columns={"branch": "Branch", "allocated": "Allocation"})
        rows = rows[["Branch", "Allocation"]]
    else:
        rows = pd.DataFrame(columns=["Branch", "Allocation"])
    scope = "-".join(branch_codes) if branch_codes else "all"
    _desc = str(res.get("description") or "").strip()
    _prod = str(res.get("product") or sku)
    _info = [
        {"field": "Product", "value": f"{_prod} - {_desc}" if _desc and _desc != _prod else _prod},
        {"field": "Quantity to split", "value": int(qty or 0)},
        {"field": "Branches", "value": ", ".join(branch_codes) if branch_codes else "all forecast branches"},
        {"field": "Generated", "value": _ts()},
    ]
    if res.get("note"):
        _info.append({"field": "Note", "value": str(res["note"])})
    info = pd.DataFrame(_info)
    path = Path(out_dir or settings.out) / f"split_allocation_{scope}_{_ts()}.xlsx"
    return write_workbook({"Split by predicted sales": rows, "Details": info}, path)


def split_allocation_batch_workbook(db, *, pairs, branch_codes=None,
                                    out_dir: Optional[Path] = None) -> Path:
    """One workbook covering several products split at once: a flat allocation
    table (one row per product per branch) plus a per-product summary."""
    scope_lbl = ", ".join(branch_codes) if branch_codes else "all forecast branches"
    detail_rows, summary_rows = [], []
    for sku, qty in pairs:
        sku = str(sku or "").strip()
        try:
            qty = int(float(qty))
        except (TypeError, ValueError):
            qty = 0
        if not sku or qty <= 0:
            continue
        res = allocation.allocate_by_forecast(db, sku=sku, qty=qty,
                                              branch_codes=branch_codes)
        desc = str(res.get("description") or "").strip()
        prod = str(res.get("product") or sku)
        name = f"{prod} - {desc}" if desc and desc != prod else prod
        wh = int(res.get("warehouse", 0) or 0)
        _basis = {"probe": "probe", "held": "no stock left", "untested": "untested"}
        for a in res.get("allocations", []):
            detail_rows.append({
                "Product": name, "SKU": sku, "Qty to split": qty,
                "Branch": a.get("branch"),
                "Allocation": a.get("allocated"),
                "Basis": _basis.get(a.get("kind"), "cover"),
            })
        if wh:
            detail_rows.append({
                "Product": name, "SKU": sku, "Qty to split": qty,
                "Branch": "Warehouse (hold)",
                "Allocation": wh, "Basis": "hold",
            })
        row = {
            "Product": name, "SKU": sku, "Qty to split": qty,
            "Sent to branches": res.get("allocated_total", 0),
            "Held at warehouse": wh,
        }
        summary_rows.append(row)

    detail_cols = ["Product", "SKU", "Qty to split", "Branch",
                   "Allocation", "Basis"]
    summary_cols = ["Product", "SKU", "Qty to split", "Sent to branches",
                    "Held at warehouse"]
    detail = pd.DataFrame(detail_rows, columns=detail_cols)
    summary = pd.DataFrame(summary_rows, columns=summary_cols)
    about = pd.DataFrame([
        {"field": "Products", "value": len(summary_rows)},
        {"field": "Units in total", "value": int(sum(r["Qty to split"] for r in summary_rows))},
        {"field": "Branches", "value": scope_lbl},
        {"field": "Generated", "value": _ts()},
    ])
    scope = "-".join(branch_codes) if branch_codes else "all"
    path = Path(out_dir or settings.out) / f"split_allocation_batch_{scope}_{_ts()}.xlsx"
    return write_workbook({"Split by predicted sales": detail,
                           "Summary": summary, "About": about}, path)


def split_batch_workbook(batch: dict, *, out_dir: Optional[Path] = None) -> Path:
    """Render an already-computed split result (one-off or weekly-order mode)
    into one workbook: a flat allocation table, a per-product summary and a
    small About sheet. ``batch`` is the dict built by the split routes."""
    rows = batch.get("rows", []) or []
    weekly = bool(batch.get("weekly"))
    detail_rows, summary_rows = [], []
    for r in rows:
        name = str(r.get("product") or r.get("sku") or "")
        sku = str(r.get("sku") or "")
        qty = int(r.get("qty") or 0)
        wh = int(r.get("warehouse") or 0)
        _basis = {"probe": "probe", "held": "no stock left", "untested": "untested"}
        for a in r.get("allocations", []):
            d = {"Product": name, "SKU": sku, "Qty to split": qty,
                 "Branch": a.get("branch"),
                 "Allocation": a.get("allocated"),
                 "Basis": _basis.get(a.get("kind"), "cover")}
            if weekly:
                d["Requested"] = a.get("requested")
            detail_rows.append(d)
        if wh:
            d = {"Product": name, "SKU": sku, "Qty to split": qty,
                 "Branch": "Warehouse (hold)",
                 "Allocation": wh, "Basis": "hold"}
            if weekly:
                d["Requested"] = None
            detail_rows.append(d)
        s = {"Product": name, "SKU": sku, "Qty to split": qty,
             "Sent to branches": int(r.get("allocated_total") or 0),
             "Held at warehouse": wh}
        if r.get("note"):
            s["Reasoning"] = str(r["note"])
        summary_rows.append(s)

    detail_cols = ["Product", "SKU", "Qty to split", "Branch",
                   "Allocation", "Basis"] + (["Requested"] if weekly else [])
    summary_cols = ["Product", "SKU", "Qty to split", "Sent to branches",
                    "Held at warehouse"]
    if any("Reasoning" in r for r in summary_rows):
        summary_cols.append("Reasoning")
    about = pd.DataFrame([
        {"field": "Mode", "value": "Weekly order" if weekly else "One off"},
        {"field": "Products", "value": len(summary_rows)},
        {"field": "Units in total", "value": int(batch.get("grand_total") or 0)},
        {"field": "Held at warehouse", "value": int(batch.get("warehouse_total") or 0)},
        {"field": "Branches", "value": str(batch.get("branches") or "")},
        {"field": "Generated", "value": _ts()},
    ])
    path = Path(out_dir or settings.out) / f"split_result_{_ts()}.xlsx"
    return write_workbook({
        "Split by predicted sales": pd.DataFrame(detail_rows, columns=detail_cols),
        "Summary": pd.DataFrame(summary_rows, columns=summary_cols),
        "About": about}, path)


def weekly_order_workbook(db, *, branch_code: str = "",
                          out_dir: Optional[Path] = None) -> Path:
    """A branch's order for the coming week: one row per product that needs
    topping up (cover target minus stock on hand), sorted by quantity, ready to
    hand to the distribution centre / supplier. A title block carries the
    branch, the week it covers and when it was generated."""
    from wms.analytics import weekly_forecast as wfc

    bc = (branch_code or "").strip()
    plan = allocation.weekly_allocation_plan(db, branch_code=bc)
    order_cols = ["SKU", "Product", "On hand", "Order qty"]
    if not plan.empty:
        plan = plan[plan["to_transport"] > 0]
    if plan.empty:
        lines = pd.DataFrame(columns=order_cols)
        branch_name = bc.upper() or "All branches"
    else:
        branch_name = str(plan["branch"].iloc[0])
        lines = (plan.rename(columns={
                     "sku": "SKU", "product": "Product",
                     "on_hand": "On hand", "to_transport": "Order qty"})
                 [order_cols]
                 .sort_values("Order qty", ascending=False)
                 .reset_index(drop=True))

    try:
        wk = pd.Timestamp(wfc.cached_panel()["weeks"][-1]) + pd.Timedelta(days=7)
        week_lbl = wk.date().isoformat()
    except Exception:                                    # noqa: BLE001
        week_lbl = ""
    total_units = int(lines["Order qty"].sum()) if not lines.empty else 0

    scope = bc.upper() or "all"
    path = Path(out_dir or settings.out) / f"weekly_order_{scope}_{_ts()}.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        book, sn, start = xw.book, "Weekly Order", 5
        title = book.add_format({"bold": True, "font_size": 14})
        label = book.add_format({"bold": True})
        head = book.add_format({"bold": True, "bg_color": "#1F3864",
                                "font_color": "white", "border": 1})
        tot = book.add_format({"bold": True, "top": 2})
        lines.to_excel(xw, sheet_name=sn, index=False, startrow=start, header=False)
        ws = xw.sheets[sn]
        ws.write(0, 0, "Weekly order", title)
        ws.write(1, 0, "Branch", label);          ws.write(1, 1, branch_name)
        ws.write(2, 0, "Week commencing", label); ws.write(2, 1, week_lbl)
        ws.write(3, 0, "Generated", label);       ws.write(3, 1, _ts())
        for c, col in enumerate(order_cols):
            ws.write(start, c, col, head)
        for c, w in enumerate((18, 52, 14, 14)):
            ws.set_column(c, c, w)
        if lines.empty:
            ws.write(start + 1, 0, "Nothing to order - every product is at or "
                                   "above its cover target.")
        else:
            end = start + len(lines)
            ws.write(end + 1, 0, "Total", tot)
            ws.write(end + 1, len(order_cols) - 1, total_units, tot)
            ws.freeze_panes(start + 1, 0)
            ws.autofilter(start, 0, end, len(order_cols) - 1)
    return path


def weekly_dispatch_branch_workbook(bb: dict, *, out_dir: Optional[Path] = None,
                                    doc_title: str = "Weekly order") -> Path:
    """One branch's order document (from a split-tool ``by_branch`` entry -
    see ``wms.web.routes._run_auto_weekly_order`` / ``_run_weekly_split`` /
    ``_run_split_batch``) as a standalone workbook: SKU, Product, Weekly
    sales, On hand and Requested (only when the source split tracked them -
    a one-off split doesn't) and Rec. Qty, ready to hand to the branch or the
    distribution centre. Used for both the single-branch download and each
    file inside a multi-branch ZIP."""
    lines = bb.get("lines", []) or []
    has_req = any(l.get("requested") for l in lines)
    has_oh = any(l.get("on_hand") is not None for l in lines)
    cols = ["SKU", "Product", "Weekly sales"]
    if has_oh:
        cols.append("On hand")
    if has_req:
        cols.append("Requested")
    cols.append("Rec. Qty")

    rows = []
    for l in lines:
        d = {"SKU": l.get("sku"), "Product": l.get("description"),
             "Weekly sales": l.get("predicted")}
        if has_oh:
            d["On hand"] = l.get("on_hand")
        if has_req:
            d["Requested"] = l.get("requested")
        d["Rec. Qty"] = l.get("rec")
        rows.append(d)
    df = pd.DataFrame(rows, columns=cols)

    branch_name = str(bb.get("branch") or "branch")
    scope = str(bb.get("code") or branch_name).replace(" ", "_")
    path = Path(out_dir or settings.out) / f"order_{scope}_{_ts()}.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        book, sn, start = xw.book, "Order", 4
        title = book.add_format({"bold": True, "font_size": 14})
        label = book.add_format({"bold": True})
        head = book.add_format({"bold": True, "bg_color": "#1F3864",
                                "font_color": "white", "border": 1})
        tot = book.add_format({"bold": True, "top": 2})
        df.to_excel(xw, sheet_name=sn, index=False, startrow=start, header=False)
        ws = xw.sheets[sn]
        ws.write(0, 0, doc_title, title)
        ws.write(1, 0, "Branch", label);    ws.write(1, 1, branch_name)
        ws.write(2, 0, "Generated", label); ws.write(2, 1, _ts())
        for c, col in enumerate(cols):
            ws.write(start, c, col, head)
        widths = {"SKU": 16, "Product": 46, "Weekly sales": 13, "On hand": 11,
                  "Requested": 12, "Rec. Qty": 11}
        for c, col in enumerate(cols):
            ws.set_column(c, c, widths.get(col, 14))
        if df.empty:
            ws.write(start + 1, 0, "Nothing to order for this branch.")
        else:
            end = start + len(df)
            ws.write(end + 1, 0, "Total", tot)
            ws.write(end + 1, len(cols) - 1, int(bb.get("rec_total") or 0), tot)
            ws.freeze_panes(start + 1, 0)
            ws.autofilter(start, 0, end, len(cols) - 1)
    return path


def branch_stats_workbook(db, *, out_dir: Optional[Path] = None) -> Path:
    path = Path(out_dir or settings.out) / f"branch_stats_{_ts()}.xlsx"
    return write_workbook({
        "Overall": pd.DataFrame([{"metric": k, "value": v}
                                 for k, v in statistics.overall_kpis(db).items()]),
        "By Branch": statistics.branch_comparison(db),
        "ABC": statistics.abc_classification(db),
    }, path)
