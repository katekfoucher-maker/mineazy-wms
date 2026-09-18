"""Branch sales analysis, forecasting & allocation."""
from __future__ import annotations

import pandas as pd

from wms.analytics import statistics, forecast, allocation
from wms.models import Branch, Product
from wms.console import ui


def run(ctx):
    while True:
        c = ui.menu("Sales, Forecasting & Allocation", [
            "Overall statistics",
            "Per-branch statistics",
            "Branch comparison",
            "Branch sales summary",
            "ABC classification (by sales value)",
            "Weekly sales trend",
            "Demand forecast / suggested branch orders",
            "Allocate one product (supply-constrained)",
            "Suggested orders per product x branch",
            "Back",
        ])
        if c in (None, "Back"):
            return
        {
            "Overall statistics": _overall,
            "Per-branch statistics": _branch,
            "Branch comparison": _comparison,
            "Branch sales summary": _sales_summary,
            "ABC classification (by sales value)": _abc,
            "Weekly sales trend": _trend,
            "Demand forecast / suggested branch orders": _forecast,
            "Allocate one product (supply-constrained)": _alloc_one,
            "Suggested orders per product x branch": _alloc_plan,
        }[c](ctx)


def _sales_summary(ctx):
    ui.show_df(statistics.branch_sales_summary(ctx.db), "Branch sales (90 days)")
    ui.pause()


def _pick_branch(ctx):
    rows = ctx.db.query(Branch).order_by(Branch.name).all()
    pick = ui.menu("branch", [b.name for b in rows] + ["Cancel"])
    if pick in (None, "Cancel"):
        return None
    return next(b for b in rows if b.name == pick)


def _overall(ctx):
    ui.show_kpis(statistics.overall_kpis(ctx.db), "Overall statistics")
    ui.pause()


def _branch(ctx):
    b = _pick_branch(ctx)
    if b:
        ui.show_kpis(statistics.branch_kpis(ctx.db, b.id), f"{b.name} statistics")
        ui.pause()


def _comparison(ctx):
    ui.show_df(statistics.branch_comparison(ctx.db), "Branch comparison")
    ui.pause()


def _abc(ctx):
    df = statistics.abc_classification(ctx.db)
    ui.show_df(df, "ABC classification", max_rows=60)
    if not df.empty:
        ui.show_df(df.groupby("class").agg(items=("sku", "count"),
                                           value=("value", "sum")).reset_index(),
                   "ABC summary")
    ui.pause()


def _trend(ctx):
    b = _pick_branch(ctx) if ui.confirm("scope to one branch?") else None
    ui.show_df(statistics.sales_trend(ctx.db, branch_id=b.id if b else None),
               "Sales trend (weekly)", max_rows=60)
    ui.pause()


def _forecast(ctx):
    b = _pick_branch(ctx) if ui.confirm("scope to one branch?") else None
    df = forecast.forecast_table(ctx.db, branch_id=b.id if b else None)
    ui.show_df(df, "Demand forecast", max_rows=60)
    so = forecast.suggested_orders(ctx.db, branch_id=b.id if b else None)
    ui.show_df(so, "Suggested branch orders (anticipated)", max_rows=60)
    if not so.empty and ui.confirm("export suggested orders to Excel?"):
        from wms.exports import excel
        ui.open_file(excel.suggested_orders_workbook(ctx.db, branch_id=b.id if b else None))
    ui.pause()


def _alloc_one(ctx):
    sku = (ui.ask_text("product SKU") or "").strip()
    p = ctx.db.query(Product).filter(Product.sku == sku).first()
    if not p:
        ui.console.print("[red]unknown SKU[/red]")
        return
    avail = ui.ask_int(f"available qty of {p.sku} to allocate", minimum=1)
    if avail is None:
        return
    res = allocation.allocate_product(ctx.db, product_id=p.id, available_qty=avail)
    ui.console.print(f"[bold]{res.get('product')}[/bold] allocate {avail} "
                     f"(policy {res.get('policy')})")
    ui.show_df(pd.DataFrame(res.get("allocations", [])), "Proposed allocation")
    ui.console.print(f"allocated {res.get('allocated_total')}, "
                     f"unallocated {res.get('unallocated')}")
    ui.pause()


def _alloc_plan(ctx):
    df = allocation.allocation_plan(ctx.db)
    ui.show_df(df, "Allocation plan", max_rows=80)
    if not df.empty and ui.confirm("export to Excel?"):
        from wms.exports import excel
        ui.open_file(excel.allocation_workbook(ctx.db))
    ui.pause()
