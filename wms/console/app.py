"""Mineazy WMS - Interactive Console (choose reports & run operations).

    python -m wms.console            # interactive menu
    python -m wms.console --demo     # non-interactive read-only tour
"""
from __future__ import annotations

import argparse

from wms.config import get_settings
from wms.console import ui
from wms.console.session import Context
from wms.console.menus import dashboard, backorders, analytics, reports

MENU = [
    ("Dashboard & KPIs", dashboard.run),
    ("Backorders — processing flow, entry & analysis", backorders.run),
    ("Sales, Forecasting & Allocation", analytics.run),
    ("Reports & Exports (Excel / CSV / charts / audit)", reports.run),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Mineazy WMS console")
    ap.add_argument("--user", default="admin", help="acting username (audit tracking)")
    ap.add_argument("--demo", action="store_true", help="non-interactive read-only tour")
    args = ap.parse_args()

    if args.demo:
        _demo(args.user)
        return

    ctx = Context(actor=args.user)
    s = get_settings()
    try:
        ui.banner("MINEAZY WMS  -  INTERACTIVE CONSOLE",
                  f"user: {ctx.user.username} ({ctx.user.role})   db: {s.database_url}")
        while True:
            c = ui.menu("Main Menu", [label for label, _ in MENU] + ["Exit"])
            if c in (None, "Exit"):
                ui.console.print("bye.")
                return
            try:
                dict(MENU)[c](ctx)
            except Exception as exc:
                ui.console.print(f"[red bold]error:[/red bold] {exc}")
                ui.pause()
    finally:
        ctx.close()


def _demo(user: str) -> None:
    from wms.analytics import loaders, statistics
    from wms.analytics import backorder_flow as bof
    from wms.exports import excel
    from wms.viz import charts

    ctx = Context(actor=user)
    try:
        ui.banner("MINEAZY WMS  -  CONSOLE DEMO (read-only)")
        ui.show_kpis(statistics.overall_kpis(ctx.db), "Overall KPIs")

        bo = loaders.back_orders_df(ctx.db)
        items = loaders.back_order_items_df(ctx.db)
        events = loaders.back_order_events_df(ctx.db)
        ui.show_kpis(bof.fulfilment_metrics(bo, items), "Backorder flow - fulfilment")
        ui.console.print(f"[bold]bottleneck stage:[/bold] {bof.bottleneck_stage(events)}")
        ui.show_df(bof.stage_funnel(bo), "Stage funnel")
        ui.show_df(bof.cycle_times(events), "Cycle times (days per stage)")
        ui.show_df(bof.aging(bo), "Aging of open back orders")
        ui.show_df(bof.by_branch(bo, items), "By branch")
        ui.show_df(bof.branch_backorders_vs_sales(ctx.db), "Backorders vs sales by branch")
        ui.show_df(bof.trend(bo, events).tail(8), "Weekly trend")

        wb = excel.backorder_flow_workbook(ctx.db)
        ui.console.print(f"[green]workbook:[/green] {wb}")
        pngs = charts.backorder_flow_pack(ctx.db)
        ui.console.print(f"[green]charts:[/green] {len(pngs)} PNG(s) in {wb.parent}")
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
