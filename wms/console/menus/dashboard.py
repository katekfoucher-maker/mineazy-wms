"""Dashboard & KPIs."""
from __future__ import annotations

from wms.analytics import loaders, statistics
from wms.analytics import backorder_flow as bof
from wms.models import Branch
from wms.console import ui


def run(ctx):
    while True:
        c = ui.menu("Dashboard & KPIs", [
            "Overall snapshot",
            "Backorder flow snapshot",
            "Per-branch snapshot",
            "Branch sales summary",
            "Back",
        ])
        if c in (None, "Back"):
            return
        if c == "Overall snapshot":
            ui.show_kpis(statistics.overall_kpis(ctx.db), "Overall KPIs")
            ui.pause()
        elif c == "Backorder flow snapshot":
            bo = loaders.back_orders_df(ctx.db)
            items = loaders.back_order_items_df(ctx.db)
            events = loaders.back_order_events_df(ctx.db)
            ui.show_kpis(bof.fulfilment_metrics(bo, items), "Backorder fulfilment")
            ui.console.print(f"[bold]bottleneck:[/bold] {bof.bottleneck_stage(events)}")
            ui.show_df(bof.stage_funnel(bo), "Stage funnel")
            ui.pause()
        elif c == "Per-branch snapshot":
            rows = ctx.db.query(Branch).order_by(Branch.name).all()
            pick = ui.menu("branch", [b.name for b in rows] + ["Back"])
            if pick not in (None, "Back"):
                b = next(x for x in rows if x.name == pick)
                ui.show_kpis(statistics.branch_kpis(ctx.db, b.id), f"{b.name} KPIs")
                ui.pause()
        elif c == "Branch sales summary":
            ui.show_df(statistics.branch_sales_summary(ctx.db),
                       "Branch sales (last 90 days)")
            ui.pause()
