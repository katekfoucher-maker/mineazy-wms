"""Reports & Exports: Excel, CSV, charts, audit trail."""
from __future__ import annotations

import pandas as pd

from wms.analytics import loaders
from wms.exports import excel, csv_export
from wms.viz import charts
from wms.models import AuditLog, User
from wms.console import ui


def run(ctx):
    while True:
        c = ui.menu("Reports & Exports", [
            "Audit trail - recent",
            "Audit trail - by user",
            "Excel - backorder flow workbook",
            "Excel - delivery-note fill workbook",
            "Excel - branch sales",
            "Excel - suggested orders",
            "Excel - allocation plan",
            "Excel - branch statistics pack",
            "CSV - back orders",
            "CSV - sales history",
            "Charts - dashboard pack",
            "Charts - backorder-flow pack",
            "Back",
        ])
        if c in (None, "Back"):
            return
        try:
            _dispatch(ctx, c)
        except Exception as e:
            ui.console.print(f"[red]{e}[/red]")
            ui.pause()


def _dispatch(ctx, c):
    if c == "Audit trail - recent":
        rows = ctx.db.query(AuditLog).order_by(AuditLog.id.desc()).limit(60).all()
        ui.show_df(_audit_df(rows), "Audit trail", max_rows=60)
        ui.pause()
    elif c == "Audit trail - by user":
        users = ctx.db.query(User).order_by(User.username).all()
        pick = ui.menu("user", [u.username for u in users] + ["Cancel"])
        if pick in (None, "Cancel"):
            return
        u = next(x for x in users if x.username == pick)
        rows = (ctx.db.query(AuditLog).filter(AuditLog.user_id == u.id)
                .order_by(AuditLog.id.desc()).limit(100).all())
        ui.show_df(_audit_df(rows), f"Audit trail - {u.username}", max_rows=80)
        ui.pause()
    elif c == "Excel - backorder flow workbook":
        ui.open_file(excel.backorder_flow_workbook(ctx.db))
    elif c == "Excel - delivery-note fill workbook":
        ui.open_file(excel.backorder_workbook(ctx.db))
    elif c == "Excel - branch sales":
        ui.open_file(excel.sales_workbook(ctx.db))
    elif c == "Excel - suggested orders":
        ui.open_file(excel.suggested_orders_workbook(ctx.db))
    elif c == "Excel - allocation plan":
        ui.open_file(excel.allocation_workbook(ctx.db))
    elif c == "Excel - branch statistics pack":
        ui.open_file(excel.branch_stats_workbook(ctx.db))
    elif c == "CSV - back orders":
        ui.open_file(csv_export.dataframe_to_csv(loaders.back_orders_df(ctx.db), "back_orders"))
    elif c == "CSV - sales history":
        ui.open_file(csv_export.dataframe_to_csv(loaders.sales_df(ctx.db, days=180), "sales"))
    elif c == "Charts - dashboard pack":
        _charts(ctx, charts.dashboard_pack)
    elif c == "Charts - backorder-flow pack":
        _charts(ctx, charts.backorder_flow_pack)


def _charts(ctx, fn):
    paths = fn(ctx.db)
    for p in paths:
        ui.open_file(p)
    if not paths:
        ui.console.print("[yellow]no charts produced[/yellow]")
    ui.pause()


def _audit_df(rows):
    return pd.DataFrame([{
        "ts": r.created_at, "entity": r.entity_type, "entity_id": r.entity_id,
        "action": r.action, "reason": r.reason, "user_id": r.user_id} for r in rows])
