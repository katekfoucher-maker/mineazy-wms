"""Backorder processing flow: enter, advance stages, and analyse."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd

from wms.analytics import loaders
from wms.analytics import backorders as dn_an          # delivery-note fill analysis
from wms.analytics import backorder_flow as bof        # backorder flow analysis
from wms.enums import STAGE_LABEL, BackOrderStage
from wms.models import Branch, Product
from wms.services import backorders as dn_svc
from wms.services import backorder_entry as bo_entry
from wms.services import backorder_stages as bo_stage
from wms.console import ui


def run(ctx):
    while True:
        c = ui.menu("Backorders", [
            "Active back orders (browse / filter by stage)",
            "Enter a back order (manual)",
            "Advance a back order (change stage)",
            "Enter a delivery-note document (auto back order on shortfall)",
            "Import a delivery note from CSV",
            "--- ANALYSIS ---",
            "Backorder flow analysis - OVERALL",
            "Backorder flow analysis - BY BRANCH",
            "Backorders vs sales by branch",
            "Delivery-note fill analysis",
            "Export backorder-flow workbook (Excel)",
            "Export delivery-note fill workbook (Excel)",
            "Chart pack (Matplotlib / Seaborn)",
            "Back",
        ])
        if c in (None, "Back", "--- ANALYSIS ---"):
            if c == "--- ANALYSIS ---":
                continue
            return
        {
            "Active back orders (browse / filter by stage)": _active,
            "Enter a back order (manual)": _enter_bo,
            "Advance a back order (change stage)": _advance,
            "Enter a delivery-note document (auto back order on shortfall)": _enter_dn,
            "Import a delivery note from CSV": _import_csv,
            "Backorder flow analysis - OVERALL": _flow_overall,
            "Backorder flow analysis - BY BRANCH": _flow_branch,
            "Backorders vs sales by branch": _bo_vs_sales,
            "Delivery-note fill analysis": _dn_fill,
            "Export backorder-flow workbook (Excel)": _export_flow,
            "Export delivery-note fill workbook (Excel)": _export_dn,
            "Chart pack (Matplotlib / Seaborn)": _charts,
        }[c](ctx)


def _pick_branch(ctx):
    rows = ctx.db.query(Branch).order_by(Branch.name).all()
    pick = ui.menu("branch", [f"{b.code}  {b.name}" for b in rows] + ["Cancel"])
    if pick in (None, "Cancel"):
        return None
    return next(b for b in rows if pick.startswith(b.code + "  "))


# ----------------------------------------------------------------------
# Active back orders + entry + stage advance
# ----------------------------------------------------------------------
def _active(ctx):
    stage = ui.menu("stage filter", ["(all open)"] + [STAGE_LABEL[s] for s in BackOrderStage]
                    + ["Back"])
    if stage in (None, "Back"):
        return
    st = None
    open_only = stage == "(all open)"
    if not open_only:
        st = next(s.value for s in BackOrderStage if STAGE_LABEL[s] == stage)
    rows = bo_entry.list_back_orders(ctx.db, stage=st, open_only=open_only)
    ui.show_df(pd.DataFrame([{
        "bo_no": b.bo_no, "branch": b.branch.name, "stage": STAGE_LABEL[BackOrderStage(b.stage)],
        "items": len(b.items), "ordered": b.qty_ordered, "fulfilled": b.qty_fulfilled,
        "fulfil_%": round(b.fulfil_pct * 100, 1), "status": b.status,
        "priority": b.priority, "po_no": b.po_no or "", "source": b.source,
    } for b in rows]), f"Active back orders - {stage}", max_rows=80)
    ui.pause()


def _enter_bo(ctx):
    branch = _pick_branch(ctx)
    if not branch:
        return
    priority = ui.menu("priority", ["NORMAL", "HIGH", "LOW"]) or "NORMAL"
    notes = ui.ask_text("notes") or None
    items = []
    ui.console.print("[dim]enter items; blank SKU to finish[/dim]")
    while True:
        sku = ui.ask_text("  SKU")
        if not sku:
            break
        p = ctx.db.query(Product).filter(Product.sku == sku.strip()).first()
        if not p:
            ui.console.print("  [red]unknown SKU[/red]")
            continue
        qty = ui.ask_int(f"  qty for {p.sku}", minimum=1)
        if qty:
            items.append({"sku": p.sku, "qty": qty})
            ui.console.print(f"  [green]+[/green] {p.sku} x {qty}")
    if not items or not ui.confirm(f"Create back order for {branch.name} ({len(items)} items)?"):
        return
    try:
        bo = bo_entry.create_back_order(ctx.db, branch_id=branch.id, items=items,
                                        priority=priority, notes=notes, user_id=ctx.user.id)
        ui.console.print(f"[green]created[/green] {bo.bo_no} at {bo.stage}")
    except Exception as e:
        ui.console.print(f"[red]{e}[/red]")
    ui.pause()


def _advance(ctx):
    no = ui.ask_text("back order no (BO-######)")
    if not no:
        return
    try:
        bo = bo_entry.get(ctx.db, no)
    except Exception as e:
        ui.console.print(f"[red]{e}[/red]")
        return
    doc = bo_entry.serialize(bo)
    ui.console.print(f"[bold]{doc['bo_no']}[/bold]  {doc['branch']}  "
                     f"stage={STAGE_LABEL[BackOrderStage(doc['stage'])]}  status={doc['status']}")
    ui.show_df(pd.DataFrame(doc["items"]), "items")
    allowed = bo_stage.allowed_next(bo.stage)
    if not allowed:
        ui.console.print("[yellow]terminal stage - nothing to do[/yellow]")
        ui.pause()
        return
    labels = {STAGE_LABEL[BackOrderStage(s)]: s for s in allowed}
    pick = ui.menu("advance to", list(labels) + ["Cancel"])
    if pick in (None, "Cancel"):
        return
    to = labels[pick]
    kw = {}
    from wms.enums import STAGE_ITEM_QTY
    qfield = STAGE_ITEM_QTY.get(BackOrderStage(to))
    items = {}
    if qfield:
        for it in doc["items"]:
            v = ui.ask_int(f"  {qfield} for {it['sku']}", default=it["qty_ordered"], minimum=0)
            if v is not None:
                items[it["item_id"]] = v
    kw["items"] = items or None
    kw["note"] = ui.ask_text("note") or None
    try:
        bo = bo_stage.advance(ctx.db, bo_no=no, to_stage=to, user_id=ctx.user.id, **kw)
        ui.console.print(f"[green]{bo.bo_no} -> {STAGE_LABEL[BackOrderStage(bo.stage)]}[/green]")
    except Exception as e:
        ui.console.print(f"[red]{e}[/red]")
    ui.pause()


# ----------------------------------------------------------------------
# Delivery-note entry / stages (unchanged mechanics)
# ----------------------------------------------------------------------
def _enter_dn(ctx):
    branch = _pick_branch(ctx)
    if not branch:
        return
    from_loc = ui.ask_text("from location", default="DC") or "DC"
    comment = ui.ask_text("comment", default="stock transfer") or None
    lines = []
    ui.console.print("[dim]enter lines; blank SKU to finish[/dim]")
    while True:
        sku = ui.ask_text("  SKU")
        if not sku:
            break
        p = ctx.db.query(Product).filter(Product.sku == sku.strip()).first()
        if not p:
            ui.console.print("  [red]unknown SKU[/red]")
            continue
        req = ui.ask_int(f"  requested qty for {p.sku}", minimum=1)
        sent = ui.ask_int("  sent qty (blank = 0)", default=0, minimum=0)
        if req:
            lines.append({"sku": p.sku, "requested_qty": req, "sent_qty": sent})
    if not lines or not ui.confirm(f"Save to {branch.name} ({len(lines)} lines)?"):
        return
    try:
        dn = dn_svc.enter_delivery_note(ctx.db, branch_id=branch.id, lines=lines,
                                        from_location_label=from_loc, doc_date=date.today(),
                                        comment=comment, user_id=ctx.user.id)
        doc = dn_svc.serialize(ctx.db, dn)
        ui.show_df(pd.DataFrame(doc["lines"]), f"Saved {doc['dn_no']}")
        n = sum(1 for l in doc["lines"] if l["backorder_qty"] > 0)
        ui.console.print(f"[yellow]{n} shortfall line(s) -> back order created[/yellow]")
    except Exception as e:
        ui.console.print(f"[red]{e}[/red]")
    ui.pause()


def _import_csv(ctx):
    branch = _pick_branch(ctx)
    if not branch:
        return
    path = ui.ask_text("CSV path (cols: SKU/Item No, Req. Qty, Sent Qty)")
    if not path or not Path(path).exists():
        ui.console.print("[red]file not found[/red]")
        return
    lines = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
            sku = row.get("sku") or row.get("item no") or row.get("item_no")
            if not sku:
                continue
            req = int(float(row.get("req. qty") or row.get("requested_qty") or row.get("req qty") or 0))
            sent_raw = row.get("sent qty") or row.get("sent_qty") or ""
            lines.append({"sku": sku, "requested_qty": req,
                          "sent_qty": int(float(sent_raw)) if sent_raw else 0})
    if not lines or not ui.confirm(f"import {len(lines)} lines?"):
        return
    try:
        dn = dn_svc.enter_delivery_note(ctx.db, branch_id=branch.id, lines=lines,
                                        doc_date=date.today(), user_id=ctx.user.id)
        ui.console.print(f"[green]imported[/green] {dn.dn_no}")
    except Exception as e:
        ui.console.print(f"[red]{e}[/red]")
    ui.pause()




# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def _flow_frames(ctx, branch_id=None):
    return (loaders.back_orders_df(ctx.db, branch_id=branch_id),
            loaders.back_order_items_df(ctx.db, branch_id=branch_id),
            loaders.back_order_events_df(ctx.db))


def _show_flow(ctx, bo, items, events, title):
    ui.show_kpis(bof.fulfilment_metrics(bo, items), title)
    b = bof.bottleneck_stage(events)
    if b:
        ui.console.print(f"[bold]bottleneck stage:[/bold] {b}")
    ui.show_df(bof.stage_funnel(bo), "Stage funnel")
    ui.show_df(bof.cycle_times(events), "Cycle times (days per stage)")
    ui.show_df(bof.aging(bo), "Aging of open back orders")
    ui.show_df(bof.by_branch(bo, items), "By branch")
    ui.show_df(bof.top_items(items, top=15), "Top items outstanding")
    ui.show_df(bof.trend(bo, events), "Weekly trend")
    ui.pause()


def _flow_overall(ctx):
    bo, items, events = _flow_frames(ctx)
    _show_flow(ctx, bo, items, events, "Backorder flow - OVERALL")


def _flow_branch(ctx):
    b = _pick_branch(ctx)
    if not b:
        return
    bo, items, events = _flow_frames(ctx, b.id)
    _show_flow(ctx, bo, items, events, f"Backorder flow - {b.name}")


def _bo_vs_sales(ctx):
    ui.show_df(bof.branch_backorders_vs_sales(ctx.db),
               "Backorders vs sales, by branch (last 90 days)", max_rows=40)
    ui.pause()


def _dn_fill(ctx):
    dl = loaders.dn_lines_df(ctx.db)
    ui.show_kpis(dn_an.overall_summary(dl), "Delivery-note fill - OVERALL")
    ui.show_df(dn_an.by_fill_status(dl), "By fill status (FULL / PARTIAL / NIL)")
    ui.show_df(dn_an.by_branch(dl), "By branch")
    ui.show_df(dn_an.by_item(dl, top=15), "Top 15 short items")
    ui.pause()


def _export_flow(ctx):
    from wms.exports import excel
    branch = _pick_branch(ctx) if ui.confirm("scope to one branch?") else None
    ui.open_file(excel.backorder_flow_workbook(ctx.db, branch_id=branch.id if branch else None))
    ui.pause()


def _export_dn(ctx):
    from wms.exports import excel
    ui.open_file(excel.backorder_workbook(ctx.db))
    ui.pause()


def _charts(ctx):
    from wms.viz import charts
    paths = charts.backorder_flow_pack(ctx.db)
    for p in paths:
        ui.open_file(p)
    if not paths:
        ui.console.print("[yellow]no charts produced[/yellow]")
    ui.pause()
