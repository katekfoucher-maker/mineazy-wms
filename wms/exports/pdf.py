"""PDF exports (fpdf2). Currently: the weekly allocation plan as a printable
order sheet."""
from __future__ import annotations

import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# the built-in PDF fonts are latin-1 only; map the punctuation that turns up in
# HansaWorld product names and drop anything else still out of range
_PUNCT = {"（": "(", "）": ")", "‘": "'", "’": "'",
          "“": '"', "”": '"', "–": "-", "—": "-",
          " ": " ", "…": "...", "×": "x", "′": "'",
          "″": '"', "、": ",", "。": "."}


def _lat1(s) -> str:
    s = str(s)
    for a, b in _PUNCT.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("latin-1", "replace").decode("latin-1")

from wms.config import get_settings
from wms.analytics import allocation

settings = get_settings()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _out(name: str, out_dir: Optional[Path]) -> Path:
    p = Path(out_dir or settings.out) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def allocation_plan_pdf(db, *, branch_code: str = "",
                        out_dir: Optional[Path] = None) -> Path:
    """The weekly allocation plan for a branch as a landscape PDF: one row per
    product, **highest recent weekly sales first**, products not currently being
    sold at the branch left out. Columns: SKU, Product, recent weekly sales,
    forecast weekly demand, cover target, on hand, order qty."""
    from fpdf import FPDF
    from wms.analytics import weekly_forecast as wfc

    bc = (branch_code or "").strip()
    plan = allocation.weekly_allocation_plan(db, branch_code=bc)

    # pull recent weekly sales / weeks-sold from the forecast state to rank by
    # and to drop products with no recent sales at this branch
    st = wfc.cached_run().get("state")
    if st is not None and not st.empty:
        s = st.copy()
        if bc:
            s = s[(s["branch"].str.upper() == bc.upper())
                  | s["branch_name"].str.lower().str.startswith(bc.lower())]
        s = (s[["sku", "recent_sales", "weeks_sold"]]
             .groupby("sku", as_index=False).sum())
        plan = plan.merge(s, on="sku", how="left")
    if "recent_sales" not in plan.columns:
        plan["recent_sales"] = plan.get("weekly_demand", 0)
        plan["weeks_sold"] = 1
    plan["recent_sales"] = pd.to_numeric(plan["recent_sales"],
                                         errors="coerce").fillna(0)
    plan = plan[plan["recent_sales"] > 0]                 # being sold at the branch
    plan = plan.sort_values(["recent_sales", "weekly_demand"],
                            ascending=False).reset_index(drop=True)

    branch_name = (str(plan["branch"].iloc[0]) if not plan.empty
                   else (bc.upper() or "All branches"))
    try:
        wk = pd.Timestamp(wfc.cached_panel()["weeks"][-1]) + pd.Timedelta(days=7)
        week_lbl = wk.date().isoformat()
    except Exception:                                     # noqa: BLE001
        week_lbl = ""
    order_total = int(pd.to_numeric(plan.get("to_transport", 0),
                                    errors="coerce").fillna(0).sum())

    cols = [("SKU", 32), ("Product", 155), ("Wkly sales", 26),
            ("On hand", 26), ("Order qty", 26)]
    keys = ["sku", "product", "recent_sales", "on_hand", "to_transport"]

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_title(f"Weekly allocation plan - {branch_name}")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, "Weekly allocation plan", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _lat1(f"Branch: {branch_name}    Week commencing: {week_lbl}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    def header():
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_fill_color(31, 56, 100)
        pdf.set_text_color(255)
        for name, w in cols:
            pdf.cell(w, 7, name, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_text_color(0)

    header()
    pdf.set_font("Helvetica", "", 8)
    fill = False
    for _, r in plan.iterrows():
        if pdf.will_page_break(6):
            pdf.add_page()
            header()
            pdf.set_font("Helvetica", "", 8)
        pdf.set_fill_color(244, 246, 250)
        for (name, w), k in zip(cols, keys):
            v = r.get(k, "")
            if k == "product":
                pdf.cell(w, 6, _lat1(v)[:98], border="LR", align="L", fill=fill)
            elif k == "sku":
                pdf.cell(w, 6, _lat1(v)[:22], border="LR", align="L", fill=fill)
            else:
                try:
                    txt = f"{int(round(float(v))):,}"
                except (TypeError, ValueError):
                    txt = _lat1(v)
                pdf.cell(w, 6, txt, border="LR", align="R", fill=fill)
        pdf.ln()
        fill = not fill
    pdf.set_font("Helvetica", "B", 8.5)
    pre = sum(w for _n, w in cols[:-1])
    pdf.cell(pre, 7, "Total order qty", border="T", align="R")
    pdf.cell(cols[-1][1], 7, f"{order_total:,}", border="T", align="R")

    scope = bc.upper() or "all"
    path = _out(f"allocation_plan_{scope}_{_ts()}.pdf", out_dir)
    pdf.output(str(path))
    return path


def split_allocation_pdf(db, *, sku: str, qty: int, branch_codes=None,
                         out_dir: Optional[Path] = None) -> Path:
    """The 'split a quantity by predicted sales' result as a portrait PDF."""
    from fpdf import FPDF

    res = allocation.allocate_by_forecast(db, sku=sku, qty=int(qty or 0),
                                          branch_codes=branch_codes)
    allocs = res.get("allocations", [])
    scope_lbl = ", ".join(branch_codes) if branch_codes else "all forecast branches"
    _prod = str(res.get("product") or sku)
    _desc = str(res.get("description") or "").strip()
    prod_line = f"{_prod} - {_desc}" if _desc and _desc != _prod else _prod

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_title("Split by predicted sales")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, "Split by predicted sales", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _lat1(
        f"Product: {prod_line}\n"
        f"Quantity to split: {int(qty or 0):,}\n"
        f"Branches: {scope_lbl}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    cols = [("Branch", 145), ("Allocation", 35)]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(31, 56, 100)
    pdf.set_text_color(255)
    for name, w in cols:
        pdf.cell(w, 7, name, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(0)
    pdf.set_font("Helvetica", "", 9)
    fill = False
    for a in allocs:
        pdf.set_fill_color(244, 246, 250)
        pdf.cell(cols[0][1], 6, _lat1(a["branch"])[:56], border="LR", fill=fill)
        pdf.cell(cols[1][1], 6, f"{int(a['allocated']):,}", border="LR",
                 align="R", fill=fill)
        pdf.ln()
        fill = not fill
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(cols[0][1], 7, "Total", border="T", align="R")
    pdf.cell(cols[1][1], 7, f"{int(res.get('allocated_total', 0)):,}",
             border="T", align="R")

    scope = "-".join(branch_codes) if branch_codes else "all"
    path = _out(f"split_allocation_{scope}_{_ts()}.pdf", out_dir)
    pdf.output(str(path))
    return path


def split_allocation_batch_pdf(db, *, pairs, branch_codes=None,
                               out_dir: Optional[Path] = None) -> Path:
    """Several products split at once, in one portrait PDF: a header block then
    one predicted-sales / allocation table per product."""
    from fpdf import FPDF

    scope_lbl = ", ".join(branch_codes) if branch_codes else "all forecast branches"
    clean = []
    for sku, qty in pairs:
        sku = str(sku or "").strip()
        try:
            qty = int(float(qty))
        except (TypeError, ValueError):
            qty = 0
        if sku and qty > 0:
            clean.append((sku, qty))

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_title("Split by predicted sales")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, "Split by predicted sales", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _lat1(
        f"Products: {len(clean)}\n"
        f"Units in total: {sum(q for _s, q in clean):,}\n"
        f"Branches: {scope_lbl}\n"
        f"Generated: {_ts()}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    cols = [("Branch", 145), ("Allocation", 35)]
    for sku, qty in clean:
        res = allocation.allocate_by_forecast(db, sku=sku, qty=qty,
                                              branch_codes=branch_codes)
        _prod = str(res.get("product") or sku)
        _desc = str(res.get("description") or "").strip()
        prod_line = f"{_prod} - {_desc}" if _desc and _desc != _prod else _prod

        if pdf.will_page_break(40):
            pdf.add_page()
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, _lat1(f"{prod_line}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _lat1(f"{sku}   Quantity to split: {qty:,}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(31, 56, 100)
        pdf.set_text_color(255)
        for name, w in cols:
            pdf.cell(w, 7, name, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_text_color(0)
        pdf.set_font("Helvetica", "", 9)
        fill = False
        for a in res.get("allocations", []):
            pdf.set_fill_color(244, 246, 250)
            tag = {"probe": "  (probe)", "seed": "  (new branch)"}.get(a.get("kind"), "")
            lbl = a["branch"] + tag
            pdf.cell(cols[0][1], 6, _lat1(lbl)[:56], border="LR", fill=fill)
            pdf.cell(cols[1][1], 6, f"{int(a['allocated']):,}", border="LR",
                     align="R", fill=fill)
            pdf.ln()
            fill = not fill
        wh = int(res.get("warehouse", 0) or 0)
        if wh:
            pdf.set_fill_color(244, 246, 250)
            pdf.cell(cols[0][1], 6, _lat1("Warehouse (hold)"), border="LR", fill=fill)
            pdf.cell(cols[1][1], 6, f"{wh:,}", border="LR", align="R", fill=fill)
            pdf.ln()
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(cols[0][1], 7, "Total", border="T", align="R")
        pdf.cell(cols[1][1], 7, f"{int(res.get('allocated_total', 0)) + wh:,}",
                 border="T", align="R")
        pdf.ln()

    scope = "-".join(branch_codes) if branch_codes else "all"
    path = _out(f"split_allocation_batch_{scope}_{_ts()}.pdf", out_dir)
    pdf.output(str(path))
    return path


def split_batch_pdf(batch: dict, *, out_dir: Optional[Path] = None) -> Path:
    """Render an already-computed split result (one-off or weekly-order mode)
    into one portrait PDF: a header block then one table per product."""
    from fpdf import FPDF

    rows = batch.get("rows", []) or []
    weekly = bool(batch.get("weekly"))
    title = "Split by predicted sales" + (" (weekly order)" if weekly else "")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_title("Split by predicted sales")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, _lat1(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _lat1(
        f"Products: {len(rows)}\n"
        f"Units in total: {int(batch.get('grand_total') or 0):,}\n"
        f"Held at warehouse: {int(batch.get('warehouse_total') or 0):,}\n"
        f"Branches: {batch.get('branches') or ''}\n"
        f"Generated: {_ts()}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if weekly:
        cols = [("Branch", 100), ("Requested", 40), ("Allocation", 40)]
    else:
        cols = [("Branch", 145), ("Allocation", 35)]

    for r in rows:
        name = str(r.get("product") or r.get("sku") or "")
        sku = str(r.get("sku") or "")
        qty = int(r.get("qty") or 0)
        wh = int(r.get("warehouse") or 0)

        if pdf.will_page_break(40):
            pdf.add_page()
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, _lat1(name), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _lat1(f"{sku}   Quantity to split: {qty:,}"),
                 new_x="LMARGIN", new_y="NEXT")
        if r.get("note"):
            pdf.set_text_color(150, 0, 0)
            pdf.multi_cell(0, 5, _lat1(str(r["note"])), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0)
        pdf.ln(1)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(31, 56, 100)
        pdf.set_text_color(255)
        for cname, w in cols:
            pdf.cell(w, 7, cname, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_text_color(0)
        pdf.set_font("Helvetica", "", 9)
        fill = False
        for a in r.get("allocations", []):
            pdf.set_fill_color(244, 246, 250)
            tag = {"probe": "  (probe)", "seed": "  (new branch)"}.get(a.get("kind"), "")
            lbl = str(a.get("branch", "")) + tag
            pdf.cell(cols[0][1], 6, _lat1(lbl)[:44], border="LR", fill=fill)
            if weekly:
                rq = a.get("requested")
                pdf.cell(cols[1][1], 6, "" if rq is None else f"{int(rq):,}",
                         border="LR", align="R", fill=fill)
            pdf.cell(cols[-1][1], 6, f"{int(a.get('allocated') or 0):,}", border="LR",
                     align="R", fill=fill)
            pdf.ln()
            fill = not fill
        if wh:
            pdf.set_fill_color(244, 246, 250)
            pdf.cell(cols[0][1], 6, _lat1("Warehouse (hold)"), border="LR", fill=fill)
            for _c, w in cols[1:-1]:
                pdf.cell(w, 6, "", border="LR", fill=fill)
            pdf.cell(cols[-1][1], 6, f"{wh:,}", border="LR", align="R", fill=fill)
            pdf.ln()
        pre = sum(w for _c, w in cols[:-1])
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(pre, 7, "Total", border="T", align="R")
        pdf.cell(cols[-1][1], 7, f"{int(r.get('allocated_total') or 0) + wh:,}",
                 border="T", align="R")
        pdf.ln()

    path = _out(f"split_result_{_ts()}.pdf", out_dir)
    pdf.output(str(path))
    return path


def weekly_dispatch_pdf(batch: dict, *, out_dir: Optional[Path] = None,
                        title: str = "weekly order allocation") -> Path:
    """One filled Stock-Movement dispatch note per branch: the branch's own
    requested lines (when there were any - a one-off split has none) with the
    **Rec. Qty** column set from the split (quantities based on predicted
    sales, snapped to the request when close, for a weekly order)."""
    from fpdf import FPDF

    notes = batch.get("by_branch", []) or []
    has_req = any(ln.get("requested") for bb in notes for ln in (bb.get("lines") or []))
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_title(f"Stock Movement - {title}")

    cols = [("Item No", 34)]
    if has_req:
        cols.append(("Req. Qty", 20))
    cols.append(("Description", 116 if has_req else 136))
    cols.append(("Rec. Qty", 22))

    def _label(x, y, text):
        pdf.set_xy(x, y)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(90)
        pdf.cell(0, 3, _lat1(text), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)

    def _value(x, y, text, size=11):
        pdf.set_xy(x, y + 3)
        pdf.set_font("Helvetica", "", size)
        pdf.cell(0, 5, _lat1(text), new_x="LMARGIN", new_y="NEXT")

    for bb in (notes or [None]):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_xy(pdf.l_margin, 14)
        pdf.cell(120, 8, "Mineazy Mining Solutions")
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_xy(130, 16)
        pdf.cell(0, 8, "Stock Movement", align="R", new_x="LMARGIN", new_y="NEXT")
        if bb and bb.get("doc_id"):
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_xy(130, 24)
            pdf.cell(0, 7, _lat1(str(bb["doc_id"])), align="R",
                     new_x="LMARGIN", new_y="NEXT")

        if not bb:
            pdf.set_xy(pdf.l_margin, 40)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 6, "No branch order files.")
            break

        top = 32
        _label(pdf.l_margin, top, "Date");        _value(pdf.l_margin, top, bb.get("date") or "")
        _label(pdf.l_margin, top + 12, "To Location")
        _value(pdf.l_margin, top + 12, bb.get("code") or "")
        _label(pdf.l_margin, top + 24, "Name")
        _value(pdf.l_margin, top + 24, bb.get("branch") or "")
        _label(pdf.l_margin, top + 40, "From Location");   _value(pdf.l_margin, top + 40, "DC")
        _label(pdf.l_margin, top + 52, "Name")
        _value(pdf.l_margin, top + 52, "DISTRIBUTION CENTER")
        _label(pdf.l_margin, top + 68, "Comment")
        _value(pdf.l_margin, top + 68, title, size=10)
        pdf.set_xy(pdf.l_margin, top + 80)

        def _header_row():
            pdf.set_font("Helvetica", "B", 8.5)
            for name, w in cols:
                pdf.cell(w, 7, name, border="B")
            pdf.ln()
            return pdf.get_x(), pdf.get_y()

        x0, y0 = _header_row()
        pdf.set_font("Helvetica", "", 8.5)
        for ln in bb.get("lines", []):
            if pdf.will_page_break(6):
                # close the box for the rows printed so far on this page
                # before moving on - x0/y0 belong to THIS page and must not
                # be reused after add_page() resets the page's coordinates,
                # or the box ends up with a bogus (often negative) height
                # and gets drawn across the wrong part of the next page,
                # visually overlapping/duplicating rows near the page break
                pdf.rect(x0 - 1, y0 - 1, sum(w for _n, w in cols) + 2,
                        (pdf.get_y() - y0) + 2)
                pdf.add_page()
                pdf.set_xy(pdf.l_margin, pdf.t_margin)
                x0, y0 = _header_row()
                pdf.set_font("Helvetica", "", 8.5)
            vals = [str(ln.get("sku") or "")]
            if has_req:
                vals.append(f"{int(ln.get('requested') or 0):,}")
            vals.append(_lat1(str(ln.get("description") or ""))[:70])
            vals.append(f"{int(ln.get('rec') or 0):,}")
            aligns = ["L", "R", "L", "R"] if has_req else ["L", "L", "R"]
            for (name, w), v, al in zip(cols, vals, aligns):
                pdf.cell(w, 5.5, v, align=al)
            pdf.ln()
        # box around the last (or only) page's item block
        pdf.rect(x0 - 1, y0 - 1, sum(w for _n, w in cols) + 2,
                 (pdf.get_y() - y0) + 2)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 8.5)
        pre = sum(w for _n, w in cols[:-1])
        pdf.cell(pre, 6, "Total Rec. Qty", align="R")
        pdf.cell(cols[-1][1], 6, f"{int(bb.get('rec_total') or 0):,}", align="R")

    path = _out(f"weekly_dispatch_{_ts()}.pdf", out_dir)
    pdf.output(str(path))
    return path
