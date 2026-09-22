"""Browser pages (server-rendered, session auth).

Backorder processing flow · branch sales analytics · reports · a read-only
stock-on-hand inventory summary. No ledger, adjustments, counts or ASN pages
(see README).
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from datetime import date, datetime

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from wms.analytics import loaders, allocation, demand_forecast, inventory as inv_mod
from wms.analytics import monthly_sales
from wms.analytics import weekly_forecast as weekly_fc
from wms.analytics import backorders as bo_an          # delivery-note fill analysis
from wms.enums import (
    CYCLE_LABEL, STAGE_ITEM_QTY, STAGE_LABEL, BackOrderCycle, BackOrderStage,
)
from wms.exports import excel, csv_export
from wms.exports import pdf as pdf_export
from wms.models import (
    BackOrder, Branch, DeliveryNote, DispatchOrder, Product, ReceivingOrder,
    StockOnHand, User,
)
from wms.security import ROLE_LABEL, ROLES, verify_password
from wms.services import backorders as dn_svc
from wms.services import backorder_entry as bo_entry
from wms.services import backorder_stages as bo_stage
from wms.services import dispatch as dispatch_svc
from wms.services import receiving as recv_svc
from wms.services import stock as stock_svc
from wms.services import doc_import
from wms.services import catalog
from wms.services import catalogue
from wms.services import google_oauth
from wms.web.deps import (
    Redirect, current_user, db_session, flash, pop_flashes, render, require_login,
    require_perm,
)

router = APIRouter()
_STAGE_CHOICES = [(s.value, STAGE_LABEL[s]) for s in BackOrderStage]
_CYCLE_CHOICES = [(c.value, CYCLE_LABEL[c]) for c in BackOrderCycle]
_CYCLE_LABEL = {c.value: CYCLE_LABEL[c] for c in BackOrderCycle}    # str -> label
_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _df(df):
    if df is None or df.empty:
        return []
    d = df.replace([np.inf, -np.inf], np.nan)
    return d.astype(object).where(d.notna(), None).to_dict("records")


def _bid(v) -> int | None:
    """A blank <select> submits branch_id='' - treat it (and junk) as no filter."""
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _resolve_period(weeks_raw: str, date_from: str, date_to: str) -> tuple[int, str, str]:
    """The Flow Analysis "Period" control submits a preset (an int as a
    string, or the literal "custom") plus a From/To pair that only matters in
    custom mode. -> ``(weeks, date_from, date_to)`` ready for
    weekly_forecast's ``sales_mix``/``branch_mix``/``weekly_sales_series`` -
    From/To are dropped (not just ignored) whenever the preset isn't
    "custom", so switching back to a preset can't be defeated by stale
    values left in those two fields from an earlier custom pick."""
    weeks_raw = (weeks_raw or "0").strip().lower()
    if weeks_raw == "custom":
        return 0, date_from.strip(), date_to.strip()
    try:
        return max(0, int(weeks_raw)), "", ""
    except ValueError:
        return 0, "", ""


def _stock_cov(db) -> dict:
    """Stock-on-hand coverage for the upload card: the DB balance, else any
    still-uploaded spreadsheet snapshot."""
    cov = stock_svc.coverage(db)
    return cov if cov.get("rows") else inv_mod.coverage()


# ======================================================================
# AUTH
# ======================================================================
@router.get("/login")
def login_form(request: Request, user=Depends(current_user)):
    if user:
        return RedirectResponse("/", 303)
    from wms.web.deps import templates
    return templates.TemplateResponse("login.html", {
        "request": request, "flashes": pop_flashes(request)})


@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...),
          db: Session = Depends(db_session)):
    u = db.query(User).filter(User.username == username, User.is_active.is_(True)).first()
    if not u or not verify_password(password, u.password_hash):
        flash(request, "Invalid username or password.", "error")
        return RedirectResponse("/login", 303)
    request.session["uid"] = u.id
    nxt = request.session.pop("_next", "/")
    flash(request, f"Welcome, {u.full_name}.", "success")
    return RedirectResponse(nxt, 303)


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", 303)


# ---- self-service signup ("Standard User" role: Google-verified, then
# needs a "users.admin" holder to approve it on /users before first login) --
def _google_redirect_uri(request: Request) -> str:
    return str(request.base_url).rstrip("/") + "/auth/google/callback"


def _unique_username(db: Session, base: str) -> str:
    base = re.sub(r"[^a-z0-9._-]", "", base.strip().lower())[:50] or "user"
    candidate = base
    i = 2
    while db.query(User).filter(User.username == candidate).first():
        candidate = f"{base}{i}"
        i += 1
    return candidate


@router.get("/signup")
def signup_form(request: Request, user=Depends(current_user)):
    if user:
        return RedirectResponse("/", 303)
    from wms.web.deps import templates
    return templates.TemplateResponse("signup.html", {
        "request": request, "flashes": pop_flashes(request),
        "google_configured": google_oauth.configured(),
    })


@router.get("/auth/google/start")
def google_start(request: Request):
    if not google_oauth.configured():
        flash(request, "Google sign-in isn't set up yet. Contact the administrator.", "error")
        return RedirectResponse("/signup", 303)
    state = google_oauth.new_state()
    request.session["oauth_state"] = state
    return RedirectResponse(google_oauth.auth_url(_google_redirect_uri(request), state), 303)


@router.get("/auth/google/callback")
def google_callback(request: Request, code: str = "", state: str = "", error: str = "",
                    db: Session = Depends(db_session)):
    expected_state = request.session.pop("oauth_state", None)
    if error:
        flash(request, "Google sign-in was cancelled.", "error")
        return RedirectResponse("/signup", 303)
    if not code or not state or not expected_state or state != expected_state:
        flash(request, "Google sign-in failed (the request expired or was tampered with). Try again.", "error")
        return RedirectResponse("/signup", 303)

    try:
        claims = google_oauth.exchange_code(code, _google_redirect_uri(request))
    except google_oauth.GoogleAuthError as e:
        flash(request, str(e), "error")
        return RedirectResponse("/signup", 303)

    u = db.query(User).filter(User.google_sub == claims["sub"]).first()
    if u is None:
        u = User(username=_unique_username(db, claims["email"].split("@")[0]),
                 full_name=claims["name"], email=claims["email"],
                 google_sub=claims["sub"], role="user", password_hash=None,
                 is_active=True, is_approved=False)
        db.add(u)
        db.commit()
        flash(request, f"Thanks, {u.full_name}! Your sign-up has been sent to the "
                       f"administrator for approval - you'll be able to sign in once "
                       f"it's approved.", "success")
        return RedirectResponse("/login", 303)

    if not u.is_active:
        flash(request, "Your account has been disabled. Contact the administrator.", "error")
        return RedirectResponse("/login", 303)
    if not u.is_approved:
        flash(request, "Your account is still awaiting administrator approval.", "error")
        return RedirectResponse("/login", 303)

    request.session["uid"] = u.id
    nxt = request.session.pop("_next", "/")
    flash(request, f"Welcome, {u.full_name}.", "success")
    return RedirectResponse(nxt, 303)


# ---- Users (admin-only: approve Google sign-ups, manage roles) -------------
@router.get("/users")
def users_page(request: Request, db: Session = Depends(db_session),
              user: User = Depends(require_perm("users.admin"))):
    pending = (db.query(User).filter(User.is_approved.is_(False))
              .order_by(User.created_at).all())
    everyone = db.query(User).order_by(User.role, User.username).all()
    return render(request, "users.html", user, pending=pending, users=everyone,
                  roles=ROLES)


@router.post("/users/{uid}/approve")
def users_approve(request: Request, uid: int, role: str = Form(""),
                  db: Session = Depends(db_session),
                  user: User = Depends(require_perm("users.admin"))):
    target = db.query(User).filter(User.id == uid).first()
    if not target:
        flash(request, "User not found.", "error")
        return RedirectResponse("/users", 303)
    if role.strip() in ROLES:
        target.role = role.strip()
    target.is_approved = True
    db.commit()
    flash(request, f"Approved {target.full_name} as {ROLE_LABEL.get(target.role, target.role)}.",
         "success")
    return RedirectResponse("/users", 303)


@router.post("/users/{uid}/reject")
def users_reject(request: Request, uid: int, db: Session = Depends(db_session),
                 user: User = Depends(require_perm("users.admin"))):
    target = db.query(User).filter(User.id == uid).first()
    if not target:
        flash(request, "User not found.", "error")
    elif target.is_approved:
        flash(request, "Only a pending sign-up can be rejected - disable an "
                       "approved user instead.", "error")
    else:
        db.delete(target)
        db.commit()
        flash(request, f"Rejected and removed {target.full_name}.", "success")
    return RedirectResponse("/users", 303)


@router.post("/users/{uid}/role")
def users_set_role(request: Request, uid: int, role: str = Form(...),
                   db: Session = Depends(db_session),
                   user: User = Depends(require_perm("users.admin"))):
    if uid == user.id:
        flash(request, "You can't change your own role here.", "error")
        return RedirectResponse("/users", 303)
    target = db.query(User).filter(User.id == uid).first()
    if not target:
        flash(request, "User not found.", "error")
    elif role.strip() not in ROLES:
        flash(request, "Not a valid role.", "error")
    else:
        target.role = role.strip()
        db.commit()
        flash(request, f"{target.full_name} is now {ROLE_LABEL.get(target.role, target.role)}.",
             "success")
    return RedirectResponse("/users", 303)


@router.post("/users/{uid}/toggle-active")
def users_toggle_active(request: Request, uid: int, db: Session = Depends(db_session),
                        user: User = Depends(require_perm("users.admin"))):
    if uid == user.id:
        flash(request, "You can't disable your own account.", "error")
        return RedirectResponse("/users", 303)
    target = db.query(User).filter(User.id == uid).first()
    if not target:
        flash(request, "User not found.", "error")
    else:
        target.is_active = not target.is_active
        db.commit()
        flash(request, f"{target.full_name} is now "
                       f"{'active' if target.is_active else 'disabled'}.", "success")
    return RedirectResponse("/users", 303)


@router.get("/")
def home(user: User = Depends(require_login)):
    # Active Back Orders is temporarily off the nav - land on Flow Analysis instead
    return RedirectResponse("/analysis", 303)


# ======================================================================
# BACKORDER PROCESSING FLOW
# ======================================================================
@router.get("/backorders")
def backorders(request: Request, stage: str = "", branch_id: str = "",
               q: str = "", db: Session = Depends(db_session),
               user: User = Depends(require_perm("nav.full"))):
    """Active Back Orders grid - list + Stage filter + Search."""
    branches = db.query(Branch).order_by(Branch.name).all()
    bid = _bid(branch_id)
    rows = bo_entry.list_back_orders(db, branch_id=bid, stage=stage or None,
                                     open_only=not stage, q=q or None)
    return render(request, "backorders.html", user, branches=branches,
                  stage=stage, branch_id=bid, q=q, stage_choices=_STAGE_CHOICES,
                  stage_label=STAGE_LABEL, cycle_label=_CYCLE_LABEL, BOStage=BackOrderStage,
                  orders=[bo_entry.serialize(b) for b in rows])


@router.get("/analysis")
def backorders_analysis(request: Request, branch_id: str = "", sku: str = "",
                        fa_metric: str = "sales",
                        bmix_sku: str = "", bmix_b1: str = "", bmix_b2: str = "",
                        bmix_b3: str = "", bmix_weeks: str = "0",
                        bmix_from: str = "", bmix_to: str = "",
                        pie_branch: str = "", pie_p1: str = "", pie_p2: str = "",
                        pie_p3: str = "", pie_top: int = 8, pie_weeks: str = "0",
                        pie_from: str = "", pie_to: str = "",
                        worst_branch: str = "",
                        growth_bcode: str = "BM",
                        db: Session = Depends(db_session),
                        user: User = Depends(require_login)):
    branches = db.query(Branch).order_by(Branch.name).all()
    bid = _bid(branch_id)
    bcode = ""
    if bid:
        b = db.query(Branch).filter(Branch.id == bid).first()
        bcode = b.code if b else ""
    pie_top = max(8, min(int(pie_top or 8), 500))     # "Other" expands in steps, capped
    pie_picks = [pie_p1.strip(), pie_p2.strip(), pie_p3.strip()]
    pie_w, pie_df, pie_dt = _resolve_period(pie_weeks, pie_from, pie_to)
    pie_kw = dict(bcode=pie_branch.strip(), skus=pie_picks, top=pie_top,
                  weeks=pie_w, date_from=pie_df, date_to=pie_dt)
    bmix_bs = [bmix_b1.strip(), bmix_b2.strip(), bmix_b3.strip()]
    bmix_w, bmix_df, bmix_dt = _resolve_period(bmix_weeks, bmix_from, bmix_to)
    bmix_kw = dict(sku=bmix_sku.strip(), bcodes=bmix_bs,
                   weeks=bmix_w, date_from=bmix_df, date_to=bmix_dt)
    worst_branch = worst_branch.strip().upper()
    worst_bcodes = [worst_branch] if worst_branch else []

    # Flow Analysis runs on monthly data by default - real monthly Hansa
    # exports cover every branch, while real weekly uploads are still only a
    # few months deep for a handful of branches (see
    # monthly_sales.cached_matrix_panel docstring). Both panels share the
    # weekly-forecast helpers' shape, so only the panel/period_fmt passed in
    # changes.
    mp = monthly_sales.cached_matrix_panel()
    mfmt = monthly_sales.short_month
    period_options = [{"value": w, "label": mfmt(w)} for w in mp["weeks"]]

    # Power-BI-style KPI strip
    _fs = weekly_fc.flow_summary(panel=mp, period_fmt=mfmt)
    growth_bcode = growth_bcode.strip().upper()
    _growth = weekly_fc.growth_overview(bcode=growth_bcode, panel=mp, period_fmt=mfmt)
    kpis = []
    if _fs.get("has_data"):
        kpis += [
            {"label": "Units sold, last month", "value": f"{_fs['week_units']:,}",
             "sub": _fs['week_label'], "delta": _fs["wow_pct"],
             "good": "up"},
            {"label": "Revenue, last month",
             "value": f"{_fs['week_revenue']:,}" if _fs["week_revenue"] is not None else "—",
             "sub": _fs['week_label'], "delta": _fs["revenue_wow_pct"]},
            {"label": "Gross margin",
             "value": f"{_fs['margin_pct']}%" if _fs["margin_pct"] is not None else "—",
             "sub": "profit / revenue, all-time"},
            {"label": "Top branch", "value": _fs["top_branch"] or "—",
             "sub": f"{_fs['top_branch_pct']:.0f}% of units"},
            {"label": "Best-selling product", "value": _fs["top_prod"] or "—",
             "sub": f"{_fs['top_prod_units']:,} units · {_fs['top_prod_pct']:.0f}%"},
            {"label": "Top product by profit", "value": _fs["top_profit_prod"] or "—",
             "sub": f"{_fs['top_profit_val']:,} profit · {_fs['top_profit_pct']:.0f}%"},
        ]
    if _growth.get("has_data"):
        kpis += [
            {"label": "Active branches", "value": str(_growth["active_branches"]),
             "sub": "sold something in the latest month"},
            {"label": "Active products", "value": str(_growth["active_products"]),
             "sub": "distinct SKUs sold in the latest month"},
            {"label": f"Units sold ({_growth['n_weeks']} mo)",
             "value": f"{_growth['units_period']:,}", "sub": "network-wide"},
            {"label": "Growth",
             "value": f"{_growth['overall_pct_growth']}%"
                      if _growth["overall_pct_growth"] is not None else "—",
             "sub": "second half vs first half of the window"},
        ]

    return render(request, "backorder_analysis.html", user, branches=branches,
                  kpis=kpis, growth=_growth, growth_bcode=growth_bcode,
                  branch_id=bid, sku=sku.strip(), period_options=period_options,
                  pie_branch=pie_branch.strip(), pie_p1=pie_p1.strip(),
                  pie_p2=pie_p2.strip(), pie_p3=pie_p3.strip(), pie_top=pie_top,
                  pie_weeks=pie_weeks.strip().lower(), pie_from=pie_from.strip(), pie_to=pie_to.strip(),
                  bmix_weeks=bmix_weeks.strip().lower(), bmix_from=bmix_from.strip(), bmix_to=bmix_to.strip(),
                  bmix_sku=bmix_sku.strip(), bmix_b1=bmix_b1.strip(),
                  bmix_b2=bmix_b2.strip(), bmix_b3=bmix_b3.strip(),
                  forced_model=weekly_fc.forced_model(),
                  ckpt=weekly_fc.checkpoint_status(),
                  fa_metric=(fa_metric or "sales").strip().lower(),
                  sales_series=weekly_fc.weekly_sales_series(
                      bcode=bcode, sku=sku, metric=fa_metric, panel=mp, period_fmt=mfmt),
                  bmix_units=weekly_fc.branch_mix(metric="units", panel=mp, period_fmt=mfmt, **bmix_kw),
                  bmix_profit=weekly_fc.branch_mix(metric="profit", panel=mp, period_fmt=mfmt, **bmix_kw),
                  mix_units=weekly_fc.sales_mix(metric="units", panel=mp, period_fmt=mfmt, **pie_kw),
                  mix_profit=weekly_fc.sales_mix(metric="profit", panel=mp, period_fmt=mfmt, **pie_kw),
                  sku_options=weekly_fc.sales_products(panel=mp),
                  worst=weekly_fc.worst_performers(db, bcodes=worst_bcodes, panel=mp),
                  worst_branch=worst_branch,
                  model_scores=weekly_fc.model_scores(),
                  model_options=weekly_fc.model_options())


@router.post("/analysis/model")
def backorders_analysis_model(request: Request, model: str = Form(""),
                              user: User = Depends(require_perm("backorder.enter"))):
    """Pin which weekly model every forecast uses (or '' = auto-select)."""
    weekly_fc.set_forced_model(model)
    flash(request, f"Weekly forecast model set to "
                   f"{model or 'auto (bias-aware pick)'}. Recomputing…", "success")
    return RedirectResponse("/analysis", 303)


@router.post("/analysis/retrain")
def backorders_analysis_retrain(request: Request, quick: str = Form(""),
                                user: User = Depends(require_perm("backorder.enter"))):
    """Kick off an offline, thorough retrain of the esrnn_ratio network in a
    detached subprocess (does not block the request). The app picks up the new
    weights on its next forecast once the file is written."""
    root = pathlib.Path(__file__).resolve().parents[2]
    args = [sys.executable, "-m", "wms.scripts.train_weekly"]
    if quick:
        args.append("--quick")
    try:
        subprocess.Popen(args, cwd=str(root),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        flash(request, "Retraining the forecast network in the background "
                       f"({'quick' if quick else 'full'} pass, "
                       f"~{'2' if quick else '10'}-15 min). The new model loads "
                       "automatically when it finishes.", "success")
    except Exception as e:                                # noqa: BLE001
        flash(request, f"Could not start retraining: {e}", "error")
    return RedirectResponse("/analysis", 303)


def _recent_dispatches(db, limit: int = 40) -> list[dict]:
    rows = (db.query(DeliveryNote).order_by(DeliveryNote.id.desc()).limit(limit).all())
    bo_by_ref: dict[str, str] = {}
    for bo in db.query(BackOrder).filter(BackOrder.source == "DISPATCH").all():
        for ref in (bo.source_ref or "").split(","):
            if ref:
                bo_by_ref[ref] = bo
    out = []
    for dn in rows:
        bo = bo_by_ref.get(dn.dn_no)
        out.append({
            "dn_no": dn.dn_no, "branch": dn.branch.name if dn.branch else "",
            "doc_date": dn.doc_date, "lines": len(dn.lines),
            "requested": dn.total_requested, "sent": dn.total_sent,
            "short": dn.total_requested - dn.total_sent,
            "bo_no": bo.bo_no if bo else None,
            "bo_open": bool(bo) and bo.status == "OPEN",
            "reversible": (bo is None) or bo.status == "OPEN",
        })
    return out


def _bo_new_ctx(db, prefill=None):
    return dict(branches=db.query(Branch).order_by(Branch.name).all(),
                products=db.query(Product).order_by(Product.sku).limit(8000).all(),
                cycle_choices=_CYCLE_CHOICES, dispatches=_recent_dispatches(db),
                today=date.today().isoformat(), prefill=prefill)


# legacy path -> new /dispatch URL (307 keeps the method + body for POSTs)
@router.api_route("/backorders/new", methods=["GET"])
def _legacy_dispatch_new(request: Request):
    return RedirectResponse("/dispatch/new", 307)


@router.api_route("/backorders/new/upload", methods=["POST"])
def _legacy_dispatch_upload(request: Request):
    return RedirectResponse("/dispatch/new/upload", 307)


@router.get("/dispatch/new")
def bo_new(request: Request, db: Session = Depends(db_session),
           user: User = Depends(require_perm("backorder.enter"))):
    return render(request, "backorder_new.html", user, **_bo_new_ctx(db))


@router.post("/dispatch/new/upload")
def bo_upload(request: Request, files: list[UploadFile] = File(...),
              db: Session = Depends(db_session),
              user: User = Depends(require_perm("backorder.enter"))):
    """Parse one or more uploaded stock-movement document pages/photos (as one
    document) and pre-fill the form."""
    try:
        parsed = doc_import.parse_documents(
            [(f.file.read(), f.filename or "") for f in files])
    except Exception as e:
        flash(request, f"Could not read the document: {e}", "error")
        return render(request, "backorder_new.html", user, **_bo_new_ctx(db))

    # resolve branch text -> id; fill descriptions from the catalogue
    prods = {p.sku: p for p in db.query(Product).all()}
    branch_id = None
    guess = (parsed.get("branch") or "").strip().lower()
    if guess:
        for b in db.query(Branch).all():
            if guess in (b.code.lower(), b.name.lower()) or guess in b.name.lower():
                branch_id = b.id
                break
    for ln in parsed["lines"]:
        if ln.get("sku_derived"):
            match = catalogue.match_product_by_description(
                ln.get("description") or "", list(prods.values()))
            if match:
                ln["sku"] = match.sku
        p = prods.get(ln["sku"])
        if p and not ln.get("description"):
            ln["description"] = p.name
    d = parsed.get("doc_date")
    prefill = {
        "stock_movement_id": parsed.get("stock_movement_id"),
        "branch_id": branch_id,
        "doc_date": d.isoformat() if d else None,
        "lines": parsed["lines"],
        "warnings": ([] if branch_id else
                     ([f"Branch '{parsed['branch']}' not matched - pick it below."]
                      if parsed.get("branch") else ["Branch not found in the document - pick it below."]))
                    + parsed.get("warnings", []),
    }
    label = files[0].filename if len(files) == 1 else f"{len(files)} files"
    flash(request, f"Parsed {len(parsed['lines'])} line(s) from {label}.", "success")
    return render(request, "backorder_new.html", user, **_bo_new_ctx(db, prefill))


@router.api_route("/backorders/new", methods=["POST"])
def _legacy_dispatch_create(request: Request):
    return RedirectResponse("/dispatch/new", 307)


@router.post("/dispatch/new")
def bo_create(request: Request, branch_id: int = Form(...),
              stock_movement_id: str = Form(""), doc_date: str = Form(""),
              priority: str = Form("NORMAL"), cycle: str = Form("WEEKLY"),
              notes: str = Form(""),
              sku: list[str] = Form(default=[]),
              description: list[str] = Form(default=[]),
              requested_qty: list[str] = Form(default=[]),
              sent_qty: list[str] = Form(default=[]),
              db: Session = Depends(db_session),
              user: User = Depends(require_perm("backorder.enter"))):
    descs = description + [""] * (len(sku) - len(description))
    lines = [{"sku": s.strip(), "description": (d or "").strip(),
              "requested_qty": int(r), "sent_qty": int(x or 0)}
             for s, d, r, x in zip(sku, descs, requested_qty, sent_qty) if s.strip() and r]
    try:
        dd = datetime.strptime(doc_date, "%Y-%m-%d").date() if doc_date else date.today()
        # 1. record the dispatch note (sent -> branch sales); no per-note back order
        dn = dn_svc.enter_delivery_note(
            db, branch_id=branch_id, doc_no=(stock_movement_id.strip() or None),
            doc_date=dd, comment=(notes or None), lines=lines,
            raise_backorder=False, commit=False, user_id=user.id)
        # 2. dispatched units land in the branch's stock-on-hand balance
        moved = stock_svc.add_stock(
            db, branch_id=branch_id,
            items=[{"sku": ln["sku"], "qty": ln["sent_qty"]} for ln in lines
                   if ln["sent_qty"] > 0],
            user_id=user.id, commit=False)
        db.commit()
        short = sum(max(0, ln["requested_qty"] - ln["sent_qty"]) for ln in lines)
        branch = db.query(Branch).filter(Branch.id == branch_id).first()
        msg = f"{dn.dn_no}: {moved} unit(s) added to {branch.name if branch else 'branch'} stock."
        if short:
            msg += f" {short} unit(s) short of what was requested."
        flash(request, msg, "success")
        return RedirectResponse(f"/delivery-notes/{dn.dn_no}", 303)
    except Exception as e:
        db.rollback()
        flash(request, str(e), "error")
        prefill = {
            "stock_movement_id": stock_movement_id, "branch_id": branch_id,
            "doc_date": doc_date or None, "priority": priority, "cycle": cycle,
            "notes": notes, "warnings": [],
            "lines": [{"sku": s, "description": "", "requested_qty": r, "sent_qty": x}
                      for s, r, x in zip(sku, requested_qty, sent_qty) if s.strip()],
        }
        return render(request, "backorder_new.html", user, **_bo_new_ctx(db, prefill))


@router.post("/dispatch/{dn_no}/delete")
def dispatch_delete(dn_no: str, request: Request, db: Session = Depends(db_session),
                    user: User = Depends(require_perm("backorder.enter"))):
    """Reverse every effect of a dispatch: stock, recorded sales, and its
    contribution to the branch's open back order; then delete the note."""
    try:
        res = bo_entry.reverse_dispatch(db, dn_no, user_id=user.id)
        flash(request, f"Reversed dispatch {dn_no}: pulled {res['units_pulled']} unit(s) "
                       f"from stock, removed {res['sales_deleted']} sales row(s)"
                       + (f", {res['back_order']}" if res.get("back_order") else "")
                       + ".", "success")
    except Exception as e:
        db.rollback()
        flash(request, str(e), "error")
    return RedirectResponse("/dispatch/new", 303)


# ======================================================================
# RECEIVING ORDERS  -  stock arriving into the warehouse (mirrors Recon:
# upload a document, review/edit the parsed lines, confirm - except a
# receipt only ever adds to stock, no back order / requested-vs-sent split)
# ======================================================================
def _recent_receiving(db, limit: int = 40) -> list[dict]:
    rows = (db.query(ReceivingOrder).order_by(ReceivingOrder.id.desc()).limit(limit).all())
    return [{
        "ro_no": ro.ro_no, "branch": ro.branch.name if ro.branch else "",
        "supplier": ro.supplier, "doc_date": ro.doc_date, "lines": len(ro.lines),
        "received": ro.total_received,
    } for ro in rows]


def _receiving_ctx(db, prefill=None):
    dc = db.query(Branch).filter(Branch.code == "DC").first()
    return dict(branches=db.query(Branch).order_by(Branch.name).all(),
                products=db.query(Product).order_by(Product.sku).limit(8000).all(),
                receipts=_recent_receiving(db), today=date.today().isoformat(),
                default_branch_id=dc.id if dc else None, prefill=prefill)


@router.get("/receiving/new")
def receiving_new(request: Request, db: Session = Depends(db_session),
                  user: User = Depends(require_perm("receiving.enter"))):
    return render(request, "receiving_new.html", user, **_receiving_ctx(db))


@router.post("/receiving/new/upload")
def receiving_upload(request: Request, files: list[UploadFile] = File(...),
                     db: Session = Depends(db_session),
                     user: User = Depends(require_perm("receiving.enter"))):
    """Parse one or more uploaded receiving document pages/photos (as one
    document) and pre-fill the form."""
    try:
        parsed = doc_import.parse_documents(
            [(f.file.read(), f.filename or "") for f in files])
    except Exception as e:
        flash(request, f"Could not read the document: {e}", "error")
        return render(request, "receiving_new.html", user, **_receiving_ctx(db))

    prods = {p.sku: p for p in db.query(Product).all()}
    branch_id = None
    guess = (parsed.get("branch") or "").strip().lower()
    if guess:
        for b in db.query(Branch).all():
            if guess in (b.code.lower(), b.name.lower()) or guess in b.name.lower():
                branch_id = b.id
                break
    lines = []
    for ln in parsed["lines"]:
        sku = ln["sku"]
        if ln.get("sku_derived"):
            match = catalogue.match_product_by_description(
                ln.get("description") or "", list(prods.values()))
            if match:
                sku = match.sku
        p = prods.get(sku)
        lines.append({
            "sku": sku,
            "description": ln.get("description") or (p.name if p else ""),
            # whichever qty column the document actually had for what arrived
            "received_qty": ln.get("sent_qty") or ln.get("requested_qty") or 0,
        })
    d = parsed.get("doc_date")
    prefill = {
        "ro_no": parsed.get("stock_movement_id"),
        "branch_id": branch_id,
        "doc_date": d.isoformat() if d else None,
        "lines": lines,
        "warnings": ([] if branch_id else
                     (["Location not matched - pick it below."]))
                    + parsed.get("warnings", []),
    }
    label = files[0].filename if len(files) == 1 else f"{len(files)} files"
    flash(request, f"Parsed {len(lines)} line(s) from {label}.", "success")
    return render(request, "receiving_new.html", user, **_receiving_ctx(db, prefill))


@router.post("/receiving/new")
def receiving_create(request: Request, branch_id: int = Form(...),
                     ro_no: str = Form(""), doc_date: str = Form(""),
                     supplier: str = Form(""), notes: str = Form(""),
                     sku: list[str] = Form(default=[]),
                     description: list[str] = Form(default=[]),
                     received_qty: list[str] = Form(default=[]),
                     db: Session = Depends(db_session),
                     user: User = Depends(require_perm("receiving.enter"))):
    descs = description + [""] * (len(sku) - len(description))
    lines = [{"sku": s.strip(), "description": (d or "").strip(),
              "received_qty": int(q)}
             for s, d, q in zip(sku, descs, received_qty) if s.strip() and q]
    try:
        dd = datetime.strptime(doc_date, "%Y-%m-%d").date() if doc_date else date.today()
        ro = recv_svc.enter_receiving_order(
            db, branch_id=branch_id, doc_no=(ro_no.strip() or None), doc_date=dd,
            supplier=(supplier.strip() or None), comment=(notes or None),
            lines=lines, user_id=user.id)
        flash(request, f"{ro.ro_no}: {ro.total_received} unit(s) added to "
                       f"{ro.branch.name} stock.", "success")
        return RedirectResponse("/receiving/new", 303)
    except Exception as e:
        db.rollback()
        flash(request, str(e), "error")
        prefill = {
            "ro_no": ro_no, "branch_id": branch_id, "doc_date": doc_date or None,
            "supplier": supplier, "notes": notes, "warnings": [],
            "lines": [{"sku": s, "description": "", "received_qty": q}
                      for s, q in zip(sku, received_qty) if s.strip()],
        }
        return render(request, "receiving_new.html", user, **_receiving_ctx(db, prefill))


@router.post("/receiving/{ro_no}/delete")
def receiving_delete(ro_no: str, request: Request, db: Session = Depends(db_session),
                     user: User = Depends(require_perm("receiving.enter"))):
    """Reverse a receiving order: pull its units back out of stock, then delete it."""
    try:
        res = recv_svc.reverse_receiving_order(db, ro_no, user_id=user.id)
        flash(request, f"Reversed receiving order {ro_no}: pulled "
                       f"{res['units_pulled']} unit(s) from {res['branch']} stock.",
              "success")
    except Exception as e:
        db.rollback()
        flash(request, str(e), "error")
    return RedirectResponse("/receiving/new", 303)


# ======================================================================
# RECON  -  stock leaving the warehouse for a branch. Upload a dispatch
# document, review/edit the parsed lines, confirm - a dispatch only ever
# moves stock (warehouse -> branch), no requested-vs-sent split and no
# back order. (The older DeliveryNote/BackOrder flow is untouched and
# still reachable directly; this is a separate, simpler record.)
# ======================================================================
def _recent_dispatches_recon(db, limit: int = 40) -> list[dict]:
    rows = (db.query(DispatchOrder).order_by(DispatchOrder.id.desc()).limit(limit).all())
    return [{
        "do_no": do.do_no, "branch": do.branch.name if do.branch else "",
        "doc_date": do.doc_date, "lines": len(do.lines),
        "dispatched": do.total_dispatched,
    } for do in rows]


def _recon_ctx(db, prefill=None):
    return dict(branches=[b for b in db.query(Branch).order_by(Branch.name).all()
                          if b.code != "DC"],
                products=db.query(Product).order_by(Product.sku).limit(8000).all(),
                dispatches=_recent_dispatches_recon(db), today=date.today().isoformat(),
                prefill=prefill)


@router.get("/recon/new")
def recon_new(request: Request, db: Session = Depends(db_session),
             user: User = Depends(require_perm("receiving.enter"))):
    return render(request, "recon_new.html", user, **_recon_ctx(db))


@router.post("/recon/new/upload")
def recon_upload(request: Request, files: list[UploadFile] = File(...),
                 db: Session = Depends(db_session),
                 user: User = Depends(require_perm("receiving.enter"))):
    """Parse one or more uploaded dispatch document pages/photos (as one
    document) and pre-fill the form."""
    try:
        parsed = doc_import.parse_documents(
            [(f.file.read(), f.filename or "") for f in files])
    except Exception as e:
        flash(request, f"Could not read the document: {e}", "error")
        return render(request, "recon_new.html", user, **_recon_ctx(db))

    prods = {p.sku: p for p in db.query(Product).all()}
    branch_id = None
    guess = (parsed.get("branch") or "").strip().lower()
    if guess:
        for b in db.query(Branch).all():
            if b.code != "DC" and (guess in (b.code.lower(), b.name.lower())
                                   or guess in b.name.lower()):
                branch_id = b.id
                break
    lines = []
    for ln in parsed["lines"]:
        sku = ln["sku"]
        if ln.get("sku_derived"):
            match = catalogue.match_product_by_description(
                ln.get("description") or "", list(prods.values()))
            if match:
                sku = match.sku
        p = prods.get(sku)
        lines.append({
            "sku": sku,
            "description": ln.get("description") or (p.name if p else ""),
            "dispatched_qty": ln.get("sent_qty") or ln.get("requested_qty") or 0,
        })
    d = parsed.get("doc_date")
    prefill = {
        "do_no": parsed.get("stock_movement_id"),
        "branch_id": branch_id,
        "doc_date": d.isoformat() if d else None,
        "lines": lines,
        "warnings": ([] if branch_id else
                     (["Branch not matched - pick it below."]))
                    + parsed.get("warnings", []),
    }
    label = files[0].filename if len(files) == 1 else f"{len(files)} files"
    flash(request, f"Parsed {len(lines)} line(s) from {label}.", "success")
    return render(request, "recon_new.html", user, **_recon_ctx(db, prefill))


@router.post("/recon/new")
def recon_create(request: Request, branch_id: int = Form(...),
                 do_no: str = Form(""), doc_date: str = Form(""),
                 notes: str = Form(""),
                 sku: list[str] = Form(default=[]),
                 description: list[str] = Form(default=[]),
                 dispatched_qty: list[str] = Form(default=[]),
                 db: Session = Depends(db_session),
                 user: User = Depends(require_perm("receiving.enter"))):
    descs = description + [""] * (len(sku) - len(description))
    lines = [{"sku": s.strip(), "description": (d or "").strip(),
              "dispatched_qty": int(q)}
             for s, d, q in zip(sku, descs, dispatched_qty) if s.strip() and q]
    try:
        dd = datetime.strptime(doc_date, "%Y-%m-%d").date() if doc_date else date.today()
        do, warnings = dispatch_svc.enter_dispatch_order(
            db, branch_id=branch_id, doc_no=(do_no.strip() or None), doc_date=dd,
            comment=(notes or None), lines=lines, user_id=user.id)
        for w in warnings:
            flash(request, w, "info")
        flash(request, f"{do.do_no}: {do.total_dispatched} unit(s) dispatched to "
                       f"{do.branch.name}.", "success")
        return RedirectResponse("/recon/new", 303)
    except Exception as e:
        db.rollback()
        flash(request, str(e), "error")
        prefill = {
            "do_no": do_no, "branch_id": branch_id, "doc_date": doc_date or None,
            "notes": notes, "warnings": [],
            "lines": [{"sku": s, "description": "", "dispatched_qty": q}
                      for s, q in zip(sku, dispatched_qty) if s.strip()],
        }
        return render(request, "recon_new.html", user, **_recon_ctx(db, prefill))


@router.post("/recon/{do_no}/delete")
def recon_delete(do_no: str, request: Request, db: Session = Depends(db_session),
                 user: User = Depends(require_perm("receiving.enter"))):
    """Reverse a dispatch order: return its units to the warehouse, then delete it."""
    try:
        res = dispatch_svc.reverse_dispatch_order(db, do_no, user_id=user.id)
        flash(request, f"Reversed dispatch order {do_no}: returned "
                       f"{res['units_pulled']} unit(s) from {res['branch']} to the "
                       f"warehouse.", "success")
    except Exception as e:
        db.rollback()
        flash(request, str(e), "error")
    return RedirectResponse("/recon/new", 303)


@router.get("/backorders/{bo_no}")
def bo_detail(bo_no: str, request: Request, db: Session = Depends(db_session),
              user: User = Depends(require_perm("nav.full"))):
    try:
        bo = bo_entry.get(db, bo_no)
    except Exception:
        raise Redirect("/backorders")
    allowed = [(s, STAGE_LABEL[BackOrderStage(s)]) for s in bo_stage.allowed_next(bo.stage)]
    # requested-vs-sent from the originating stock-movement document (the whole point
    # of a back order is the gap between what a branch asked for and what was shipped)
    req_sent: dict[int, dict] = {}
    refs = [r for r in (bo.source_ref or "").split(",") if r]
    if refs:
        for dn in db.query(DeliveryNote).filter(DeliveryNote.dn_no.in_(refs)).all():
            for l in dn.lines:
                agg = req_sent.setdefault(l.product_id, {"requested": 0, "sent": 0})
                agg["requested"] += l.requested_qty
                agg["sent"] += l.sent_qty
    return render(request, "backorder_detail.html", user, doc=bo_entry.serialize(bo),
                  allowed=allowed, stage_label=STAGE_LABEL, cycle_label=_CYCLE_LABEL,
                  req_sent=req_sent, BOStage=BackOrderStage, qty_fields=STAGE_ITEM_QTY)


@router.post("/backorders/{bo_no}/advance")
def bo_advance(bo_no: str, request: Request, to_stage: str = Form(...),
               requisition_no: str = Form(""), po_no: str = Form(""),
               supplier: str = Form(""), note: str = Form(""),
               item_id: list[int] = Form(default=[]), item_qty: list[str] = Form(default=[]),
               db: Session = Depends(db_session),
               user: User = Depends(require_perm("backorder.manage"))):
    items = {int(i): int(x) for i, x in zip(item_id, item_qty) if x != ""}
    try:
        bo = bo_stage.advance(db, bo_no=bo_no, to_stage=to_stage, items=items or None,
                              requisition_no=requisition_no or None, po_no=po_no or None,
                              supplier=supplier or None, note=note or None, user_id=user.id)
        flash(request, f"{bo.bo_no} -> {STAGE_LABEL[BackOrderStage(bo.stage)]}.", "success")
    except Exception as e:
        flash(request, str(e), "error")
    return RedirectResponse(f"/backorders/{bo_no}", 303)


# --- delivery note (the requested-vs-sent source document) detail ---
@router.get("/delivery-notes/{dn_no}")
def dn_detail(dn_no: str, request: Request, db: Session = Depends(db_session),
              user: User = Depends(require_perm("nav.full"))):
    dn = db.query(DeliveryNote).filter(DeliveryNote.dn_no == dn_no).first()
    if not dn:
        raise Redirect("/backorders")
    return render(request, "dn_detail.html", user, doc=dn_svc.serialize(db, dn))


# ======================================================================
# ANALYTICS
# ======================================================================
def _alloc_ctx(db, *, bcode: str = "", q: str = "", alloc_sku: str = "",
               alloc_qty: str = "", alloc_br=None) -> dict:
    """Everything the Sales & Forecasting -> Allocation plan tab renders."""
    alloc_br = [b.strip().upper() for b in (alloc_br or []) if b.strip()]
    st = weekly_fc.cached_run().get("state")
    fc_branches = ([] if st is None or st.empty
                   else sorted({(r.branch, r.branch_name) for r in st.itertuples()},
                               key=lambda t: t[1]))
    result = None
    if alloc_sku.strip():
        result = allocation.allocate_by_forecast(
            db, sku=alloc_sku.strip(), qty=_bid(alloc_qty) or 0,
            branch_codes=alloc_br or None)

    # the pick list must cover every product that can actually be split - that is
    # the weekly-forecast universe (thousands of SKUs, most not in the Product
    # table), plus any Product rows for good measure
    seen, split_skus = set(), []
    if st is not None and not st.empty:
        for sk, nm in (st[["sku", "item"]].drop_duplicates()
                       .itertuples(index=False, name=None)):
            k = str(sk).strip()
            if k and k.upper() not in seen:
                seen.add(k.upper())
                split_skus.append((k, str(nm or "").strip()))
    for p in db.query(Product).order_by(Product.sku).limit(8000):
        if p.sku and p.sku.upper() not in seen:
            seen.add(p.sku.upper())
            split_skus.append((p.sku, p.name or ""))
    split_skus.sort(key=lambda t: (t[1] or t[0]).lower())

    abc = weekly_fc.abc_classification()
    alloc_class = (abc.get("class_by_sku", {}).get(result["product"].upper())
                   if result else None)

    return dict(
        tab="allocation",
        branches=db.query(Branch).order_by(Branch.name).all(),
        fc_branches=fc_branches, fc_codes=[c for c, _n in fc_branches],
        split_skus=split_skus,
        bcode=bcode, q=q,
        alloc_sku=alloc_sku, alloc_qty=alloc_qty, alloc_branches=alloc_br,
        alloc=result, alloc_rows=(result["allocations"] if result else []),
        abc=abc, alloc_class=alloc_class, receipts=_recent_receiving(db))


@router.get("/analytics")
def analytics(request: Request, tab: str = "", branch_id: str = "",
              bcode: str = "", q: str = "", alloc_sku: str = "", alloc_qty: str = "",
              alloc_branches: list[str] = Query(default_factory=list),
              db: Session = Depends(db_session), user: User = Depends(require_login)):
    branches = db.query(Branch).order_by(Branch.name).all()

    if tab != "demand":
        # default view: the allocation plan (bare /analytics, tab=allocation,
        # or any other/unrecognised tab value all land here)
        return render(request, "analytics.html", user,
                      **_alloc_ctx(db, bcode=bcode, q=q, alloc_sku=alloc_sku,
                                   alloc_qty=alloc_qty, alloc_br=alloc_branches))

    # explicit tab=demand: the weekly per-SKU demand forecast
    disp = weekly_fc.display_frame(bcode=bcode, q=q)
    fc_total = len(disp)
    if bcode:
        preview = disp.head(400)
    else:
        # capped PER BRANCH (not just the first 400 rows overall) - branch_name
        # is the primary sort key, so a single large branch (e.g. Belmont Shop's
        # 2000+ lines) would otherwise fill the entire preview by itself and
        # every other branch would silently vanish from "All branches"
        preview = disp.groupby("Location", group_keys=False).head(40)
    forecast_rows = _df(preview)
    fc_cols = list(disp.columns)
    return render(request, "analytics.html", user, tab="demand", branches=branches,
                  bcode=bcode, q=q, inv_cov=_stock_cov(db),
                  has_weekly=weekly_fc.has_data(),
                  fc_cols=fc_cols, fc_total=fc_total, forecast_rows=forecast_rows)


def _run_split_batch(db, pairs, brs: list[str]) -> dict:
    """Split each (sku, qty) pair across branches by predicted weekly sales.
    Shared by the 'upload a list' and the manual line-item forms."""
    import urllib.parse as _url
    rows = []
    eq = ""
    for sk, qv in pairs:
        sk = str(sk or "").strip()
        try:
            qv = int(float(qv))
        except (TypeError, ValueError):
            qv = 0
        if not sk or qv <= 0:
            continue
        res = allocation.allocate_by_forecast(db, sku=sk, qty=qv,
                                              branch_codes=brs or None)
        code = res.get("product") or sk          # what the typed text resolved to
        nm = res.get("description") or code
        eq += f"&sku={_url.quote(code)}&qty={qv}"
        rows.append({
            "sku": code, "qty": qv,
            "product": nm,
            "description": nm,
            "note": res.get("note"), "error": res.get("error"),
            "allocations": res.get("allocations", []),
            "allocated_total": res.get("allocated_total", 0),
            "warehouse": res.get("warehouse", 0),
            "slow": res.get("slow"),
        })
    for c in brs:
        eq += f"&alloc_branches={c}"

    # the same result, re-sorted branch-first instead of product-first - one
    # self-contained document per branch, for the "sort by branch" view and
    # the per-branch / ZIP downloads (mirrors weekly order's by_branch)
    code_by_name = {b.name: b.code.upper() for b in db.query(Branch).all()}
    by_branch_lines: dict = {}
    for r in rows:
        for a in r.get("allocations", []):
            if not a.get("allocated"):
                continue
            by_branch_lines.setdefault(a["branch"], []).append({
                "sku": r["sku"], "description": r.get("description") or r.get("product"),
                "predicted": a.get("predicted"), "rec": a.get("allocated"),
            })
    by_branch = []
    for bname, lines in by_branch_lines.items():
        lines.sort(key=lambda l: -l["rec"])
        by_branch.append({
            "code": code_by_name.get(bname, bname), "branch": bname,
            "lines": lines, "rec_total": sum(l["rec"] for l in lines),
            "req_total": 0,
        })
    by_branch.sort(key=lambda b: b["branch"])

    return {
        "rows": rows, "n": len(rows),
        "branches": ", ".join(brs) if brs else "all forecast branches",
        "by_branch": by_branch,
        "warnings": [], "grand_total": sum(r["qty"] for r in rows),
        "warehouse_total": sum(r["warehouse"] for r in rows),
        "export_q": eq.lstrip("&"),
    }


def _rec_qty(alloc: int, req: int) -> int:
    """Rec. Qty for the dispatch note: if the sales-based allocation agrees with
    the branch's request (within ~10% or 2 units) just honour the request,
    otherwise write the forecast-based number."""
    alloc, req = int(alloc or 0), int(req or 0)
    if req and abs(alloc - req) <= max(2, round(0.1 * req)):
        return req
    return alloc


def _run_weekly_split(db, pairs, branch_files) -> dict:
    """Weekly-order mode: split the warehouse quantity for each SKU across the
    branches that requested it, weighted by predicted weekly sales (what is
    likely to sell by week's end) and capped at each branch's requested amount;
    whatever is left stays at the warehouse.

    ``branch_files`` is ``[(branch_code, parsed_document), ...]``. The result
    also carries ``by_branch`` - one filled dispatch note per branch, with the
    Rec. Qty column set from the allocation.
    """
    from wms.services import stock as stock_svc

    st = weekly_fc.cached_run().get("state")
    name_by_code = {b.code.upper(): b.name for b in db.query(Branch).all()}
    fc, item_by_sku = {}, {}
    if st is not None and not st.empty:
        for r in st.itertuples():
            fc[(str(r.branch).upper(), str(r.sku).upper())] = float(r.weekly_demand or 0)
            item_by_sku.setdefault(str(r.sku).upper(), str(r.item or "").strip())
    prod_name = {p.sku.upper(): p.name for p in db.query(Product).all()}

    inv = stock_svc.levels_df(db)
    on_hand: dict = {}
    if not inv.empty:
        for r in inv.itertuples():
            on_hand[(str(r.branch_code).upper(), str(r.sku).upper())] = int(r.on_hand or 0)

    req: dict = {}
    seen_codes: list = []
    for bc, parsed in branch_files:
        bcu = str(bc or "").strip().upper()
        if bcu and bcu not in seen_codes:
            seen_codes.append(bcu)
        for ln in (parsed.get("lines") if isinstance(parsed, dict) else parsed) or []:
            sk = str(ln.get("sku") or "").strip().upper()
            q = ln.get("requested_qty")
            if not sk or q in (None, ""):
                continue
            req.setdefault(sk, {})
            key = bcu or "?"
            req[sk][key] = req[sk].get(key, 0) + int(q)

    alloc_by: dict = {}          # (sku_u, branch_u) -> allocated units
    rows = []
    for sk, qv in pairs:
        sk = str(sk or "").strip()
        sk_u = sk.upper()
        try:
            qv = int(float(qv))
        except (TypeError, ValueError):
            qv = 0
        if not sk or qv <= 0:
            continue
        nm = item_by_sku.get(sk_u) or prod_name.get(sk_u) or sk
        want = req.get(sk_u, {})
        if not want:
            rows.append({"sku": sk, "qty": qv, "product": nm, "description": nm,
                         "note": "no branch requested this product", "error": None,
                         "allocations": [], "allocated_total": 0, "warehouse": qv})
            continue
        weights = {bc: max(0.0, fc.get((bc, sk_u), 0.0)) for bc in want}
        caps = {bc: max(0, int(want[bc])) for bc in want}
        got = allocation._split_capped(qv, weights, caps)
        allocs = []
        for bc in sorted(want, key=lambda c: (-got.get(c, 0), c)):
            raw = int(got.get(bc, 0))
            rq = int(want[bc])
            rec = _rec_qty(raw, rq)                 # snap to the request when close
            alloc_by[(sk_u, bc)] = rec
            allocs.append({
                "branch": name_by_code.get(bc, bc),
                "predicted": int(round(fc.get((bc, sk_u), 0.0))),
                "requested": rq,
                "allocated": rec,
                "kind": "cover",
            })
        sent = min(qv, sum(a["allocated"] for a in allocs))
        wh = max(0, qv - sent)
        rows.append({
            "sku": sk, "qty": qv, "product": nm, "description": nm,
            "note": (f"{wh} unit(s) held at the warehouse (branches asked for less "
                     f"or the forecast caps it)" if wh else None),
            "error": None, "allocations": allocs,
            "allocated_total": sent, "warehouse": wh,
        })

    # one filled dispatch note per branch
    by_branch = []
    for bc, parsed in branch_files:
        bcu = str(bc or "").strip().upper()
        pd_lines = (parsed.get("lines") if isinstance(parsed, dict) else parsed) or []
        d = parsed.get("doc_date") if isinstance(parsed, dict) else None
        dn_lines = []
        for ln in pd_lines:
            sk_u = str(ln.get("sku") or "").strip().upper()
            if not sk_u:
                continue
            rq = int(ln.get("requested_qty") or 0)
            dn_lines.append({
                "sku": ln.get("sku"),
                "description": (ln.get("description")
                               or item_by_sku.get(sk_u) or prod_name.get(sk_u) or ""),
                "predicted": int(round(fc.get((bcu, sk_u), 0.0))),
                "on_hand": on_hand.get((bcu, sk_u), 0),
                "requested": rq,
                "sent": int(ln.get("sent_qty") or 0),
                "rec": int(alloc_by.get((sk_u, bcu), 0)),   # already snapped
            })
        by_branch.append({
            "code": bcu, "branch": name_by_code.get(bcu, bcu or "branch"),
            "doc_id": (parsed.get("stock_movement_id") if isinstance(parsed, dict) else None),
            "date": (d.isoformat() if d else ""),
            "lines": dn_lines,
            "rec_total": sum(l["rec"] for l in dn_lines),
            "req_total": sum(l["requested"] for l in dn_lines),
        })

    br_label = (", ".join(name_by_code.get(c, c) for c in seen_codes)
                or "branches in the uploaded files")
    return {
        "rows": rows, "n": len(rows), "weekly": True, "branches": br_label,
        "by_branch": by_branch,
        "warnings": [], "grand_total": sum(r["qty"] for r in rows),
        "warehouse_total": sum(r["warehouse"] for r in rows), "export_q": "",
    }


def _run_auto_weekly_order(db, brs: list[str], warehouse_pairs=None) -> dict:
    """Weekly-order mode with no uploaded branch request files: generate the
    order directly from real weekly sales history and current stock, no
    manual branch requests needed - the top products each branch is short on,
    ranked by how badly they're needed network-wide.

    A branch-product qualifies only if ALL of:
      * it has a real weekly demand forecast (actually sells there), even a
        slow one - this is not restricted to top-quartile fast movers;
      * it has sold in roughly the last 4 weeks at that branch (``recent_sales``)
        - a product with a positive long-run rate but nothing recent is
          treated as gone quiet, not "in need";
      * on-hand doesn't already cover the review week plus the dispatch
        transit delay (``target = ceil(weekly_demand x (7 + transit) / 7)``,
        the same cover the Weekly Allocation plan uses) - a branch that
        already has enough is skipped, not topped up further.

    ``warehouse_pairs`` (``[(sku, qty), ...]``), when given, is what is
    actually available to send out - each SKU's total need across the
    selected branches is then capped at that quantity and split between them
    weighted by weekly sales (never handing out more than there is), with any
    shortfall reported as held at the warehouse - same as an uploaded
    per-branch request file does, just without the requests.
    """
    from wms.services import stock as stock_svc
    from wms.config import get_settings

    st = weekly_fc.cached_run().get("state")
    if st is None or st.empty:
        return {"rows": [], "n": 0, "weekly": True, "branches": "", "by_branch": [],
                "warnings": ["No weekly sales history to generate an order from."],
                "grand_total": 0, "warehouse_total": 0, "export_q": ""}

    s = get_settings()
    cover_days = getattr(s, "review_period_days", 7) + getattr(s, "dispatch_transit_days", 3)

    inv = stock_svc.levels_df(db)
    on_hand: dict = {}
    if not inv.empty:
        for r in inv.itertuples():
            on_hand[(str(r.branch_code).upper(), str(r.sku).upper())] = int(r.on_hand or 0)

    name_by_code = {b.code.upper(): b.name for b in db.query(Branch).all()}
    want_bc = {c.upper() for c in brs} if brs else None

    warehouse_qty: dict = {}
    for sk, qv in (warehouse_pairs or []):
        sk_u = str(sk or "").strip().upper()
        try:
            qv = int(float(qv))
        except (TypeError, ValueError):
            continue
        if sk_u and qv > 0:
            warehouse_qty[sk_u] = warehouse_qty.get(sk_u, 0) + qv

    needs: dict = {}                  # sku_u -> {sku, description, per_branch: {bc: {...}}}
    for r in st.itertuples():
        bc = str(r.branch).upper()
        if want_bc and bc not in want_bc:
            continue
        rate = float(r.weekly_demand or 0)
        if rate <= 0:
            continue                                   # never actually forecast to sell here
        if float(getattr(r, "recent_sales", 0) or 0) <= 0:
            continue                                   # not sold there in the last ~4 weeks
        sku_u = str(r.sku).upper()
        target = int(np.ceil(rate * cover_days / 7))
        oh = on_hand.get((bc, sku_u), 0)
        need = target - oh
        if need <= 0:
            continue                                   # already covers the week + transit delay
        d = needs.setdefault(sku_u, {"sku": r.sku,
                                     "description": str(r.item or "").strip() or r.sku,
                                     "per_branch": {}})
        d["per_branch"][bc] = {"weekly_demand": round(rate, 1), "need": need, "on_hand": oh}

    rows = []
    by_branch_lines: dict = {}
    for sku_u, d in needs.items():
        total_need = sum(info["need"] for info in d["per_branch"].values())
        avail = warehouse_qty.get(sku_u)
        if avail is not None:
            weights = {bc: info["weekly_demand"] for bc, info in d["per_branch"].items()}
            caps = {bc: info["need"] for bc, info in d["per_branch"].items()}
            got = allocation._split_capped(avail, weights, caps)
        else:
            got = {bc: info["need"] for bc, info in d["per_branch"].items()}

        allocs = []
        total = 0
        for bc, info in sorted(d["per_branch"].items(), key=lambda kv: -kv[1]["need"]):
            rec = int(got.get(bc, 0))
            total += rec
            allocs.append({"branch": name_by_code.get(bc, bc), "predicted": info["weekly_demand"],
                           "requested": info["need"] if avail is not None else 0,
                           "allocated": rec, "kind": "cover"})
            by_branch_lines.setdefault(bc, []).append({
                # "requested" is 0 (not "need") when there is no warehouse cap
                # to honour - nothing was actually requested in this
                # no-file-attached path, only "rec" is real. When a warehouse
                # quantity IS on hand, "requested" carries the uncapped need
                # so the branch document can show what it's short by.
                "sku": d["sku"], "description": d["description"],
                "predicted": info["weekly_demand"], "on_hand": info["on_hand"],
                "requested": info["need"] if avail is not None else 0,
                "sent": 0, "rec": rec,
            })
        short = max(0, total_need - total)
        # "warehouse" is deliberately left 0 here (unlike _run_weekly_split):
        # a shortfall against real need isn't stock "held back" at the
        # warehouse - it's stock that doesn't exist - so it must not feed the
        # shared "X held at warehouse" summary line, which would say the
        # opposite of what's actually true
        rows.append({
            "sku": d["sku"], "qty": total_need, "product": d["description"],
            "description": d["description"],
            "note": (f"only {avail:,} on hand at the warehouse vs {total_need:,} "
                     f"needed - {short:,} unit(s) short" if short else None),
            "error": None, "allocations": allocs, "allocated_total": total,
            "warehouse": 0,
        })
    rows.sort(key=lambda r: -r["qty"])                 # top products in need first, network-wide

    by_branch = []
    for bc, lines in by_branch_lines.items():
        lines.sort(key=lambda l: -l["rec"])
        by_branch.append({
            "code": bc, "branch": name_by_code.get(bc, bc),
            "doc_id": None, "date": "",
            "lines": lines,
            "rec_total": sum(l["rec"] for l in lines),
            "req_total": sum(l["requested"] for l in lines),
        })
    by_branch.sort(key=lambda b: b["branch"])

    br_label = (", ".join(b["branch"] for b in by_branch)
                or "no branch currently needs anything")
    return {
        "rows": rows, "n": len(rows), "weekly": True, "branches": br_label,
        "by_branch": by_branch,
        "warnings": [], "grand_total": sum(r["qty"] for r in rows),
        "warehouse_total": 0, "export_q": "",
    }


def _save_last_split(batch: dict) -> None:
    """Persist the most recent split so the Excel / PDF buttons can render it
    without recomputing (works for both one-off and weekly-order results)."""
    import json
    from wms.config import get_settings
    p = pathlib.Path(get_settings().out) / "_last_split.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(batch, default=str), encoding="utf-8")
    except OSError:
        pass


def _dc_stock_pairs(db) -> list[tuple[str, int]]:
    """Warehouse stock on hand from Receiving Orders (stock_on_hand at the
    distribution centre) - the real, audited running total, not the old
    upload-a-snapshot flat file."""
    dc = db.query(Branch).filter(Branch.code == "DC").first()
    if not dc:
        return []
    rows = (db.query(StockOnHand.sku, StockOnHand.qty_on_hand)
            .filter(StockOnHand.branch_id == dc.id).all())
    return [(sku, qty) for sku, qty in rows if qty]


@router.post("/analytics/split")
def analytics_split(request: Request,
                    split_mode: str = Form("oneoff"),
                    stock_file: UploadFile = File(None),
                    ro_no: str = Form(""),
                    man_sku: list[str] = Form(default=[]),
                    man_qty: list[str] = Form(default=[]),
                    split_branches: list[str] = Form(default=[]),
                    wk_branch: list[str] = Form(default=[]),
                    wk_file: list[UploadFile] = File(default=[]),
                    db: Session = Depends(db_session),
                    user: User = Depends(require_perm("backorder.enter"))):
    """Unified split tool. The stock to distribute comes from the manual product
    lines, else one specific picked Receiving Order, else an uploaded stock
    file (a quick one-off override), else the warehouse's real stock on hand
    as recorded by Receiving Orders overall. One-off mode splits it across the
    picked branches; weekly-order mode splits it against per-branch
    order-request files."""
    mode = "weekly" if split_mode == "weekly" else "oneoff"
    warnings: list = []

    pairs = [(s, q) for s, q in zip(man_sku, man_qty)
             if str(s or "").strip() and str(q or "").strip()]
    src = "manual entry"
    ro_no = (ro_no or "").strip()
    if not pairs and ro_no:
        ro = db.query(ReceivingOrder).filter(ReceivingOrder.ro_no == ro_no).first()
        if ro:
            pairs = [(l.product.sku, l.received_qty) for l in ro.lines]
            src = f"receiving order {ro.ro_no}"
        else:
            warnings.append(f"Receiving order '{ro_no}' not found.")
    if not pairs and stock_file is not None and (stock_file.filename or ""):
        try:
            parsed = doc_import.parse_qty_list(stock_file.file.read(),
                                               stock_file.filename or "")
            pairs = [(l.get("sku"), l.get("qty")) for l in parsed.get("lines", [])]
            warnings += parsed.get("warnings", [])
            src = stock_file.filename
            # a one-off override for this split only - never persisted as
            # warehouse inventory (Receiving Orders is the real, tracked way
            # stock enters the warehouse)
        except Exception as e:                            # noqa: BLE001
            flash(request, f"Could not read the stock file: {e}", "error")
            return render(request, "analytics.html", user, split_mode=mode,
                          **_alloc_ctx(db))
    man_rows = [{"sku": (s or "").strip(), "qty": (q or "").strip()}
                for s, q in zip(man_sku, man_qty)]
    brs = [b.strip().upper() for b in split_branches if b and b.strip()]

    if mode == "weekly":
        branch_files = []
        for i, f in enumerate(wk_file or []):
            if not f or not (f.filename or ""):
                continue
            bc = wk_branch[i].strip() if i < len(wk_branch) else ""
            try:
                p = doc_import.parse_document(f.file.read(), f.filename or "")
            except Exception as e:                        # noqa: BLE001
                warnings.append(f"{f.filename}: {e}")
                continue
            branch_files.append((bc or p.get("branch") or "", p))

        if not branch_files:
            # no per-branch request files uploaded - generate the order
            # directly from real weekly sales history + current stock
            # (reorder-point style), rather than requiring one
            auto_pairs = pairs
            if not auto_pairs:
                auto_pairs = _dc_stock_pairs(db)
            batch = _run_auto_weekly_order(db, brs, warehouse_pairs=auto_pairs)
            src = ("system-generated, capped to warehouse stock on hand"
                  if auto_pairs else
                  "system-generated (real weekly sales history + current stock)")
        else:
            if not pairs:
                pairs = _dc_stock_pairs(db)
                src = "warehouse stock on hand (Receiving Orders)"
            if not pairs:
                # no warehouse quantity on hand either - a weekly order still
                # works from sales predictions alone: assume there is enough
                # stock to cover what every branch actually asked for, and let
                # the forecast (not a warehouse cap) decide each branch's Rec. Qty
                totals: dict = {}
                for _bc, parsed in branch_files:
                    lines = (parsed.get("lines") if isinstance(parsed, dict) else parsed) or []
                    for ln in lines:
                        sk = str(ln.get("sku") or "").strip()
                        q = ln.get("requested_qty")
                        if not sk or q in (None, ""):
                            continue
                        totals[sk] = totals.get(sk, 0) + int(q)
                pairs = list(totals.items())
                src = "branch order requests (no inventory data on hand)"

            batch = _run_weekly_split(db, pairs, branch_files)
    else:
        if not pairs:
            pairs = _dc_stock_pairs(db)
            src = "warehouse stock on hand (Receiving Orders)"
        if not pairs:
            flash(request, "Add stock to split: enter product lines, or record a "
                           "Receiving Order first.", "error")
            return render(request, "analytics.html", user, split_mode=mode,
                          man_rows=man_rows, split_branches=brs, **_alloc_ctx(db))
        batch = _run_split_batch(db, pairs, brs)
        batch["weekly"] = False

    batch["warnings"] = warnings + batch.get("warnings", [])
    batch["source"] = src
    _save_last_split(batch)
    verb = "Weekly order created" if mode == "weekly" else "Split"
    flash(request, f"{verb}: {batch['n']} product line(s) from {src}"
                   + (f"; {batch['warehouse_total']:,} held at warehouse."
                      if batch.get("warehouse_total") else "."), "success")
    return render(request, "analytics.html", user, split_batch=batch,
                  split_mode=mode, man_rows=man_rows, split_branches=brs,
                  **_alloc_ctx(db))


@router.get("/download/split-result")
def download_split_result(fmt: str = "xlsx",
                          user: User = Depends(require_login)):
    """Excel / PDF of the most recent split (one-off or weekly-order)."""
    import json
    from wms.config import get_settings
    p = pathlib.Path(get_settings().out) / "_last_split.json"
    if not p.exists():
        raise Redirect("/analytics?tab=allocation")
    batch = json.loads(p.read_text(encoding="utf-8"))
    if fmt == "pdf":
        if batch.get("weekly") and batch.get("by_branch"):
            path = pdf_export.weekly_dispatch_pdf(batch)   # filled Rec. Qty per branch
        else:
            path = pdf_export.split_batch_pdf(batch)
        return FileResponse(path, media_type="application/pdf", filename=path.name)
    path = excel.split_batch_workbook(batch)
    return FileResponse(path, media_type=_XLSX, filename=path.name)


def _safe_filename(text: str) -> str:
    s = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(text or "")).strip()
    return s or "branch"


def _load_last_split() -> dict:
    import json
    from wms.config import get_settings
    p = pathlib.Path(get_settings().out) / "_last_split.json"
    if not p.exists():
        raise Redirect("/analytics?tab=allocation")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/download/split-result/branch/{code}")
def download_split_result_branch(code: str, fmt: str = "pdf",
                                 user: User = Depends(require_login)):
    """One branch's own order from the most recent split (one-off or weekly),
    as its own PDF or Excel file - for downloading a single branch's order on
    its own rather than the combined multi-branch result."""
    batch = _load_last_split()
    by_branch = batch.get("by_branch") or []
    bb = next((b for b in by_branch
              if str(b.get("code") or "").upper() == code.strip().upper()), None)
    if bb is None:
        raise Redirect("/analytics?tab=allocation")
    doc_title = "Weekly order" if batch.get("weekly") else "Split allocation"
    if fmt == "xlsx":
        path = excel.weekly_dispatch_branch_workbook(bb, doc_title=doc_title)
        return FileResponse(path, media_type=_XLSX, filename=path.name)
    path = pdf_export.weekly_dispatch_pdf({"by_branch": [bb]},
                                          title=f"{doc_title.lower()} allocation")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.get("/download/split-result/zip")
def download_split_result_zip(doc: str = "pdf",
                              user: User = Depends(require_login)):
    """A split that covers more than one branch (one-off or weekly): one PDF
    (or Excel) file per branch, zipped together so each branch's own order
    can be handed out separately."""
    import zipfile
    from wms.config import get_settings

    batch = _load_last_split()
    by_branch = batch.get("by_branch") or []
    if not by_branch:
        raise Redirect("/analytics?tab=allocation")
    doc_title = "Weekly order" if batch.get("weekly") else "Split allocation"

    zpath = (pathlib.Path(get_settings().out)
            / f"orders_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip")
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for bb in by_branch:
            name = _safe_filename(bb.get("branch") or bb.get("code"))
            if doc == "xlsx":
                fpath = excel.weekly_dispatch_branch_workbook(bb, doc_title=doc_title)
                arcname = f"{name}.xlsx"
            else:
                fpath = pdf_export.weekly_dispatch_pdf(
                    {"by_branch": [bb]}, title=f"{doc_title.lower()} allocation")
                arcname = f"{name}.pdf"
            zf.write(fpath, arcname=arcname)
    return FileResponse(zpath, media_type="application/zip", filename=zpath.name)


@router.post("/analytics/upload")
def analytics_upload(request: Request, kind: str = Form(...),
                     branch_code: str = Form(""), file: UploadFile = File(...),
                     db: Session = Depends(db_session),
                     user: User = Depends(require_perm("backorder.enter"))):
    """Accept a monthly sales export or a branch stock-on-hand snapshot."""
    name = (file.filename or "").strip()
    try:
        if kind == "inventory":
            bc = branch_code.strip().upper()
            if not bc:
                raise ValueError("Pick the branch this inventory file is for.")
            br = db.query(Branch).filter(Branch.code == bc).first()
            if br is None:
                raise ValueError(f"Unknown branch code '{bc}'.")
            bslice = inv_mod.parse_upload(file.file.read(), name, bc)
            rows = int(len(bslice))
            # the upload REPLACES this branch's stock-on-hand balance
            stock_svc.set_branch_stock(
                db, branch_id=br.id,
                items=[{"sku": r.sku, "qty": int(r.on_hand)}
                       for r in bslice.itertuples()], user_id=user.id)
            flash(request, f"Inventory for {bc}: {rows} product line(s) loaded, "
                           f"see the Inventory page.", "success")
            return RedirectResponse("/inventory", 303)

        if kind == "sales":
            bc = branch_code.strip().upper()
            if not bc:
                raise ValueError("Pick the branch this sales file is for.")
            month, _bc, _yr = monthly_sales._parse_name(name)
            if not month:
                raise ValueError("Name the file with its month, e.g. 'AUGUST SALES.xlsx'.")
            rows = monthly_sales.parse_upload(file.file.read(), name, branch_code=bc)
            if rows.empty:
                raise ValueError("Could not read any product rows from that file.")
            period = rows["period"].iloc[0]
            day_from, day_to = int(rows["day_from"].iloc[0]), int(rows["day_to"].iloc[0])
            n_saved = monthly_sales.merge_month(bc, period, rows)
            demand_forecast._CACHE.clear()
            panel = monthly_sales.load_panel()
            mon_title = period.strftime("%B")
            partial = (day_from, day_to) != (1, int(period.day))
            period_lbl = f"{mon_title} {day_from}-{day_to}" if partial else mon_title
            msg = (f"Sales for {bc} {period_lbl} added ({n_saved} line(s)), history now "
                  f"{monthly_sales.coverage(panel).get('month_range', '')}.")
            flash(request, msg, "success")
            return RedirectResponse("/analytics?tab=demand", 303)

        raise ValueError("Unknown upload type.")
    except Exception as e:
        flash(request, f"Upload failed: {e}", "error")
        back = "/inventory" if kind == "inventory" else "/analytics?tab=demand"
        return RedirectResponse(back, 303)


def _retrain_weekly_model_in_background() -> None:
    """Runs after the upload response is sent (see BackgroundTasks below) - a
    thorough retrain takes real time (minutes if the neural net is enabled,
    seconds otherwise), so it must never block the upload request itself.
    Persists straight into the database (see sync_checkpoints_to_db in
    weekly_forecast.py) so the trained weights survive a restart on a host
    with no persistent disk."""
    try:
        weekly_fc.train_and_save(quick=True)
    except Exception as e:                                # noqa: BLE001
        import warnings
        warnings.warn(f"weekly auto-retrain failed: {e}")


@router.post("/analytics/upload-weekly")
def analytics_upload_weekly(request: Request, background_tasks: BackgroundTasks,
                            files: list[UploadFile] = File(...),
                            branch_code: str = Form(""),
                            db: Session = Depends(db_session),
                            user: User = Depends(require_perm("backorder.enter"))):
    """Accept many weekly branch sales exports at once. The week is read from each
    filename (``<BRANCH> DD-MM-YYYY to DD-MM-YYYY Sales.xlsx``); the branch is
    read from the name too, unless a ``branch_code`` is picked in the form, which
    then applies to every file in the upload. Newest upload for a given
    branch-week replaces the old one."""
    picked = (branch_code or "").strip().upper()
    valid = {b.code.upper() for b in db.query(Branch).all()}
    if picked and picked not in valid:
        picked = ""
    saved, skipped = 0, []
    for f in files or []:
        name = (f.filename or "").strip()
        if not name:
            continue
        ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ".xlsx"
        if ext not in (".xlsx", ".xls"):
            skipped.append(f"{name} (not a spreadsheet)")
            continue
        parsed = weekly_fc.parse_upload_sales(f.file.read(), name, branch_code=picked or None)
        if not parsed:
            skipped.append(f"{name} (no {'week date' if picked else 'branch/week'} in name, "
                           "or unreadable)")
            continue
        code, ws, rows = parsed
        weekly_fc.save_week(code, ws, rows)
        saved += 1
    if saved:
        cov = weekly_fc.cached_run()["coverage"]
        background_tasks.add_task(_retrain_weekly_model_in_background)
        flash(request, f"{saved} weekly file(s) loaded, model now covers "
                       f"{', '.join(cov.get('branches', []))} over {cov.get('weeks', 0)} "
                       f"weeks ({cov.get('week_range', '')}). Retraining in the "
                       f"background - the improved model will be live shortly.", "success")
    if skipped:
        flash(request, "Skipped: " + "; ".join(skipped[:6])
              + (" …" if len(skipped) > 6 else ""), "error")
    if not saved and not skipped:
        flash(request, "No files received.", "error")
    return RedirectResponse("/analytics?tab=demand", 303)


@router.post("/analytics/upload-weekly-inventory")
def analytics_upload_weekly_inventory(request: Request,
                                      files: list[UploadFile] = File(...),
                                      branch_code: str = Form(""),
                                      db: Session = Depends(db_session),
                                      user: User = Depends(require_perm("backorder.enter"))):
    """Accept weekly Hansa stock-on-hand exports (one per branch-week) - used
    to spot stockout weeks in the weekly demand model. Same naming/branch-pick
    rules as the weekly sales upload above."""
    picked = (branch_code or "").strip().upper()
    valid = {b.code.upper() for b in db.query(Branch).all()}
    if picked and picked not in valid:
        picked = ""
    saved, skipped = 0, []
    for f in files or []:
        name = (f.filename or "").strip()
        if not name:
            continue
        ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ".xlsx"
        if ext not in (".xlsx", ".xls"):
            skipped.append(f"{name} (not a spreadsheet)")
            continue
        parsed = weekly_fc.parse_upload_inventory(f.file.read(), name, branch_code=picked or None)
        if not parsed:
            skipped.append(f"{name} (no {'week date' if picked else 'branch/week'} in name, "
                           "or no quantity column found)")
            continue
        code, ws, rows = parsed
        weekly_fc.save_inventory_week(code, ws, rows)
        saved += 1
    if saved:
        flash(request, f"{saved} weekly stock file(s) loaded.", "success")
    if skipped:
        flash(request, "Skipped: " + "; ".join(skipped[:6])
              + (" …" if len(skipped) > 6 else ""), "error")
    if not saved and not skipped:
        flash(request, "No files received.", "error")
    return RedirectResponse("/analytics?tab=demand", 303)


# ======================================================================
# INVENTORY
# ======================================================================
@router.get("/inventory")
def inventory_page(request: Request, bcode: str = "", q: str = "", low_bcode: str = "",
                   excess_bcode: str = "",
                   db: Session = Depends(db_session),
                   user: User = Depends(require_login)):
    """Current stock-on-hand per branch: a per-branch summary (line count,
    total units, when it was last uploaded), a searchable, branch-filterable
    detail table, and computed planning views (low-stock and excess-stock
    alerts) derived from that snapshot plus the weekly demand forecast.
    Read-only + the snapshot upload - no ledger, adjustments, counts or
    ASNs (see README)."""
    branches = db.query(Branch).order_by(Branch.name).all()
    levels = stock_svc.levels_df(db)
    prod_name = {p.sku.upper(): p.name for p in db.query(Product).all()}
    item_by_sku: dict = {}
    st = weekly_fc.cached_run().get("state") if weekly_fc.has_data() else None
    if st is not None and not st.empty:
        for r in st.itertuples():
            item_by_sku.setdefault(str(r.sku).upper(), str(r.item or "").strip())

    by_branch = (levels.groupby("branch_code").agg(lines=("sku", "nunique"),
                                                    units=("on_hand", "sum"))
                if not levels.empty else None)
    total_branches = int(len(by_branch)) if by_branch is not None else 0

    rows = []
    if not levels.empty:
        det = levels[levels["on_hand"] > 0].copy()
        if bcode.strip():
            det = det[det["branch_code"].str.upper() == bcode.strip().upper()]
        name_by_code = {b.code.upper(): b.name for b in branches}
        if q.strip():
            s = q.strip().lower()
            names = det["sku"].str.upper().map(
                lambda sk: prod_name.get(sk) or item_by_sku.get(sk) or "")
            det = det[det["sku"].str.lower().str.contains(s, regex=False)
                     | names.str.lower().str.contains(s, regex=False)]
        det = det.sort_values(["branch_code", "on_hand"], ascending=[True, False])
        for r in det.itertuples():
            sku_u = str(r.sku).upper()
            rows.append({
                "Branch": name_by_code.get(r.branch_code.upper(), r.branch_code),
                "SKU": r.sku,
                "Product": prod_name.get(sku_u) or item_by_sku.get(sku_u) or "",
                "On hand": int(r.on_hand),
            })

    return render(request, "inventory.html", user, branches=branches,
                  rows=rows, bcode=bcode, q=q,
                  total_branches=total_branches,
                  total_lines=int(len(levels)) if not levels.empty else 0,
                  total_units=int(levels["on_hand"].sum()) if not levels.empty else 0,
                  low_stock=weekly_fc.low_stock_alerts(db, bcode=low_bcode),
                  low_bcode=low_bcode,
                  excess=weekly_fc.excess_stock(db, bcode=excess_bcode),
                  excess_bcode=excess_bcode)


# ======================================================================
# PRODUCTS
# ======================================================================
@router.get("/products")
def products_page(request: Request, q: str = "", category: str = "",
                  db: Session = Depends(db_session), user: User = Depends(require_perm("nav.full"))):
    """The product catalogue: browse/search, and (with products.manage) add
    a new product or edit an existing one by SKU.

    This is a SEPARATE, smaller table (currently ~90 rows, seeded from one
    historical document - see seed.py) from the much larger SKU universe the
    weekly sales / inventory file uploads carry on their own; it exists as a
    name/category/price reference for SKUs those uploads don't otherwise
    resolve (e.g. a branch-inventory line with no matching weekly sales
    history has no name to show unless it's catalogued here)."""
    query = db.query(Product)
    if q.strip():
        s = f"%{q.strip()}%"
        query = query.filter(or_(Product.sku.ilike(s), Product.name.ilike(s)))
    if category.strip():
        query = query.filter(Product.category == category.strip())
    products = query.order_by(Product.sku).limit(500).all()
    total = db.query(Product).count()
    categories = sorted({c for (c,) in db.query(Product.category).distinct() if c})
    return render(request, "products.html", user, products=products, q=q,
                  category=category, categories=categories, total=total,
                  shown=len(products))


@router.post("/products/new")
def products_new(request: Request, sku: str = Form(...), name: str = Form(...),
                 category: str = Form(""), uom: str = Form("EA"),
                 unit_price: str = Form(""),
                 db: Session = Depends(db_session),
                 user: User = Depends(require_perm("products.manage"))):
    sku = sku.strip().upper()
    name = name.strip()
    if not sku or not name:
        flash(request, "SKU and product name are both required.", "error")
        return RedirectResponse("/products", 303)
    price = None
    if unit_price.strip():
        try:
            price = float(unit_price)
        except ValueError:
            flash(request, f"Unit price '{unit_price}' isn't a number.", "error")
            return RedirectResponse("/products", 303)
    existing = db.query(Product).filter(Product.sku == sku).first()
    catalog.upsert_product(db, sku=sku, name=name, category=category.strip() or None,
                           uom=uom.strip() or "EA", unit_price=price, user_id=user.id)
    flash(request, f"{'Updated' if existing else 'Added'} product {sku} — {name}.",
          "success")
    return RedirectResponse(f"/products?q={sku}", 303)


# ======================================================================
# REPORTS / EXPORTS / CHARTS
# ======================================================================
@router.get("/reports")
def reports(request: Request, user: User = Depends(require_login)):
    return render(request, "reports.html", user)


@router.get("/download/{kind}")
def download(kind: str, request: Request, bcode: str = "", q: str = "",
             alloc_sku: str = "", alloc_qty: str = "",
             alloc_branches: list[str] = Query(default_factory=list),
             sku: list[str] = Query(default_factory=list),
             qty: list[str] = Query(default_factory=list),
             db: Session = Depends(db_session),
             user: User = Depends(require_login)):
    _abr = [b.strip().upper() for b in alloc_branches if b.strip()] or None
    _aq = _bid(alloc_qty) or 0
    _pairs = list(zip(sku, qty))
    builders = {
        "backorders": lambda: (excel.backorder_workbook(db), _XLSX),
        "backorder-flow": lambda: (excel.backorder_flow_workbook(db), _XLSX),
        "branch-sales": lambda: (excel.sales_workbook(db), _XLSX),
        "suggested-orders": lambda: (excel.suggested_orders_workbook(db), _XLSX),
        "demand-forecast": lambda: (excel.demand_forecast_workbook(bcode=bcode, q=q), _XLSX),
        "allocation-plan": lambda: (excel.allocation_workbook(db, branch_code=bcode, q=q), _XLSX),
        "allocation-plan-pdf": lambda: (pdf_export.allocation_plan_pdf(db, branch_code=bcode),
                                        "application/pdf"),
        "split-allocation": lambda: (excel.split_allocation_workbook(
            db, sku=alloc_sku.strip(), qty=_aq, branch_codes=_abr), _XLSX),
        "split-allocation-pdf": lambda: (pdf_export.split_allocation_pdf(
            db, sku=alloc_sku.strip(), qty=_aq, branch_codes=_abr), "application/pdf"),
        "split-batch": lambda: (excel.split_allocation_batch_workbook(
            db, pairs=_pairs, branch_codes=_abr), _XLSX),
        "split-batch-pdf": lambda: (pdf_export.split_allocation_batch_pdf(
            db, pairs=_pairs, branch_codes=_abr), "application/pdf"),
        "weekly-order": lambda: (excel.weekly_order_workbook(db, branch_code=bcode), _XLSX),
        "branch-stats": lambda: (excel.branch_stats_workbook(db), _XLSX),
        "back-orders-csv": lambda: (csv_export.dataframe_to_csv(
            loaders.back_orders_df(db), "back_orders"), "text/csv"),
        "sales-csv": lambda: (csv_export.dataframe_to_csv(
            monthly_sales.cached_panel(), "sales"), "text/csv"),
    }
    if kind not in builders:
        raise Redirect("/reports")
    path, media = builders[kind]()
    return FileResponse(path, media_type=media, filename=path.name)


@router.post("/charts")
def make_charts(request: Request, pack: str = Form("dashboard"),
                db: Session = Depends(db_session), user: User = Depends(require_login)):
    from wms.viz import charts
    fn = charts.backorder_flow_pack if pack == "backorder-flow" else charts.dashboard_pack
    paths = fn(db)
    request.session["charts"] = [p.name for p in paths]
    flash(request, f"Generated {len(paths)} chart(s).", "success")
    return RedirectResponse(request.headers.get("referer") or "/reports", 303)
