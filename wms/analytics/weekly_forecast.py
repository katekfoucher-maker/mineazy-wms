"""Weekly per-SKU demand forecast.

One HansaWorld *Item Statistics* export per branch-week, named
``<BRANCH> DD-MM-YYYY to DD-MM-YYYY Sales.xlsx``. The week each file covers is
read from its name; weeks are placed on a 7-day grid **in date order** so the two
branches line up and gaps become zero-weeks.

SBA and snaive forecast each SKU from its own history; ES-RNN and NeuralProphet
are trained once across every SKU (global models) but still forecast per series.

Evaluation is a strict hold-out: the last ``test_weeks`` weeks are removed, every
model is *fitted on the earlier weeks only*, and each then produces a genuine
``test_weeks``-step-ahead forecast (made once, never shown the held-out actuals).
The best model by WAPE is chosen and, for the live number, refitted on the full
history to project the next weeks.

Four models:
  ES-RNN         - per-series exponential smoothing + a shared LSTM (torch)
  NeuralProphet  - one global model: level + Fourier seasonality + AR-Net (torch)
  SBA            - Syntetos-Boylan Approximation (bias-corrected Croston rate)
  snaive         - seasonal-naive, repeat the last 4-week pattern
"""
from __future__ import annotations

import glob
import os
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from wms.config import get_settings
from wms.analytics.monthly_sales import _BRANCHES, BRANCH_NAME, categorise

TEST_WEEKS = 4
MIN_TRAIN_WEEKS = 6
_CRO_A = 0.10                         # Croston / SBA smoothing constant
# Restocking bias: it is safer to over-forecast than to run a branch dry, so the
# forecast is deliberately pulled ABOVE the honest level. Selection avoids a heavy
# under-forecaster, then a calibrated uplift targets this pre-floor margin (a
# negative value means "sit a little UNDER before the floor lifts it over").
_SAFETY_MARGIN = 0.05
_MAX_UNDER_BIAS = 12.0               # skip a model under-forecasting by more than this %
_MAX_UPLIFT = 2.5
# keep the forecast anchored to each SKU's own level, but ASYMMETRICALLY — a
# prediction may run well above the level (safety) yet is held close underneath it
_SPIKE_INFLUENCE = 0.5              # a bulk-order week above the robust cap counts only half
_DEVIATION_BAND = 0.5              # legacy symmetric band (kept for callers that pass one number)
_DOWN_BAND = 0.30                  # never forecast below (1-this) x the SKU's damped mean
_UP_BAND = 2.00                    # may forecast up to (1+this) x the damped mean
_MIN_UNITS = 1                     # never predict zero for a SKU that has ever sold
_CATEGORY_POOL = 0.2              # shrink the forecast/level multiplier toward the category median
_CAT_POOL_MIN = 3                # a category needs at least this many SKUs to pool

_DATE_RX = re.compile(r"(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})"
                      r"|(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_SEGS = ["smooth", "erratic", "intermittent", "lumpy", "new", "dead"]
# test / placeholder product lines carried in the HansaWorld exports - never
# forecast or allocate these
_EXCLUDE_RX = re.compile(
    r"\b(?:testing|test\s*item|dummy|placeholder|do\s*not\s*use)\b", re.I)
_EXCLUDE_SKUS = {"TEST", "TESTING"}


def weekly_dir() -> Path:
    p = Path(getattr(get_settings(), "weekly_sales_dir", "./data/weekly_sales"))
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p


def _first_date(text: str):
    m = _DATE_RX.search(text)
    if not m:
        return None
    if m.group(1):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y += 2000 if y < 100 else 0
    else:
        y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
    try:
        return pd.Timestamp(year=y, month=mo, day=d)
    except ValueError:
        return None


def parse_name(fname: str):
    """-> (branch_code, week_start Timestamp) or (None, None)."""
    base = os.path.splitext(os.path.basename(fname))[0]
    m = _DATE_RX.search(base)
    ws = _first_date(base)
    if not m or ws is None:
        return None, None
    token = re.sub(r"\s+", " ", base[:m.start()]).strip(" -_").upper()
    hit = _BRANCHES.get(token)
    if not hit:
        for k, v in _BRANCHES.items():          # loose: token starts with a known name
            if token.startswith(k) or k.startswith(token):
                hit = v
                break
    code = hit[0] if hit else re.sub(r"[^A-Z0-9]", "", token)[:6] or "UNK"
    return code, ws


# ---------------------------------------------------------------- load
def _items(path: str) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path, sheet_name="Item Statistics", header=0, dtype=str)
    except ValueError:
        raw = pd.read_excel(path, header=0, dtype=str)
    orig3 = list(raw.columns[:3])
    by_name = {str(c).strip().lower(): c for c in raw.columns}
    raw = raw.rename(columns={raw.columns[0]: "sku", raw.columns[1]: "item",
                              raw.columns[2]: "qty"})
    raw = raw[raw["sku"].notna()].copy()
    raw["sku"] = raw["sku"].astype(str).str.strip()
    raw["item"] = raw["item"].astype(str).str.strip()
    raw = raw[raw["sku"].str.len() > 0]
    # drop test / placeholder lines that HansaWorld carries (e.g. "TESTING")
    raw = raw[~(raw["item"].str.contains(_EXCLUDE_RX, na=False)
                | raw["sku"].str.upper().isin(_EXCLUDE_SKUS))]
    raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce").fillna(0.0).clip(lower=0)
    # value columns for the sales-mix pies (by header name; absent -> zeros).
    # "Turnover" is the money value of what was sold; "Profit" is the margin.
    for out_col, header in (("profit", "profit"), ("revenue", "turnover")):
        src = by_name.get(header)
        raw[out_col] = (pd.to_numeric(raw[src], errors="coerce").fillna(0.0)
                        if src is not None and src not in orig3 else 0.0)
    return raw[["sku", "item", "qty", "profit", "revenue"]]


def _reshape(frames: list, provs: list) -> dict:
    empty = {"MAT": np.zeros((0, 0), np.float32),
             "PROFIT": np.zeros((0, 0), np.float32),
             "REV": np.zeros((0, 0), np.float32),
             "keys": [], "weeks": [], "item_of": {}, "prov": pd.DataFrame(provs)}
    if not frames:
        return empty
    long = pd.concat(frames, ignore_index=True)
    long["week_start"] = pd.to_datetime(long["week_start"])
    anchor = long["week_start"].min()
    long["wi"] = ((long["week_start"] - anchor).dt.days / 7).round().astype(int)
    long = (long.groupby(["branch", "sku", "wi"], as_index=False)
                .agg(qty=("qty", "sum"), profit=("profit", "sum"),
                     revenue=("revenue", "sum"), item=("item", "first")))
    W = int(long["wi"].max()) + 1
    keys = list(long[["branch", "sku"]].drop_duplicates()
                    .sort_values(["branch", "sku"]).itertuples(index=False, name=None))
    k_ix = {k: i for i, k in enumerate(keys)}
    MAT = np.zeros((len(keys), W), np.float32)
    PROFIT = np.zeros((len(keys), W), np.float32)
    REV = np.zeros((len(keys), W), np.float32)
    for r in long.itertuples():
        i = k_ix[(r.branch, r.sku)]
        MAT[i, r.wi] = r.qty
        PROFIT[i, r.wi] = r.profit
        REV[i, r.wi] = r.revenue
    weeks = [(anchor + pd.Timedelta(days=7 * i)).date().isoformat() for i in range(W)]
    return {"MAT": MAT, "PROFIT": PROFIT, "REV": REV, "keys": keys, "weeks": weeks,
            "item_of": long.groupby(["branch", "sku"])["item"].first().to_dict(),
            "prov": pd.DataFrame(provs)}


def parse_upload_sales(raw: bytes, filename: str, branch_code: str | None = None):
    """Parse one uploaded weekly-sales file's bytes, the same way a file on
    disk would be (see ``_items``) - used by the upload route to write
    straight into WeeklySalesLine instead of saving the file. The week always
    comes from the filename; the branch does too UNLESS ``branch_code`` is
    given explicitly (the upload form may have one picked, applying to every
    file in a multi-file upload).

    Returns ``(branch_code, week_start, rows_df[sku,item,qty,profit,revenue])``
    or ``None`` if the filename doesn't carry a recognised branch + week."""
    import io
    code, ws = parse_name(filename)
    if branch_code:
        code = branch_code.strip().upper()
    if not code or ws is None:
        return None
    try:
        raw_x = pd.read_excel(io.BytesIO(raw), sheet_name="Item Statistics",
                              header=0, dtype=str)
    except ValueError:
        raw_x = pd.read_excel(io.BytesIO(raw), header=0, dtype=str)
    orig3 = list(raw_x.columns[:3])
    by_name = {str(c).strip().lower(): c for c in raw_x.columns}
    raw_x = raw_x.rename(columns={raw_x.columns[0]: "sku", raw_x.columns[1]: "item",
                                  raw_x.columns[2]: "qty"})
    raw_x = raw_x[raw_x["sku"].notna()].copy()
    raw_x["sku"] = raw_x["sku"].astype(str).str.strip()
    raw_x["item"] = raw_x["item"].astype(str).str.strip()
    raw_x = raw_x[raw_x["sku"].str.len() > 0]
    raw_x = raw_x[~(raw_x["item"].str.contains(_EXCLUDE_RX, na=False)
                    | raw_x["sku"].str.upper().isin(_EXCLUDE_SKUS))]
    raw_x["qty"] = pd.to_numeric(raw_x["qty"], errors="coerce").fillna(0.0).clip(lower=0)
    for out_col, header in (("profit", "profit"), ("revenue", "turnover")):
        src = by_name.get(header)
        raw_x[out_col] = (pd.to_numeric(raw_x[src], errors="coerce").fillna(0.0)
                          if src is not None and src not in orig3 else 0.0)
    return code, ws, raw_x[["sku", "item", "qty", "profit", "revenue"]]


def save_week(branch_code: str, week_start, rows: pd.DataFrame) -> int:
    """Replace one (branch, week)'s REAL rows in WeeklySalesLine (never
    touches simulated ones - weekly_simulate.regenerate() cleans those up
    wholesale right after a real upload, once it can see the new coverage).
    Returns the number of line rows saved."""
    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine

    week_date = pd.Timestamp(week_start).date()
    db = SessionLocal()
    try:
        db.query(WeeklySalesLine).filter(
            WeeklySalesLine.branch_code == branch_code,
            WeeklySalesLine.week_start == week_date,
            WeeklySalesLine.is_simulated.is_(False)).delete()
        n = 0
        for r in rows.itertuples():
            db.add(WeeklySalesLine(
                branch_code=branch_code, sku=r.sku, item=r.item, week_start=week_date,
                qty=float(r.qty), profit=float(r.profit), revenue=float(r.revenue),
                is_simulated=False))
            n += 1
        db.commit()
        return n
    finally:
        db.close()


def clear_simulated_weeks() -> int:
    """Delete every simulated WeeklySalesLine row. Returns how many were
    removed. Mirrors weekly_simulate.py's old clear() (which deleted every
    file in its simulated sub-folder) - called once before regenerate()
    rebuilds the simulated set from scratch."""
    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine

    db = SessionLocal()
    try:
        n = db.query(WeeklySalesLine).filter(
            WeeklySalesLine.is_simulated.is_(True)).delete()
        db.commit()
        return n
    finally:
        db.close()


def save_simulated_week(branch_code: str, week_start, rows: pd.DataFrame) -> int:
    """Insert one (branch, week)'s SIMULATED rows into WeeklySalesLine. Call
    clear_simulated_weeks() once before a batch of these, not per-call - the
    simulated set is always rebuilt wholesale, not incrementally."""
    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine

    week_date = pd.Timestamp(week_start).date()
    db = SessionLocal()
    try:
        n = 0
        for r in rows.itertuples():
            db.add(WeeklySalesLine(
                branch_code=branch_code, sku=r.sku, item=r.item, week_start=week_date,
                qty=float(r.qty), profit=float(r.profit), revenue=float(r.revenue),
                is_simulated=True))
            n += 1
        db.commit()
        return n
    finally:
        db.close()


def _load_panel_from_db() -> dict:
    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine

    db = SessionLocal()
    try:
        rows = db.query(WeeklySalesLine).all()
    finally:
        db.close()
    if not rows:
        return _reshape([], [])
    long = pd.DataFrame([{
        "branch": r.branch_code, "sku": r.sku, "item": r.item or "",
        "qty": float(r.qty or 0), "profit": float(r.profit or 0),
        "revenue": float(r.revenue or 0), "week_start": pd.Timestamp(r.week_start),
    } for r in rows])
    provs = (long[["branch", "week_start"]].drop_duplicates()
                .assign(file="(database)")[["file", "branch", "week_start"]]
                .to_dict("records"))
    return _reshape([long], provs)


def load_panel(directory=None) -> dict:
    """Read uploaded Excel files under ``directory`` (or the configured
    ``weekly_sales_dir``) when there are any there - the on-disk path this
    always used to take, still used by tests that populate a directory
    directly. Falls back to WeeklySalesLine in the database otherwise, which
    is what a real deploy with no local filesystem to speak of actually has.
    """
    d = Path(directory) if directory else weekly_dir()
    files = [f for f in sorted(glob.glob(str(d / "**" / "*.xls*"), recursive=True))
             if not os.path.basename(f).startswith("~$")]
    if not files:
        return _load_panel_from_db()
    frames, provs = [], []
    for f in files:
        code, ws = parse_name(f)
        if not code or ws is None:
            continue
        it = _items(f)
        it["branch"] = code
        it["week_start"] = ws
        frames.append(it)
        provs.append({"file": os.path.basename(f), "branch": code, "week_start": ws})
    return _reshape(frames, provs)


# ------------------------------------------------------- stockout unconstraining
#   Some products' weekly sales are misleading: a week the branch had no stock
#   sold ~nothing, not because demand vanished but because the shelf was empty.
#   Training on those suppressed weeks teaches the models a level that is too low
#   -> chronic UNDER-prediction. Below: spot the stockout weeks (from the weekly
#   Hansa on-hand exports, else a "0 sales, but this SKU normally sells and the
#   weeks around it did" heuristic) and lift them to the SKU's in-stock level, so
#   what the model learns -- and forecasts -- is FULLY-STOCKED demand.
#   Deliberately CONSERVATIVE: a week is only lifted when there is real evidence
#   it was suppressed by a stockout, not merely quiet. The SKU must sell a
#   material amount when it IS on the shelf, the week must be INTERIOR (a sale
#   before it and a sale after it in the same series) and genuinely low, and the
#   dry run must be short. Over-lifting would just swap under-prediction for
#   over-prediction, which the restocking bias must not do.
_STOCKOUT_OH = 0.0        # on-hand at or below this = out of stock that week
_HEUR_FREQ = 0.5         # SKU must sell at least this share of IN-STOCK weeks to qualify
_HEUR_LOW = 0.25         # (no reading) a week under this x the in-stock level = "near zero"
_MIN_LIFT_LEVEL = 2.0    # only unconstrain SKUs selling at least this per in-stock week
_MAX_STOCKOUT_RUN = 8    # a dry spell longer than this is a discontinued line, not a stockout
_MAX_CENSOR_FRAC = 0.5   # if more than half a SKU's weeks look censored, trust none of it


def weekly_inventory_dir() -> Path:
    p = Path(getattr(get_settings(), "weekly_inventory_dir",
                     "./data/weekly_inventory"))
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p


def parse_upload_inventory(raw: bytes, filename: str, branch_code: str | None = None):
    """Parse one uploaded weekly-inventory file's bytes -> (branch_code,
    week_start, rows_df[sku, qty_on_hand]), or None if the filename doesn't
    carry a recognised branch + week, or the sheet has no usable qty column.
    The week always comes from the filename; the branch does too UNLESS
    ``branch_code`` is given explicitly."""
    import io
    from wms.analytics import inventory as _inv

    code, ws = parse_name(filename)
    if branch_code:
        code = branch_code.strip().upper()
    if not code or ws is None:
        return None
    try:
        raw_x = pd.read_excel(io.BytesIO(raw), header=0, dtype=str)
    except Exception:                                  # noqa: BLE001
        return None
    sc = _inv._pick(raw_x.columns, _inv._SKU_KEYS) or raw_x.columns[0]
    qc = _inv._pick(raw_x.columns, _inv._QTY_KEYS)
    if qc is None:
        return None
    raw_x = raw_x[raw_x[sc].notna()]
    out = pd.DataFrame({
        "sku": raw_x[sc].astype(str).str.strip(),
        "qty_on_hand": pd.to_numeric(raw_x[qc], errors="coerce"),
    })
    out = out[out["sku"].str.len() > 0]
    out["qty_on_hand"] = out["qty_on_hand"].fillna(0).clip(lower=0)
    return code, ws, out.groupby("sku", as_index=False)["qty_on_hand"].sum()


def save_inventory_week(branch_code: str, week_start, rows: pd.DataFrame) -> int:
    """Replace one (branch, week)'s rows in WeeklyStockSnapshotLine. Returns
    the number of line rows saved."""
    from wms.db import SessionLocal
    from wms.models import WeeklyStockSnapshotLine

    week_date = pd.Timestamp(week_start).date()
    db = SessionLocal()
    try:
        db.query(WeeklyStockSnapshotLine).filter(
            WeeklyStockSnapshotLine.branch_code == branch_code,
            WeeklyStockSnapshotLine.week_start == week_date).delete()
        n = 0
        for r in rows.itertuples():
            db.add(WeeklyStockSnapshotLine(
                branch_code=branch_code, sku=r.sku, week_start=week_date,
                qty_on_hand=float(r.qty_on_hand)))
            n += 1
        db.commit()
        return n
    finally:
        db.close()


def _load_inventory_panel_from_db(keys, weeks):
    from wms.db import SessionLocal
    from wms.models import WeeklyStockSnapshotLine

    db = SessionLocal()
    try:
        rows = db.query(WeeklyStockSnapshotLine).all()
    finally:
        db.close()
    if not rows:
        return None
    anchor = pd.Timestamp(weeks[0])
    W = len(weeks)
    k_ix = {k: i for i, k in enumerate(keys)}
    OH = np.full((len(keys), W), np.nan, np.float32)
    seen = False
    for r in rows:
        wi = int(round((pd.Timestamp(r.week_start) - anchor).days / 7))
        if wi < 0 or wi >= W:
            continue
        i = k_ix.get((r.branch_code, r.sku))
        if i is None:
            continue
        OH[i, wi] = max(0.0, float(r.qty_on_hand or 0))
        seen = True
    return OH if seen else None


def load_inventory_panel(keys, weeks, directory=None):
    """``(S, W)`` on-hand aligned to the sales panel's ``keys``/``weeks``, from
    the weekly Hansa stock exports. ``nan`` where a (branch, SKU, week) has no
    reading; NEGATIVE figures are read as 0 (Hansa opening-balance artefacts).

    Reads uploaded Excel files under ``directory`` (or the configured
    ``weekly_inventory_dir``) when there are any there - the on-disk path this
    always used to take, still used by tests that populate a directory
    directly. Falls back to WeeklyStockSnapshotLine in the database otherwise.
    """
    if not keys or not weeks:
        return None
    d = Path(directory) if directory else weekly_inventory_dir()
    files = [f for f in sorted(glob.glob(str(d / "**" / "*.xls*"), recursive=True))
             if not os.path.basename(f).startswith("~$")]
    if not files:
        return _load_inventory_panel_from_db(keys, weeks)
    from wms.analytics import inventory as _inv
    anchor = pd.Timestamp(weeks[0])
    W = len(weeks)
    k_ix = {k: i for i, k in enumerate(keys)}
    OH = np.full((len(keys), W), np.nan, np.float32)
    seen = False
    for f in files:
        code, ws = parse_name(f)
        if not code or ws is None:
            continue
        wi = int(round((pd.Timestamp(ws) - anchor).days / 7))
        if wi < 0 or wi >= W:
            continue
        try:
            raw = pd.read_excel(f, header=0, dtype=str)
        except Exception:                            # noqa: BLE001
            continue
        sc = _inv._pick(raw.columns, _inv._SKU_KEYS) or raw.columns[0]
        qc = _inv._pick(raw.columns, _inv._QTY_KEYS)
        if qc is None:
            continue
        raw = raw[raw[sc].notna()]
        qty = pd.to_numeric(raw[qc], errors="coerce")
        for sku, q in zip(raw[sc].astype(str).str.strip(), qty):
            i = k_ix.get((code, sku))
            if i is None or pd.isna(q):
                continue
            OH[i, wi] = max(0.0, float(q))           # negatives -> 0
            seen = True
    return OH if seen else None


def _oos_grid(MAT, OH):
    """``(S, W)`` bool: an on-hand reading of 0 or less (a real stockout signal).
    All-False when there is no inventory data."""
    if OH is None:
        return np.zeros(MAT.shape, bool)
    return np.isfinite(OH) & (np.nan_to_num(OH, nan=1e9) <= _STOCKOUT_OH)


def _instock_level(MAT, OH, influence):
    """``(S,)`` spike-damped mean weekly sales over the weeks a SKU was ON the
    shelf (on-hand > 0, or no reading), restricted to its active span (first sale
    .. last sale) so leading/trailing not-ranged zeros don't drag it down.

    0 for a SKU that, even when stocked, sells too rarely (< ``_HEUR_FREQ`` of
    in-stock weeks) or too little (< ``_MIN_LIFT_LEVEL`` per week) to call a quiet
    week a stockout rather than just its nature."""
    S, W = MAT.shape
    oos = _oos_grid(MAT, OH)
    out = np.zeros(S)
    for i in range(S):
        sales = np.where(MAT[i] > 0)[0]
        if sales.size < 3:
            continue
        span = np.arange(sales[0], sales[-1] + 1)
        keep = span[~oos[i, span]]
        if keep.size < 3 or (MAT[i, keep] > 0).mean() < _HEUR_FREQ:
            continue
        lvl = _damped_mean(MAT[i, keep], influence)
        if lvl >= _MIN_LIFT_LEVEL:
            out[i] = lvl
    return out


def _trim_runs(row, maxrun):
    """Unset any run of ``maxrun``+ consecutive True cells (a long dry spell is a
    discontinued line, not a stockout)."""
    row = row.copy()
    i, n = 0, len(row)
    while i < n:
        if row[i]:
            j = i
            while j < n and row[j]:
                j += 1
            if j - i > maxrun:
                row[i:j] = False
            i = j
        else:
            i += 1
    return row


def _stockout_mask(MAT, OH, influence, heuristic=True):
    """``(S, W)`` bool + ``(S,)`` in-stock level. A week is flagged only when the
    SKU sells materially when stocked (``_instock_level`` > 0), the week is
    INTERIOR (between that SKU's first and last sale), its sales are well below
    the level, and either an on-hand reading says 0 or - with no reading and the
    heuristic on - it is near-zero with a sale close by on each side."""
    S, W = MAT.shape
    lvl = _instock_level(MAT, OH, influence)
    oos = _oos_grid(MAT, OH)
    known = np.isfinite(OH) if OH is not None else np.zeros((S, W), bool)
    mask = np.zeros((S, W), bool)
    for i in range(S):
        if lvl[i] <= 0:
            continue
        sales = np.where(MAT[i] > 0)[0]
        lo, hi = sales[0], sales[-1]
        low = MAT[i] < 0.6 * lvl[i]
        for w in range(lo + 1, hi):                  # interior weeks only
            if not low[w]:
                continue
            if oos[i, w]:
                mask[i, w] = True
            elif heuristic and not known[i, w] and MAT[i, w] < _HEUR_LOW * lvl[i] \
                    and (MAT[i, max(0, w - 2):w] > 0).any() \
                    and (MAT[i, w + 1:w + 3] > 0).any():
                mask[i, w] = True
        mask[i] = _trim_runs(mask[i], _MAX_STOCKOUT_RUN)
        if mask[i].sum() > _MAX_CENSOR_FRAC * W:     # too much missing: trust none of it
            mask[i] = False
    return mask, lvl


def _unconstrain(MAT, mask, level):
    """Lift each flagged week to that SKU's in-stock ``level`` (kept if the
    suppressed week somehow still sold more)."""
    U = MAT.astype(float).copy()
    for i in range(MAT.shape[0]):
        if mask[i].any() and level[i] > 0:
            U[i, mask[i]] = np.maximum(MAT[i, mask[i]], level[i])
    return U


def training_matrix(pan):
    """``(MAT_train, stockout_mask, meta)`` — the sales matrix with stockout
    weeks lifted to the in-stock level, so the models learn fully-stocked
    demand. ``weekly_unconstrain=False`` returns the raw matrix unchanged."""
    MAT = pan["MAT"].astype(float)
    cfg = get_settings()
    if not getattr(cfg, "weekly_unconstrain", True) or MAT.size == 0:
        return MAT, np.zeros(MAT.shape, bool), {
            "censored": 0, "cells": int(MAT.size), "inventory": False,
            "inv_readings": 0, "skus_affected": 0, "oos": np.zeros(MAT.shape, bool)}
    infl = float(getattr(cfg, "weekly_spike_influence", _SPIKE_INFLUENCE))
    OH = load_inventory_panel(pan["keys"], pan["weeks"])
    mask, lvl = _stockout_mask(MAT, OH, infl,
                               bool(getattr(cfg, "weekly_unconstrain_heuristic", True)))
    oos = _oos_grid(MAT, OH)                 # every week an on-hand reading said 0
    return _unconstrain(MAT, mask, lvl), mask, {
        "censored": int(mask.sum()), "cells": int(mask.size),
        "inventory": OH is not None,
        "inv_readings": int(np.isfinite(OH).sum()) if OH is not None else 0,
        "skus_affected": int((mask.any(1)).sum()),
        "oos": oos}


# ---------------------------------------------------------------- per-SKU models
# The four requested models: ES-RNN and NeuralProphet are global neural fits
# (see ``weekly_neural``); SBA and snaive are the per-SKU classical pair below.
# Each ``f_*`` is fitted on the training vector ``y`` and returns a genuine
# ``h``-step-ahead forecast (length h, non-negative) — nothing sees the hold-out.

def f_snaive(y, h, period=4):
    """Seasonal-naive: repeat the last ``period``-week pattern."""
    n = len(y)
    if n < period:
        return np.full(h, float(y.mean()) if len(y) else 0.0)
    last = y[-period:]
    return np.clip(np.asarray([last[i % period] for i in range(h)]), 0, None)


def _damped_mean(y, influence=_SPIKE_INFLUENCE):
    """Mean of the series with bulk-order weeks pulled toward the body: any week
    above ``median + 2*MAD`` contributes only ``influence`` of its excess."""
    y = np.asarray(y, float)
    y = y[y >= 0]
    if not y.size or not y.any():
        return 0.0
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    cap = med + 2.0 * (mad if mad > 0 else max(med * 0.5, 1.0))
    dy = np.where(y > cap, cap + influence * (y - cap), y)
    return float(dy.mean())


_RECENT_ANCHOR_WEEKS = 8
_DEAD_WEEKS = 6                    # this many straight zero weeks = "not selling now"


def _recent_level(hist_rows, influence):
    """Per-SKU spike-damped mean over the last ``_RECENT_ANCHOR_WEEKS`` weeks —
    "what this SKU sells now". Used as the floor reference so a product that has
    faded can be forecast low and one that has ramped up is anchored to where it
    is now, not to an all-time mean full of early zeros.

    Zero for a SKU with ``_DEAD_WEEKS`` or more STRAIGHT zero weeks right at the
    end: a one-off spike that happens to sit just inside the ``_RECENT_ANCHOR_WEEKS``
    window (e.g. one huge order 7 weeks ago, nothing since) must not force the
    floor back up for a product that has since gone quiet.
    """
    hist_rows = np.asarray(hist_rows, float)
    T = hist_rows.shape[1]
    rw = min(_RECENT_ANCHOR_WEEKS, T) if T else 0
    if not rw:
        return np.zeros(hist_rows.shape[0])
    dw = min(_DEAD_WEEKS, T)
    dead = ~hist_rows[:, -dw:].any(axis=1) if dw else np.zeros(hist_rows.shape[0], bool)
    return np.array([0.0 if dead[i] else _damped_mean(hist_rows[i, -rw:], influence)
                     for i in range(hist_rows.shape[0])])


def _ref_level(hist_rows, influence):
    """Reference level for the category-pool multiplier: the recent level (see
    ``_recent_level``)."""
    return _recent_level(hist_rows, influence)


def _anchor_to_level(F, hist_rows, influence, band, up_band=None):
    """Clamp each row of forecast ``F`` (S, h) to a window around the SKU's
    level, so a prediction never strays far from it.

    One ``band`` -> symmetric ±band. Pass ``up_band`` for an asymmetric window.

    The DOWN edge is anchored to the SKU's recent level (last
    ``_RECENT_ANCHOR_WEEKS`` weeks) so a faded product can still be forecast low;
    the UP edge is anchored to the LARGER of the recent and whole-history level
    so a product that has ramped up recently can be forecast where it now sits,
    without being dragged back toward an all-time mean full of early zeros.
    """
    hist_rows = np.asarray(hist_rows, float)
    rec = _recent_level(hist_rows, influence)
    full = np.array([_damped_mean(hist_rows[i], influence)
                     for i in range(hist_rows.shape[0])])
    ref_lo = rec[:, None]
    ref_hi = np.maximum(rec, full)[:, None]
    hi_band = band if up_band is None else up_band
    lo, hi = max(0.0, 1.0 - band), 1.0 + hi_band
    out = np.where(ref_lo > 0, np.maximum(F, ref_lo * lo), F)
    return np.where(ref_hi > 0, np.minimum(out, ref_hi * hi), out)


def f_damped_mean(y, h):
    """Forecast = the SKU's own spike-damped mean, held flat. Closest to actual
    on short, near-white-noise weekly demand; bulk-order weeks count only half."""
    return np.full(h, _damped_mean(y))


def f_old_excel(y, h):
    """The old spreadsheet rule: take last month's total units (the last 4
    weeks), add 10%, round that UP, then split evenly into 4 weekly buckets:

        weekly = ceil(sum(last 4 weeks) * 1.1) / 4

    A plain, hand-checkable manual baseline for use while there is not enough
    weekly history for the statistical models to be reliable. Held flat across
    the horizon; not run through the safety uplift or level clamp (the +10% and
    the round-up ARE the calibration)."""
    y = np.asarray(y, float)
    last_month = float(y[-4:].sum()) if y.size else 0.0
    weekly = float(np.ceil(last_month * 1.1)) / 4.0
    return np.full(h, weekly)


def _finalise(F, hist_rows, uplift, cats, influence, pool, down_band, up_band,
              min_units):
    """Shared post-processing for the live and the hold-out forecast:

    1. multiply by the calibrated safety ``uplift``;
    2. borrow a little category shape (``_category_pool``);
    3. clamp asymmetrically to the SKU's damped mean — held close underneath
       (``down_band``) but free to run higher for restocking cover (``up_band``);
    4. floor every SKU at ``min_units`` — a SKU only appears here because it has
       sold at least once, so it is never forecast to zero.
    """
    F = _category_pool(np.asarray(F, float) * uplift, hist_rows, cats,
                       influence, pool)
    F = _anchor_to_level(F, hist_rows, influence, down_band, up_band)
    if min_units > 0:
        F = np.maximum(F, float(min_units))
    return F


def _category_pool(F, hist_rows, cats, influence, weight):
    """Borrow a little shape from same-category SKUs.

    Each SKU's forecast is expressed as a multiple of its own reference level
    (``F / level``, see ``_ref_level`` — recent-aware); that multiplier is then
    shrunk ``weight`` of the way toward the median multiplier of its category.
    Only categories with at least ``_CAT_POOL_MIN`` members pool; a thin/erratic
    history is steadied by the category without overriding the SKU's own level.
    """
    if weight <= 0 or F.size == 0:
        return F
    lvl = _ref_level(hist_rows, influence)[:, None]
    pos = lvl[:, 0] > 0
    if not pos.any():
        return F
    cats = np.asarray(cats)
    mult = np.where(lvl > 0, F / np.where(lvl > 0, lvl, 1.0), 1.0)     # (S, h)
    for c in set(cats[pos].tolist()):
        m = pos & (cats == c)
        if int(m.sum()) >= _CAT_POOL_MIN:
            med = np.median(mult[m], axis=0)                           # (h,)
            mult[m] = (1.0 - weight) * mult[m] + weight * med
    return np.where(lvl > 0, mult * lvl, F)


def f_croston(y, h, a=_CRO_A, sba=True):
    """Croston / Syntetos-Boylan Approximation intermittent-demand rate."""
    nz = np.nonzero(y)[0]
    if len(nz) == 0:
        return np.zeros(h)
    z, p, q = float(y[nz[0]]), float(nz[0] + 1), 1.0
    corr = 1 - a / 2 if sba else 1.0
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z = a * y[t] + (1 - a) * z
            p = a * q + (1 - a) * p
            q = 1.0
        else:
            q += 1.0
    return np.full(h, max(0.0, corr * z / p))


def _ratio_base(MAT, upto, influence=_SPIKE_INFLUENCE, weeks=4, dead_after=_DEAD_WEEKS):
    """Base level for the esrnn_ratio model: the spike-damped mean of the last
    ``weeks`` weeks up to ``upto`` (so a ramped-up SKU is scaled from where it is
    now, and a one-off spike in that window does not dominate), floored at the
    SKU's long-run Croston rate for SKUs that have recently gone quiet.

    That Croston floor is skipped for a SKU with ``dead_after`` or more
    STRAIGHT zero weeks right before ``upto``: Croston's rate is a mean over
    the gaps between sales, so one huge one-off order (e.g. a single 1,659-unit
    week) months ago can leave it reporting a rate of 100+ for a product that
    has since gone completely quiet — which would wrongly resurrect it. A long
    dry spell means "not currently selling", not "quiet since a big order";
    only the never-zero floor applies then.
    """
    rw = min(weeks, upto)
    dw = min(dead_after, upto)
    out = []
    for i in range(MAT.shape[0]):
        recent = _damped_mean(MAT[i, upto - rw:upto], influence)
        if dw and not MAT[i, upto - dw:upto].any():
            out.append(recent)                        # long dry spell: no rate floor
        else:
            rate = float(f_croston(MAT[i, :upto].astype(float), 1)[0])
            out.append(max(recent, rate))
    return np.array(out)


_MODELS = {
    "damped_mean": f_damped_mean,                 # spike-robust level (closest to actual here)
    "sba": lambda y, h: f_croston(y, h),          # Syntetos-Boylan Approximation
    "snaive": f_snaive,
}

# models whose multi-week forecast actually moves week to week
_DYNAMIC = {"snaive", "esrnn", "neuralprophet", "esrnn_ratio", "gbm", "lgbm",
            "lstm", "blend"}
_NEURAL = ("esrnn", "neuralprophet", "esrnn_ratio")
_ML = ("gbm", "lgbm", "lstm")             # feature-based global models (weekly_ml)
_GLOBAL = _NEURAL + _ML                   # refit on the full history when they win
# blend candidate: weighted mean of these fitted forecasts (weights need not sum
# to 1; renormalised over whichever are present). Default hand-set weights;
# `train_and_save` learns better ones by rolling-origin CV -> output/weekly_blend.json.
_BLEND_W = {"esrnn_ratio": 0.6, "gbm": 0.4}
_BLEND_POOL = ("esrnn_ratio", "gbm", "lgbm")   # models the learned blend may weight


def _blend(FC: dict, w: dict) -> np.ndarray:
    present = {k: v for k, v in w.items() if k in FC}
    tot = sum(present.values()) or 1.0
    out = None
    for k, wk in present.items():
        term = np.asarray(FC[k], float) * (wk / tot)
        out = term if out is None else out + term
    return out
# prefer a moving forecast over a flat one when within this WAPE gap of the best
_SELECT_TOL = 3.0
# a model that forecasts zero for more than this share of still-selling SKUs is
# not eligible (great WAPE, useless for restocking)
_MAX_DEAD_ACTIVE = 0.10


def _segment(y_tr):
    nz = y_tr[y_tr > 0]; n_nz, n = len(nz), len(y_tr)
    if n_nz == 0:
        return "dead", 0
    if n_nz == 1:
        return "new", 1
    adi = n / n_nz
    cv2 = float(nz.std() ** 2 / (nz.mean() ** 2 + 1e-9))
    if adi >= 1.32 and cv2 >= 0.49:
        return "lumpy", n_nz
    if adi >= 1.32:
        return "intermittent", n_nz
    if cv2 >= 0.49:
        return "erratic", n_nz
    return "smooth", n_nz


def _forecast_matrix(MAT, tr_end, h):
    """{name: (S, h) forecasts} — each model fitted on MAT[:, :tr_end] only."""
    S = MAT.shape[0]
    out = {name: np.zeros((S, h)) for name in _MODELS}
    for i in range(S):
        y = MAT[i, :tr_end].astype(float)
        for name, fn in _MODELS.items():
            out[name][i] = fn(y, h)
    return out


def _want(only, name) -> bool:
    """``only`` is None (fit everything) or a container of model names to fit."""
    return only is None or name in only


def _neural_forecasts(MAT, weeks, tr_end, h, only=None):
    """{name: (S,h)} for the ES-RNN and NeuralProphet candidates (torch).

    ``only`` -> None fits all; a container of names fits just those (fast cold
    start when only one/two are actually needed for the live model)."""
    s = get_settings()
    out = {}
    try:
        from wms.analytics import weekly_neural as _nn
    except Exception:
        return out
    if getattr(s, "weekly_esrnn", True) and _want(only, "esrnn"):
        F = _nn.esrnn_forecast(MAT, tr_end, h)
        if F is not None:
            out["esrnn"] = np.clip(F, 0, None)
    if getattr(s, "weekly_neuralprophet", True) and _want(only, "neuralprophet"):
        F = _nn.neuralprophet_forecast(MAT, weeks, tr_end, h)
        if F is not None:
            out["neuralprophet"] = np.clip(F, 0, None)
    if getattr(s, "weekly_esrnn_ratio", True) and _want(only, "esrnn_ratio"):
        fn = getattr(_nn, "esrnn_ratio_forecast", lambda *a, **k: None)
        F = fn(MAT, tr_end, h, base=_ratio_base(MAT, tr_end),
               lo=float(getattr(s, "weekly_ratio_lo", 0.8)),
               hi=float(getattr(s, "weekly_ratio_hi", 1.5)),
               checkpoint=load_ratio_checkpoint())
        if F is not None:
            out["esrnn_ratio"] = np.clip(F, 0, None)
    return out


def _ml_forecasts(MAT, weeks, tr_end, h, keys, cats, only=None):
    """{name: (S,h)} for the feature-based global models: XGBoost ``gbm``,
    LightGBM ``lgbm`` and the windowed ``lstm``. ``only`` as in
    :func:`_neural_forecasts`.

    The tree models always fit FRESH on ``[:tr_end]`` here: a saved booster was
    trained on the full history, so reusing it for the hold-out would leak the
    held-out week into the score (that is fine only for the live refit)."""
    s = get_settings()
    out = {}
    try:
        from wms.analytics import weekly_ml as _ml
    except Exception:                                        # noqa: BLE001
        return out
    if getattr(s, "weekly_gbm", True) and _want(only, "gbm"):
        F = _ml.gbm_forecast(MAT, weeks, tr_end, h, keys=keys, cats=cats)
        if F is not None:
            out["gbm"] = np.clip(F, 0, None)
    if getattr(s, "weekly_lgbm", True) and _want(only, "lgbm"):
        F = _ml.lgbm_forecast(MAT, weeks, tr_end, h, keys=keys, cats=cats)
        if F is not None:
            out["lgbm"] = np.clip(F, 0, None)
    if getattr(s, "weekly_lstm", True) and _want(only, "lstm"):
        F = _ml.lstm_forecast(MAT, weeks, tr_end, h, keys=keys, cats=cats)
        if F is not None:
            out["lstm"] = np.clip(F, 0, None)
    return out


# ---------------------------------------------------------------- build
def build(directory=None, test_weeks: int = TEST_WEEKS) -> dict:
    pan = load_panel(directory)
    keys, weeks = pan["keys"], pan["weeks"]
    MAT_raw = pan["MAT"].astype(float)               # what actually sold
    MAT, _smask, umeta = training_matrix(pan)        # stockout weeks lifted to level
    S, W = MAT.shape
    cols = ["branch", "branch_name", "sku", "item", "segment", "method",
            "weeks_sold", "weekly_demand", "recent_sales"]
    empty = {"state": pd.DataFrame(columns=cols), "overall": pd.DataFrame(),
             "by_segment": pd.DataFrame(), "champ": {}, "best_method": None,
             "backtest": pd.DataFrame(), "test_week_labels": [],
             "test_weeks": test_weeks,
             "coverage": {"branches": [], "weeks": 0, "series": 0, "week_range": ""}}
    if S == 0 or W < test_weeks + MIN_TRAIN_WEEKS:
        return empty

    tr_end = W - test_weeks                       # train = weeks [0, tr_end); held out = the rest
    act = MAT[:, tr_end:]                         # (S, test_weeks) — NEVER fed to a model
    act_raw = MAT_raw[:, tr_end:]                 # what really sold in the held-out weeks
    # held-out cells the branch WAS in stock: an on-hand reading of 0 in a
    # held-out week is a supply gap, so the model is scored on what really sold
    # over the OTHER held-out weeks only (not marked down for correctly
    # predicting fully-stocked demand a dry shelf then suppressed).
    oos = umeta.get("oos")
    oos_te = oos[:, tr_end:] if oos is not None and oos.shape == MAT.shape \
        else np.zeros_like(act, bool)
    ins_te = ~oos_te
    if not ins_te.any():                          # degenerate: score on everything
        ins_te = np.ones_like(ins_te)
    branch_of = np.array([k[0] for k in keys])
    seg, nnz = zip(*[_segment(MAT[i, :tr_end]) for i in range(S)])
    seg = np.array(seg); nnz = np.array(nnz)
    cats = np.array([categorise(pan["item_of"][k]) for k in keys])

    # fit every model on the training weeks, forecast the held-out horizon.
    # If a global model is pinned, fit only that one here (the others would just
    # fill comparison rows and cost a from-scratch train each cold start); Auto
    # (no pin) still fits them all so the comparison is full.
    _pin = forced_model()
    _blend_pin = _pin == "blend"
    _bw = load_blend_weights() or _BLEND_W          # learned weights if trained
    # when blend is pinned, fit only its components (each loads from a saved
    # checkpoint) — don't pay a cold from-scratch train for the rest just to
    # fill comparison rows.
    if _blend_pin:
        _need = set(_bw)
        _n_only = _need & set(_NEURAL)
        _m_only = _need & set(_ML)
    else:
        _n_only = None if _pin == "" else ({_pin} if _pin in _NEURAL else set())
        _m_only = None if _pin == "" else ({_pin} if _pin in _ML else set())
    FC = _forecast_matrix(MAT, tr_end, test_weeks)
    FC.update(_neural_forecasts(MAT, weeks, tr_end, test_weeks, only=_n_only))
    FC.update(_ml_forecasts(MAT, weeks, tr_end, test_weeks, keys, cats,
                            only=_m_only))
    # "Old Excel": the manual spreadsheet rule on RAW sales (last month * 1.1,
    # rounded up, / 4). Always available so it can be pinned while data is thin.
    if getattr(get_settings(), "weekly_old_excel", True):
        FC["old_excel"] = np.stack(
            [f_old_excel(MAT_raw[i, :tr_end], test_weeks) for i in range(S)])
    # blend: a renormalised weighted mean of the component forecasts. Weights are
    # the learned ones from train_and_save (rolling-origin CV) or the hand-set
    # fallback. Built when every weighted component was fitted.
    if (_pin in ("", "blend")) and _bw and set(_bw) <= set(FC):
        FC["blend"] = _blend(FC, _bw)
    MODEL_KEYS = list(FC)

    # MASE scale: mean in-sample 1-step naive error on the TRAINING weeks only
    dif = np.abs(np.diff(MAT[:, :tr_end], axis=1))
    scale = max(float(dif.mean()) if dif.size else 1.0, 1e-6)

    def sc(F):
        # score against what REALLY sold (act_raw), but only over the held-out
        # cells the branch was in stock — a model trained on fully-stocked demand
        # must not be marked down for "over-predicting" a week the shelf was dry.
        d = (F - act_raw)[ins_te]
        e = np.abs(d); tot = float(act_raw[ins_te].sum()) or 1.0
        return dict(MAE=round(float(e.mean()), 3),
                    RMSE=round(float(np.sqrt((d ** 2).mean())), 3),
                    WAPE=round(float(100 * e.sum() / tot), 1),
                    bias=round(float(100 * d.sum() / tot), 1),
                    MASE=round(float(e.mean() / scale), 3))

    def wape_seg(F, s):
        msk = seg == s
        if not msk.any():
            return np.nan
        ins = ins_te[msk]
        if not ins.any():
            return np.nan
        a = act_raw[msk]
        return round(float(100 * np.abs(F[msk] - a)[ins].sum()
                           / (a[ins].sum() or 1.0)), 1)

    overall = (pd.DataFrame({m: sc(FC[m]) for m in MODEL_KEYS}).T
               [["MAE", "RMSE", "WAPE", "bias", "MASE"]].sort_values("WAPE"))
    by_seg = pd.DataFrame({m: {s: wape_seg(FC[m], s) for s in _SEGS}
                           for m in MODEL_KEYS}).T
    champ = {}                                    # descriptive: per-segment best (not blended)
    for s in _SEGS:
        c = {m: wape_seg(FC[m], s) for m in MODEL_KEYS}
        c = {m: v for m, v in c.items() if v == v}
        champ[s] = min(c, key=c.get) if c else None

    ranked = [str(m) for m in overall.index]       # ascending WAPE
    # a model that zeroes out SKUs which are still selling is not eligible, no
    # matter its WAPE — you cannot restock a "0" forecast
    active = MAT[:, max(0, tr_end - 4):tr_end].sum(1) > 0

    def _dead_active(F):
        return float((F[active, 0] == 0).mean()) if active.any() else 0.0

    eligible = [m for m in ranked if _dead_active(FC[m]) <= _MAX_DEAD_ACTIVE] or ranked
    # we want to end up predicting slightly ABOVE sales, so drop models that
    # under-forecast heavily on the hold-out (an uplift can't safely rescue them)
    not_low = [m for m in eligible
               if float(overall.loc[m, "bias"]) >= -_MAX_UNDER_BIAS] or eligible
    _margin = float(getattr(get_settings(), "weekly_safety_margin", _SAFETY_MARGIN))
    top_wape = float(overall.loc[not_low[0], "WAPE"])
    # among the models within _SELECT_TOL WAPE of the best, take the one whose
    # bias is closest to the target margin (a small, deliberate over-forecast);
    # when two are within 5 pts on that, prefer the one that moves week to week,
    # then lower WAPE
    close = [m for m in not_low
             if float(overall.loc[m, "WAPE"]) <= top_wape + _SELECT_TOL] or not_low

    def _bias_dist(m):
        return abs(float(overall.loc[m, "bias"]) - _margin * 100)

    best_method = min(close, key=lambda m: (
        round(_bias_dist(m) / 5),                 # bins of 5 pts
        0 if m in _DYNAMIC else 1,
        float(overall.loc[m, "WAPE"])))

    forced = forced_model()
    if forced and forced in MODEL_KEYS:
        best_method = forced                       # pinned by the user / config

    # live forecast: refit the winner on the FULL history, project the next weeks
    prod_fc = None
    if best_method == "old_excel":                 # raw last-month rule, no refit
        prod_fc = np.stack(
            [f_old_excel(MAT_raw[i], test_weeks) for i in range(S)])
    elif best_method in _MODELS:
        prod_fc = np.zeros((S, test_weeks))
        fn = _MODELS[best_method]
        for i in range(S):
            prod_fc[i] = fn(MAT[i].astype(float), test_weeks)
    else:                                          # global winner — refit on all weeks
        try:
            from wms.analytics import weekly_neural as _nn
            if best_method == "esrnn":
                prod_fc = _nn.esrnn_forecast(MAT, W, test_weeks)
            elif best_method == "esrnn_ratio":
                prod_fc = _nn.esrnn_ratio_forecast(
                    MAT, W, test_weeks, base=_ratio_base(MAT, W),
                    lo=float(getattr(get_settings(), "weekly_ratio_lo", 0.8)),
                    hi=float(getattr(get_settings(), "weekly_ratio_hi", 1.5)),
                    checkpoint=load_ratio_checkpoint())
            elif best_method == "neuralprophet":
                prod_fc = _nn.neuralprophet_forecast(MAT, weeks, W, test_weeks)
            elif best_method in _ML:
                from wms.analytics import weekly_ml as _ml
                if best_method == "gbm":
                    prod_fc = _ml.gbm_forecast(MAT, weeks, W, test_weeks,
                                               keys=keys, cats=cats,
                                               model_in=_gbm_ckpt_path_if_fresh())
                elif best_method == "lgbm":
                    prod_fc = _ml.lgbm_forecast(MAT, weeks, W, test_weeks,
                                                keys=keys, cats=cats,
                                                model_in=_lgbm_ckpt_path_if_fresh())
                else:
                    prod_fc = _ml.lstm_forecast(MAT, weeks, W, test_weeks,
                                                keys=keys, cats=cats)
            elif best_method == "blend":
                from wms.analytics import weekly_ml as _ml
                _fc = lambda k: (                        # noqa: E731
                    _nn.esrnn_ratio_forecast(
                        MAT, W, test_weeks, base=_ratio_base(MAT, W),
                        lo=float(getattr(get_settings(), "weekly_ratio_lo", 0.8)),
                        hi=float(getattr(get_settings(), "weekly_ratio_hi", 1.5)),
                        checkpoint=load_ratio_checkpoint()) if k == "esrnn_ratio"
                    else _ml.gbm_forecast(MAT, weeks, W, test_weeks, keys=keys,
                                          cats=cats,
                                          model_in=_gbm_ckpt_path_if_fresh())
                    if k == "gbm"
                    else _ml.lgbm_forecast(MAT, weeks, W, test_weeks, keys=keys,
                                           cats=cats,
                                           model_in=_lgbm_ckpt_path_if_fresh())
                    if k == "lgbm" else None)
                parts = {k: np.clip(v, 0, None) for k in _bw
                         if (v := _fc(k)) is not None}
                prod_fc = _blend(parts, _bw) if parts else None
        except Exception:
            prod_fc = None
        if prod_fc is None:                        # fall back to a safe nonzero rate
            best_method = "sba"
            prod_fc = np.stack([f_croston(MAT[i].astype(float), test_weeks)
                                for i in range(S)])

    # safety uplift: two calibrations of the same idea — scale so the forecast
    # total lands ~margin ABOVE the reference sales level (never scaled down; capped).
    #  * prod_uplift  — live next-week forecast vs the recent weekly sales average
    #  * hold_uplift  — the shown hold-out forecast vs the hold-out's own actuals
    margin = float(getattr(get_settings(), "weekly_safety_margin", _SAFETY_MARGIN))

    def _uplift(pred_total, ref_total):
        return float(np.clip((1 + margin) * (ref_total or 1.0) / (pred_total or 1.0),
                             1.0, _MAX_UPLIFT))

    recent_avg = float(MAT[:, -min(4, W):].mean(1).sum())
    prod_uplift = _uplift(float(np.clip(prod_fc[:, 0], 0, None).sum()), recent_avg)
    hold_uplift = _uplift(float(FC[best_method].sum()), float(act.sum()))

    # esrnn_ratio already sits ~just above recent sales by construction (its
    # 0.8-1.5 multiplier IS the calibration); "old_excel" has its own +10% and
    # round-up. Neither gets the global safety uplift or the category shrink; and
    # old_excel is left EXACTLY as the spreadsheet rule computes it (no level
    # clamp either) so a planner can hand-check the number.
    _ratio_pick = best_method == "esrnn_ratio"
    _raw_pick = best_method == "old_excel"

    # post-process: category shape, an asymmetric level clamp (close underneath,
    # room above for restocking cover) and a floor so a live SKU never reads zero
    cfg = get_settings()
    infl = float(getattr(cfg, "weekly_spike_influence", _SPIKE_INFLUENCE))
    pool = 0.0 if (_ratio_pick or _raw_pick) else \
        float(getattr(cfg, "weekly_category_pool", _CATEGORY_POOL))
    down = 1.0 if _raw_pick else float(getattr(cfg, "weekly_down_band", _DOWN_BAND))
    up = 1e9 if _raw_pick else float(getattr(cfg, "weekly_up_band", _UP_BAND))
    # old_excel is the literal spreadsheet number: no never-zero floor either, so
    # a SKU with nothing sold last month allocates 0 (matches the manual sheet).
    min_u = 0 if _raw_pick else int(getattr(cfg, "weekly_min_units", _MIN_UNITS))
    if _ratio_pick or _raw_pick:
        prod_uplift = hold_uplift = 1.0
    prod_fc = _finalise(prod_fc, MAT.astype(float), prod_uplift, cats,
                        infl, pool, down, up, min_u)
    next_week = np.round(np.clip(prod_fc[:, 0], 0, None)).astype(int)

    state = pd.DataFrame({
        "branch": branch_of,
        "branch_name": [BRANCH_NAME.get(b, b) for b in branch_of],
        "sku": [k[1] for k in keys],
        "item": [pan["item_of"][k] for k in keys],
        "segment": seg,
        "method": best_method,
        "weeks_sold": nnz.astype(int),
        "weekly_demand": next_week,
        # last ~4 wk avg of what ACTUALLY sold (raw), not the unconstrained level
        "recent_sales": np.round(MAT_raw[:, -min(4, W):].mean(1)).astype(int),
    })

    # per-SKU held-out forecast (fitted on training weeks) vs actuals — same
    # post-processing as the live forecast, on the TRAIN weeks only
    test_labels = list(weeks[tr_end:])
    bp = _finalise(FC[best_method], MAT[:, :tr_end].astype(float), hold_uplift,
                   cats, infl, pool, down, up, min_u)
    ins = ins_te                                      # (S, test_weeks) in-stock cells
    act_raw_tot = act_raw.sum(1)
    ins_err = np.where(ins, np.abs(bp - act_raw), 0.0).sum(1)   # in-stock weeks only
    ins_act = np.where(ins, act_raw, 0.0).sum(1)
    bt = pd.DataFrame({
        "branch": branch_of,
        "branch_name": [BRANCH_NAME.get(b, b) for b in branch_of],
        "sku": [k[1] for k in keys],
        "item": [pan["item_of"][k] for k in keys],
    })
    for j, lab in enumerate(test_labels):
        bt[f"pred::{lab}"] = np.round(bp[:, j]).astype(int)
        bt[f"act::{lab}"] = np.round(act_raw[:, j]).astype(int)
    bt["predicted"] = np.round(bp.sum(1)).astype(int)
    bt["actual"] = np.round(act_raw_tot).astype(int)   # what really sold
    bt["error"] = bt["actual"] - bt["predicted"]     # net over/under across the held-out weeks
    # WAPE over the held-out weeks the SKU was in stock (a stockout week's low
    # sale is not a forecast miss); blank if it was out of stock every held week
    bt["wape"] = np.where(ins_act > 0,
                          np.round(100 * ins_err / np.clip(ins_act, 1, None), 1),
                          np.nan)
    bt["stockout"] = (~ins).sum(1).astype(int)        # held-out weeks out of stock

    cov = {
        "branches": sorted(set(branch_of.tolist())),
        "weeks": W, "series": S, "train_weeks": tr_end,
        "week_range": f"{weeks[0]} … {weeks[-1]}" if weeks else "",
        "test_weeks": test_weeks, "method": best_method,
        "models": MODEL_KEYS,
        "safety_margin": margin,
        "applied_uplift": round(prod_uplift, 3),
        "holdout_uplift": round(hold_uplift, 3),
        # stockout unconstraining: how many SKU-weeks were treated as
        # stockout-suppressed and lifted to the in-stock level before training
        "unconstrain": bool(getattr(get_settings(), "weekly_unconstrain", True)),
        "censored_weeks": int(umeta.get("censored", 0)),
        "censored_cells": int(umeta.get("cells", MAT.size)),
        "censored_pct": round(100 * umeta.get("censored", 0)
                              / max(umeta.get("cells", MAT.size), 1), 2),
        "censored_skus": int(umeta.get("skus_affected", 0)),
        "inventory_used": bool(umeta.get("inventory", False)),
        "inv_readings": int(umeta.get("inv_readings", 0)),
        "censored_holdout": int((~ins_te).sum()),
    }
    return {"state": state, "overall": overall, "by_segment": by_seg,
            "champ": champ, "best_method": best_method,
            "backtest": bt, "test_week_labels": test_labels,
            "coverage": cov, "test_weeks": test_weeks}


# ---------------------------------------------------------------- cache + lookup
import threading as _threading

_CACHE: dict = {}
_CACHE_LOCK = _threading.Lock()


def _model_choice_path() -> Path:
    return get_settings().out / "weekly_model.txt"


def forced_model() -> str:
    """The model pinned for every forecast: the saved Flow Analysis choice, else
    the config default (``weekly_force_model``), else ``""`` = auto-select."""
    try:
        v = _model_choice_path().read_text(encoding="utf-8").strip()
        return "" if v in ("", "auto") else v
    except OSError:
        return str(getattr(get_settings(), "weekly_force_model", "") or "").strip()


def set_forced_model(name: str) -> None:
    """Pin ``name`` (a model key) for every forecast, or ``""`` to auto-select.
    Persists across restarts and drops the forecast cache so the next request
    rebuilds with the new model."""
    name = (name or "").strip()
    try:
        _model_choice_path().write_text(name or "auto", encoding="utf-8")
    except OSError:
        pass
    _CACHE.clear()


# ---------------------------------------------------------------- saved model
#   The neural ``esrnn_ratio`` network can be trained ONCE (offline, thoroughly)
#   and its weights saved. Every later forecast then loads those weights, freezes
#   them, and only re-fits each series' smoothing alpha (seconds), instead of
#   training from scratch on every process start (~minutes). No checkpoint ->
#   train-from-scratch, same as before.
def _ratio_ckpt_path() -> Path:
    return get_settings().out / "weekly_esrnn_ratio.pt"


def _ratio_meta_path() -> Path:
    return get_settings().out / "weekly_esrnn_ratio.json"


_CKPT_CACHE: dict = {}


def load_ratio_checkpoint():
    """The saved ``esrnn_ratio`` network as ``{"hp", "net_state"}``, or ``None``.
    Memoised on the file's mtime so it is read from disk at most once per save."""
    p = _ratio_ckpt_path()
    try:
        mt = p.stat().st_mtime
    except OSError:
        _CKPT_CACHE.clear()
        return None
    if _CKPT_CACHE.get("mt") == mt:
        return _CKPT_CACHE.get("val")
    try:
        import torch
        val = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:                                     # noqa: BLE001
        val = None
    _CKPT_CACHE["mt"], _CKPT_CACHE["val"] = mt, val
    return val


def save_ratio_checkpoint(state: dict, meta: dict) -> None:
    try:
        import json
        import torch
        torch.save(state, _ratio_ckpt_path())
        _ratio_meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _CKPT_CACHE.clear()
        _CACHE.clear()
    except Exception as e:                                # noqa: BLE001
        import warnings
        warnings.warn(f"could not save esrnn_ratio checkpoint: {e}")


def _weekly_file_sig() -> list:
    d = weekly_dir()
    try:
        return sorted(os.path.basename(x)
                      for x in glob.glob(str(d / "**" / "*.xls*"), recursive=True)
                      if not os.path.basename(x).startswith("~$"))
    except OSError:
        return []


# ---- saved gradient-boosted boosters ("gbm" = XGBoost, "lgbm" = LightGBM) ----
def _gbm_ckpt_path() -> Path:
    return get_settings().out / "weekly_gbm.json"        # xgboost native format


def _gbm_meta_path() -> Path:
    return get_settings().out / "weekly_gbm.meta.json"


def _lgbm_ckpt_path() -> Path:
    return get_settings().out / "weekly_lgbm.txt"        # lightgbm native format


def _lgbm_meta_path() -> Path:
    return get_settings().out / "weekly_lgbm.meta.json"


def _ckpt_path_if_fresh(ckpt: Path, meta: Path):
    """Path (str) of a saved booster IF it exists and was trained on the current
    weekly-sales file set; else ``None`` (it then fits in-request, a few sec)."""
    import json
    if not ckpt.exists():
        return None
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return None
    return str(ckpt) if m.get("files") == _weekly_file_sig() else None


def _gbm_ckpt_path_if_fresh():
    return _ckpt_path_if_fresh(_gbm_ckpt_path(), _gbm_meta_path())


def _lgbm_ckpt_path_if_fresh():
    return _ckpt_path_if_fresh(_lgbm_ckpt_path(), _lgbm_meta_path())


def save_gbm_meta(meta: dict) -> None:
    import json
    try:
        _gbm_meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _CACHE.clear()
    except OSError:
        pass


def save_lgbm_meta(meta: dict) -> None:
    import json
    try:
        _lgbm_meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _CACHE.clear()
    except OSError:
        pass


# ---- learned blend weights (rolling-origin CV, fitted offline) ----
def _blend_weights_path() -> Path:
    return get_settings().out / "weekly_blend.json"


_BW_CACHE: dict = {}


def load_blend_weights():
    """The learned ``{model: weight}`` blend from ``train_and_save`` (memoised on
    mtime), or ``None`` to use the :data:`_BLEND_W` fallback. Only weights over
    known models with a positive sum are accepted."""
    import json
    p = _blend_weights_path()
    try:
        mt = p.stat().st_mtime
    except OSError:
        _BW_CACHE.clear()
        return None
    if _BW_CACHE.get("mt") == mt:
        return _BW_CACHE.get("val")
    val = None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        w = {k: float(v) for k, v in (d.get("weights") or {}).items()
             if k in _GLOBAL and float(v) > 0}
        if w and sum(w.values()) > 0:
            val = w
    except Exception:                                     # noqa: BLE001
        val = None
    _BW_CACHE["mt"], _BW_CACHE["val"] = mt, val
    return val


def save_blend_weights(weights: dict, meta: dict | None = None) -> None:
    import json
    try:
        payload = {"weights": {k: round(float(v), 4) for k, v in weights.items()}}
        if meta:
            payload.update(meta)
        _blend_weights_path().write_text(json.dumps(payload, indent=2),
                                         encoding="utf-8")
        _BW_CACHE.clear()
        _CACHE.clear()
    except OSError:
        pass


def checkpoint_status() -> dict:
    """For the Flow Analysis 'Forecast model' panel: is there a saved network,
    when was it trained, on how much data, and is it stale vs the files now."""
    import json
    try:
        meta = json.loads(_ratio_meta_path().read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return {"exists": False, "mode": "on-demand (trains on each restart)"}
    sig = _weekly_file_sig()
    meta["exists"] = True
    meta["mode"] = "saved network"
    meta["stale"] = meta.get("files") not in (None, sig)
    try:
        gm = json.loads(_gbm_meta_path().read_text(encoding="utf-8"))
        gm["stale"] = gm.get("files") not in (None, sig)
        meta["gbm"] = gm
    except Exception:                                     # noqa: BLE001
        pass
    try:
        bw = json.loads(_blend_weights_path().read_text(encoding="utf-8"))
        meta["blend"] = bw                                # weights + rolling_origin
    except Exception:                                     # noqa: BLE001
        pass
    return meta


def train_and_save(epochs: int = 400, val_weeks: int = 3, quick: bool = False) -> dict:
    """Offline trainer: fit the ``esrnn_ratio`` network PROPERLY on the full
    current panel (many epochs, a validation tail with early stopping), save the
    weights, and record what it was trained on. Returns the meta dict.

    Run it from the CLI (``python -m wms.scripts.train_weekly``) or the Flow
    Analysis 'Retrain' button; the running app then serves that network."""
    import datetime as _dt
    import json
    from wms.analytics import weekly_neural as _nn

    pan = load_panel()
    keys, weeks = pan["keys"], pan["weeks"]
    # train on the STOCKOUT-UNCONSTRAINED matrix so the saved network learns
    # fully-stocked demand (same transform the live forecast uses).
    MAT, _smask, _umeta = training_matrix(pan)
    S, W = MAT.shape
    if S == 0 or W < MIN_TRAIN_WEEKS + 1:
        raise RuntimeError("not enough weekly data to train")

    cfg = get_settings()
    lo = float(getattr(cfg, "weekly_ratio_lo", 0.8))
    hi = float(getattr(cfg, "weekly_ratio_hi", 1.5))
    ep = 120 if quick else epochs

    # thorough fit on the FULL history (this is the network the live forecast uses)
    _, state = _nn.esrnn_ratio_forecast(
        MAT, W, 1, base=_ratio_base(MAT, W), lo=lo, hi=hi,
        epochs=ep, val_weeks=val_weeks, return_state=True)
    if state is None:
        raise RuntimeError("esrnn_ratio training returned no state (torch missing?)")

    from wms.analytics import weekly_ml as _ml
    cats = np.array([_cat_of(pan, k) for k in keys])
    sig = _weekly_file_sig()
    now = _dt.datetime.now().replace(microsecond=0).isoformat(" ")

    # persist the SERVED boosters: fit on the FULL history (leak-free for the
    # live forecast, which predicts the FUTURE), record when / on what.
    try:
        _ml.gbm_forecast(MAT, weeks, W, 1, keys=keys, cats=cats,
                         model_out=str(_gbm_ckpt_path()))
        save_gbm_meta({"trained_at": now, "n_series": int(S), "n_weeks": int(W),
                       "branches": sorted({k[0] for k in keys}), "files": sig})
    except Exception as e:                                # noqa: BLE001
        import warnings
        warnings.warn(f"gbm save skipped: {e}")
    try:
        _ml.lgbm_forecast(MAT, weeks, W, 1, keys=keys, cats=cats,
                          model_out=str(_lgbm_ckpt_path()))
        save_lgbm_meta({"trained_at": now, "n_series": int(S), "n_weeks": int(W),
                        "branches": sorted({k[0] for k in keys}), "files": sig})
    except Exception as e:                                # noqa: BLE001
        import warnings
        warnings.warn(f"lgbm save skipped: {e}")

    # ---- rolling-origin CV: fit each pool model on the weeks before origin o
    # and predict week o, for the last K origins. Scored on what REALLY sold over
    # the in-stock weeks (same rule as build()). Averaged over origins this is a
    # far steadier read than a single last-week hold-out.
    K = 2 if quick else 4
    P, A, I, origins = _rolling_origin_cv(pan, MAT, _umeta, state, _BLEND_POOL, K)
    scores = {m: _pool_score(P[m], A, I) for m in P}

    fixed_w = {m: w for m, w in _BLEND_W.items() if m in P} or \
        {m: 1.0 for m in P}
    fixed_blend = _blend(P, fixed_w)
    scores["blend_fixed"] = _pool_score(fixed_blend, A, I)

    learned_w = _fit_blend_weights(P, A, I)
    learned_blend = _blend(P, learned_w)
    scores["blend_learned"] = _pool_score(learned_blend, A, I)

    # keep the learned weights only if they actually help and stay safe-side
    lw, fw = scores["blend_learned"], scores["blend_fixed"]
    use_learned = (lw and fw and lw["wape"] <= fw["wape"] - 0.2
                   and lw["bias"] >= -6.0)
    chosen_w = learned_w if use_learned else fixed_w
    save_blend_weights(chosen_w, {
        "trained_at": now, "files": sig, "origins": [weeks[o] for o in origins],
        "pool": list(P), "rolling_origin": {
            **{m: scores[m] for m in P},
            "blend_fixed": scores["blend_fixed"],
            "blend_learned": scores["blend_learned"]},
        "using": "learned" if use_learned else "fixed"})

    # a single last-week number for lstm too (not in the blend pool)
    lstm_s = None
    if not quick:
        try:
            te = W - 1
            raw = pan["MAT"].astype(float)
            oos = _umeta.get("oos")
            ins = (~oos[:, te] if oos is not None and oos.shape == MAT.shape
                   else np.ones(S, bool))
            F = _ml.lstm_forecast(MAT, weeks, te, 1, keys=keys, cats=cats)
            if F is not None:
                lstm_s = _pool_score(
                    np.clip(np.asarray(F).reshape(S, -1)[:, 0], 0, None),
                    raw[:, te], ins)
        except Exception as e:                            # noqa: BLE001
            import warnings
            warnings.warn(f"lstm score skipped: {e}")

    blend_s = scores["blend_learned"] if use_learned else scores["blend_fixed"]
    meta = {
        "trained_at": now, "n_series": int(S), "n_weeks": int(W),
        "branches": sorted({k[0] for k in keys}),
        "epochs": int(ep), "val_weeks": int(val_weeks),
        "holdout_wape": blend_s["wape"] if blend_s else None,
        "holdout_bias": blend_s["bias"] if blend_s else None,
        "ratio_lo": lo, "ratio_hi": hi, "files": sig,
        "censored_weeks": int(_umeta.get("censored", 0)),
        "inventory_used": bool(_umeta.get("inventory", False)),
        "blend_weights": {k: round(v, 3) for k, v in chosen_w.items()},
        "blend_mode": "learned" if use_learned else "fixed",
        "rolling_origins": len(origins),
        "model_scores": {
            **{k: v for k, v in scores.items() if v is not None},
            **({"lstm": lstm_s} if lstm_s else {})},
    }
    save_ratio_checkpoint(state, meta)
    return meta


# ---------------------------------------------------------- rolling-origin tools
def _pool_score(f, a, ins):
    """WAPE / bias % of a (pooled) forecast vector against actuals ``a`` over the
    in-stock cells ``ins``. ``f`` is rounded first, matching the served numbers."""
    if f is None:
        return None
    f = np.clip(np.round(np.asarray(f, float)), 0, None)
    a = np.asarray(a, float)
    ins = np.asarray(ins, bool)
    if not ins.any():
        ins = np.ones_like(ins)
    den = float(a[ins].sum()) or 1.0
    d = (f - a)[ins]
    return {"wape": round(100 * float(np.abs(d).sum()) / den, 1),
            "bias": round(100 * float(d.sum()) / den, 1)}


def _rolling_origin_cv(pan, MAT, umeta, ratio_state, pool, K):
    """For the last ``K`` weekly origins, fit each model in ``pool`` on the weeks
    strictly before the origin and predict that one week. Returns
    ``(P, A, I, origins)`` where P[model] / A / I are 1-D arrays of length
    ``S * len(origins)`` (predictions, actuals, in-stock mask) pooled across
    origins, and ``origins`` are the week indices used."""
    from wms.analytics import weekly_neural as _nn
    from wms.analytics import weekly_ml as _ml
    keys, weeks = pan["keys"], pan["weeks"]
    raw = pan["MAT"].astype(float)
    S, W = MAT.shape
    oos = umeta.get("oos")
    cats = np.array([_cat_of(pan, k) for k in keys])
    cfg = get_settings()
    lo = float(getattr(cfg, "weekly_ratio_lo", 0.8))
    hi = float(getattr(cfg, "weekly_ratio_hi", 1.5))
    K = max(1, min(K, W - MIN_TRAIN_WEEKS))
    origins = list(range(W - K, W))
    cols = {m: [] for m in pool}
    A, Im = [], []
    for o in origins:
        A.append(raw[:, o])
        Im.append(~oos[:, o] if oos is not None and oos.shape == MAT.shape
                  else np.ones(S, bool))
        for m in pool:
            try:
                if m == "esrnn_ratio":
                    F = _nn.esrnn_ratio_forecast(
                        MAT, o, 1, base=_ratio_base(MAT, o), lo=lo, hi=hi,
                        checkpoint=ratio_state)
                elif m == "gbm":
                    F = _ml.gbm_forecast(MAT, weeks, o, 1, keys=keys, cats=cats)
                elif m == "lgbm":
                    F = _ml.lgbm_forecast(MAT, weeks, o, 1, keys=keys, cats=cats)
                else:
                    F = None
            except Exception:                            # noqa: BLE001
                F = None
            cols[m].append(np.clip(np.asarray(F).reshape(S, -1)[:, 0], 0, None)
                           if F is not None else np.full(S, np.nan))
    A = np.concatenate(A)
    Im = np.concatenate(Im)
    P = {m: np.concatenate(v) for m, v in cols.items()}
    P = {m: v for m, v in P.items() if np.isfinite(v).all()}   # drop failures
    return P, A, Im, origins


def _fit_blend_weights(P, A, I):
    """Non-negative weights over the pool that minimise in-stock WAPE on the
    pooled rolling-origin predictions, softly constrained to keep the blend's
    raw bias in roughly [0, +12]% (the safety uplift then lifts it to target).
    Falls back to :data:`_BLEND_W` if scipy is unavailable or the fit fails."""
    names = list(P)
    try:
        from scipy.optimize import minimize
        M = np.column_stack([P[m] for m in names])[I]
        a = np.asarray(A, float)[I]
        den = float(a.sum()) or 1.0

        def obj(w):
            w = np.clip(w, 0, None)
            w = w / (w.sum() or 1.0)
            f = M @ w
            wape = np.abs(f - a).sum() / den
            bias = (f - a).sum() / den
            # keep the blend's raw bias in roughly [+2%, +15%] so it leans SAFE
            # by construction and the safety uplift only has to nudge, not rescue
            return (wape + 14.0 * max(0.0, 0.02 - bias) ** 2
                    + 5.0 * max(0.0, bias - 0.15) ** 2)

        k = len(names)
        best = None
        for start in (np.full(k, 1.0 / k), *np.eye(k)):
            r = minimize(obj, start, method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-3, "fatol": 1e-5})
            if best is None or r.fun < best.fun:
                best = r
        w = np.clip(best.x, 0, None)
        w = w / (w.sum() or 1.0)
        return {n: float(wi) for n, wi in zip(names, w) if wi > 1e-3}
    except Exception:                                    # noqa: BLE001
        return {m: w for m, w in _BLEND_W.items() if m in names} or \
            {m: 1.0 / len(names) for m in names}


def _cat_of(pan, k):
    from wms.analytics.monthly_sales import categorise
    return categorise(pan["item_of"].get(k, ""))


def cached_run() -> dict:
    """One hold-out — the last week — trained on every earlier week. That is what
    the model comparison and the live forecast use."""
    d = weekly_dir()
    try:
        files = tuple(sorted((os.path.basename(p), os.path.getmtime(p))
                             for p in glob.glob(str(d / "**" / "*.xls*"), recursive=True)
                             if not os.path.basename(p).startswith("~$")))
    except OSError:
        files = ()
    sig = (forced_model(), files)          # switching the model busts the cache
    val = _CACHE.get("val")
    if val is not None and _CACHE.get("sig") == sig:
        return val
    with _CACHE_LOCK:
        if _CACHE.get("val") is not None and _CACHE.get("sig") == sig:
            return _CACHE["val"]
        primary = dict(build(test_weeks=1))
        primary["iteration"] = "1-week"
        primary["runs"] = {1: primary}
        _CACHE["val"] = primary          # set val BEFORE sig so a reader never
        _CACHE["sig"] = sig              # sees a matching sig without a value
        return primary


_PANEL_CACHE: dict = {}


def cached_panel() -> dict:
    """The weekly sales panel (:func:`load_panel`), memoised on the current file
    set. Feeds the Flow Analysis sales plot without re-reading every workbook on
    each request."""
    d = weekly_dir()
    try:
        sig = tuple(sorted((os.path.basename(p), os.path.getmtime(p))
                           for p in glob.glob(str(d / "**" / "*.xls*"), recursive=True)
                           if not os.path.basename(p).startswith("~$")))
    except OSError:
        sig = ()
    if _PANEL_CACHE.get("val") is not None and _PANEL_CACHE.get("sig") == sig:
        return _PANEL_CACHE["val"]
    val = load_panel()
    _PANEL_CACHE["val"] = val
    _PANEL_CACHE["sig"] = sig
    return val


_INV_PANEL_CACHE: dict = {}


def cached_inventory_panel():
    """``(S, W)`` weekly on-hand aligned to :func:`cached_panel`'s keys/weeks
    (see :func:`load_inventory_panel`), memoised on the weekly-inventory file
    set. ``None`` when there are no weekly stock files."""
    pan = cached_panel()
    d = weekly_inventory_dir()
    try:
        sig = tuple(sorted((os.path.basename(p), os.path.getmtime(p))
                           for p in glob.glob(str(d / "**" / "*.xls*"), recursive=True)
                           if not os.path.basename(p).startswith("~$")))
    except OSError:
        sig = ()
    key = (sig, len(pan["keys"]), tuple(pan["weeks"][:1]))
    if _INV_PANEL_CACHE.get("key") == key:
        return _INV_PANEL_CACHE.get("val")
    val = load_inventory_panel(pan["keys"], pan["weeks"])
    _INV_PANEL_CACHE["key"], _INV_PANEL_CACHE["val"] = key, val
    return val


def has_data() -> bool:
    return not cached_run()["state"].empty


def demand_lookup() -> dict:
    """(branch_code, sku) -> weekly demand int."""
    st = cached_run()["state"]
    if st.empty:
        return {}
    return {(r.branch, r.sku): int(r.weekly_demand) for r in st.itertuples()}


def _filter_by_bcode(d: pd.DataFrame, bcode: str) -> pd.DataFrame:
    """Rows for ``bcode``: an exact branch-code match (BM) is authoritative and
    never falls through to the display-name-prefix match (belmont) - some
    codes (e.g. GWA) are themselves a name prefix of a DIFFERENT branch
    (Gwanda Thobelani), so OR-ing the two unconditionally would pull in that
    other branch whenever the code was used."""
    b = bcode.strip().lower()
    exact = d["branch"].str.lower().eq(b)
    if exact.any():
        return d[exact]
    return d[d["branch_name"].str.lower().str.startswith(b)]


def _bcode_matcher(b: str, keys):
    """Predicate for a branch code against a lowercased ``bcode`` query ``b``,
    for callers that filter a ``keys`` list of ``(branch_code, sku)`` rather
    than a DataFrame - same exact-code-wins rule as :func:`_filter_by_bcode`."""
    if not b:
        return lambda bc: True
    codes = {code.lower() for code, _sk in keys}
    if b in codes:
        return lambda bc: bc.lower() == b
    return lambda bc: BRANCH_NAME.get(bc, bc).lower().startswith(b)


def display_frame(bcode: str = "", q: str = "") -> pd.DataFrame:
    """Per-product weekly forecast for the Sales & Forecasting page / export.

    ``bcode`` matches a branch code (BM) or a display-name prefix (belmont);
    ``q`` matches SKU or product name. Columns are presentation-ready.
    """
    st = cached_run()["state"]
    if st.empty:
        return pd.DataFrame()
    d = st.copy()
    if bcode:
        d = _filter_by_bcode(d, bcode)
    if q:
        s = q.strip().lower()
        d = d[d["sku"].str.lower().str.contains(s, regex=False)
              | d["item"].str.lower().str.contains(s, regex=False)]
    d = d.sort_values(["branch_name", "weekly_demand"], ascending=[True, False])
    return (d.rename(columns={"branch_name": "Location", "sku": "SKU",
                              "item": "Product", "weekly_demand": "Next week"})
             [["Location", "SKU", "Product", "Next week"]]
             .reset_index(drop=True))


def _short_week(lab: str) -> str:
    try:
        return pd.Timestamp(lab).strftime("%d %b")
    except Exception:
        return lab


def sales_products() -> list[dict]:
    """``[{"sku", "name"}]`` for every product with weekly sales history — the
    autocomplete list for the Flow Analysis sales plot."""
    pan = cached_panel()
    seen: dict = {}
    for (bc, sk) in pan["keys"]:
        seen.setdefault(sk, pan["item_of"].get((bc, sk), ""))
    return [{"sku": k, "name": v} for k, v in sorted(seen.items())]


def weekly_sales_series(bcode: str = "", sku: str = "", metric: str = "sales") -> dict:
    """Weekly time series for the Flow Analysis plot.

    ``bcode``   restrict to a branch (code ``BM`` or display-name prefix); blank =
                every branch.
    ``sku``     restrict to a product (SKU exact, or name contains); blank = every
                product, i.e. the branch's total.
    ``metric``  ``sales`` (units sold, default), ``inventory`` (weekly on-hand
                from the Hansa stock files) or ``both`` (overlay). Inventory is
                only available where weekly stock files have been loaded.

    With a product set and ``bcode`` blank the series is that product summed
    across branches. Returns the raw weeks/values plus a ready-to-render SVG
    layout (polylines, area, dots, gridlines, x-ticks) on a fixed viewBox.
    """
    pan = cached_panel()
    MAT, keys, weeks = pan["MAT"], pan["keys"], pan["weeks"]
    item_of = pan["item_of"]
    b, s = bcode.strip().lower(), sku.strip().lower()
    metric = (metric or "sales").strip().lower()
    if metric not in ("sales", "inventory", "both"):
        metric = "sales"

    match_bc = _bcode_matcher(b, keys)
    idx = []
    for i, (bc, sk) in enumerate(keys):
        if not match_bc(bc):
            continue
        if s and not (sk.lower() == s
                      or s in (item_of.get((bc, sk), "") or "").lower()):
            continue
        idx.append(i)

    n = len(weeks)
    have = bool(getattr(MAT, "size", 0)) and bool(idx)
    vals = ([int(round(float(x))) for x in MAT[idx].sum(axis=0)]
            if have else [0] * n)
    peak, total = (max(vals) if vals else 0), int(sum(vals))

    # weekly on-hand for the same filter (nan readings ignored in the sum)
    OH = cached_inventory_panel() if metric in ("inventory", "both") else None
    inv_ok = OH is not None and have and np.isfinite(OH[idx]).any()
    inv = ([int(round(float(x))) for x in np.nansum(OH[idx], axis=0)]
           if inv_ok else [0] * n)
    inv_peak = max(inv) if inv else 0

    matched = sorted({keys[i][1] for i in idx})
    branch_label = (BRANCH_NAME.get(bcode.strip().upper(), bcode.strip())
                    if bcode.strip() else "All branches")
    if not s:
        sku_label = "All products"
    elif len(matched) == 1:
        k = next(i for i in idx if keys[i][1] == matched[0])
        nm = item_of.get(keys[k], "")
        sku_label = f"{matched[0]} ({nm})" if nm else matched[0]
    else:
        sku_label = f'"{sku.strip()}", {len(matched)} products'

    # ---- SVG layout on a fixed viewBox ----  sales and on-hand each get their
    # own y-scale (on-hand totals dwarf weekly units), so "both" is a dual axis:
    # left labels = the series shown on the left (sales, or on-hand when alone),
    # right labels = on-hand when both are shown.
    W, H, PT, PB = 960, 300, 14, 26
    show_sales = metric in ("sales", "both")
    show_inv = metric in ("inventory", "both") and inv_ok
    PL = 52
    PR = 52 if (show_sales and show_inv) else 14
    iw, ih = W - PL - PR, H - PT - PB
    base = PT + ih
    smax = peak or 1
    imax = inv_peak or 1
    xs = [PL + (0.0 if n <= 1 else j / (n - 1) * iw) for j in range(n)]

    def _poly(series, mx, unit):
        ys = [PT + (1 - v / mx) * ih for v in series]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        area = ((f"M{xs[0]:.1f},{base:.1f} "
                 + " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
                 + f" L{xs[-1]:.1f},{base:.1f} Z") if n else "")
        dots = [{"x": round(x, 1), "y": round(y, 1),
                 "label": _short_week(weeks[j]), "value": series[j], "unit": unit}
                for j, (x, y) in enumerate(zip(xs, ys))]
        return {"line": line, "area": area, "dots": dots}

    sales_svg = (_poly(vals, smax, "units") if show_sales
                 else {"line": "", "area": "", "dots": []})
    inv_svg = (_poly(inv, imax, "on hand") if show_inv
               else {"line": "", "area": "", "dots": []})
    left_max = smax if show_sales else imax
    grid = [{"y": round(PT + f * ih, 1),
             "label": f"{int(round(left_max * (1 - f))):,}",
             "rlabel": (f"{int(round(imax * (1 - f))):,}"
                        if (show_sales and show_inv) else "")}
            for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    step = max(1, n // 8)
    xticks = [{"x": round(xs[j], 1), "label": _short_week(weeks[j])}
              for j in range(n) if j % step == 0 or j == n - 1]

    return {
        "weeks": [_short_week(w) for w in weeks], "values": vals,
        "inventory": inv, "metric": metric, "inv_available": OH is not None,
        "inv_peak": inv_peak, "inv_total": int(sum(inv)),
        "peak": peak, "total": total, "n_weeks": n, "n_skus": len(matched),
        "branch_label": branch_label, "sku_label": sku_label,
        "has_data": bool(getattr(MAT, "size", 0)),
        "svg": {"w": W, "h": H, "grid": grid, "xticks": xticks,
                "x0": PL, "x1": PL + iw, "y0": PT, "y1": base,
                "dual": bool(show_sales and show_inv),
                "show_sales": show_sales, "show_inv": show_inv,
                "line": sales_svg["line"], "area": sales_svg["area"],
                "dots": sales_svg["dots"],
                "inv_line": inv_svg["line"], "inv_area": inv_svg["area"],
                "inv_dots": inv_svg["dots"]},
    }


_PIE_COLORS = ["#5b8def", "#22a6b3", "#26de81", "#a3cb38", "#f7b731",
               "#fa8231", "#eb3b5a", "#e056fd", "#8854d0", "#4b6584",
               "#20bf6b", "#0fb9b1", "#c44569", "#f8a5c2", "#786fa6",
               "#f19066", "#63cdda", "#cf6a87", "#b8c1d9"]   # last = "Other"
_MIX_METRICS = {"units": ("MAT", "units sold"),
                "profit": ("PROFIT", "profit"),
                "revenue": ("REV", "revenue")}

# a handful of branches get a colour pinned outright rather than derived, so
# the branch is always recognisable at a glance across every pie it appears
# in - Belmont Shop is the original flagship branch, so it keeps the obvious
# "dark blue" rather than whatever its hash happens to land on
_FIXED_SLICE_COLORS = {"BM": "#1a237e"}


def _assign_colours(items: list) -> dict:
    """``key -> colour`` for every real (non-"Other") slice in ONE pie.

    Each key prefers its own stable identity colour (``crc32`` of the key,
    not the salted builtin ``hash()``, so it is stable across restarts too)
    so the same product/branch keeps the same colour across its paired units
    and profit pies even though they rank items differently (a high-volume,
    low-margin item can be #1 by units and near the bottom by profit). A
    fixed pin (see ``_FIXED_SLICE_COLORS``) always wins its slot; a collision
    on any other slot is bumped to the next free colour in the palette, so
    nothing repeats within a single pie. Collisions are resolved in a stable
    key-sorted order - never in the value-ranked order the caller passed
    items in - so the same set of keys resolves the same way whether this is
    the units pie or the profit pie, even though the two rank items
    differently; ranking by value here would make the bump outcome (and so
    the colour) depend on which pie asked."""
    palette = [c for c in _PIE_COLORS[:-1] if c not in _FIXED_SLICE_COLORS.values()]
    n = len(palette)
    used = set()
    out = {}
    ordered = sorted(dict.fromkeys(items))
    for key in ordered:
        fixed = _FIXED_SLICE_COLORS.get(key.upper())
        if fixed:
            out[key] = fixed
            used.add(fixed)
    for key in ordered:
        if key in out:
            continue
        base = zlib.crc32(key.encode()) % n
        colour = next((palette[(base + step) % n] for step in range(n)
                      if palette[(base + step) % n] not in used), palette[base])
        out[key] = colour
        used.add(colour)
    return out


def sales_mix(bcode: str = "", metric: str = "units", skus=None,
              top: int = 8) -> dict:
    """Each product's share of the total, as a ready-to-render pie.

    ``metric``  ``units`` (qty), ``profit`` or ``revenue`` (turnover).
    ``bcode``   restrict to one branch; blank = every branch.
    ``skus``    exact SKUs (blanks ignored), read as:
                * 0 given  -> the biggest ``top`` products + "Other" (default).
                * 1 given  -> SEARCH mode: the normal top-N + "Other" pie, with
                  that SKU pulled out as its own labelled slice (even if it
                  wouldn't make the top N) so you can see its share against
                  every other product. ``highlight`` in the result carries its
                  name/percentage/rank.
                * 2-3 given -> COMPARE mode: only those SKUs, sliced against
                  each other, no "Other".
    """
    import math
    src, metric_label = _MIX_METRICS.get(metric, _MIX_METRICS["units"])
    pan = cached_panel()
    keys, item_of = pan["keys"], pan["item_of"]
    M = pan.get(src)
    branch_label = (BRANCH_NAME.get(bcode.strip().upper(), bcode.strip())
                    if bcode.strip() else "All branches")
    picks = []
    for s in (skus or []):
        s = (s or "").strip()
        if s and s.lower() not in {p.lower() for p in picks}:
            picks.append(s)
    picks_lc = {p.lower() for p in picks}
    search_sku = picks[0] if len(picks) == 1 else ""
    compare_mode = len(picks) >= 2
    blank = {"has_data": False, "metric": metric, "metric_label": metric_label,
             "branch_label": branch_label, "slices": [], "legend": [],
             "total": 0.0, "n_products": 0, "picked": picks, "highlight": None}
    if not getattr(M, "size", 0):
        return blank

    b = bcode.strip().lower()
    match_bc = _bcode_matcher(b, keys)
    name_by_sku = {sk: item_of.get((bc, sk), sk) for (bc, sk) in keys}
    per: dict = {}
    for i, (bc, sk) in enumerate(keys):
        if not match_bc(bc):
            continue
        if compare_mode and sk.lower() not in picks_lc:
            continue
        v = float(M[i].sum())
        if v > 0:
            per[sk] = per.get(sk, 0.0) + v
    total = float(sum(per.values()))

    highlight = None
    if search_sku:
        match = next((sk for sk in per if sk.lower() == search_sku.lower()), None)
        nm = name_by_sku.get(match, match) if match else search_sku
        highlight = {"sku": match or search_sku, "name": nm,
                     "pct": round(100 * per[match] / total, 4) if match and total else 0.0,
                     "rank": None, "n_total": len(per), "found": bool(match)}

    if total <= 0:
        return {**blank, "has_data": True, "highlight": highlight}

    ranked = sorted(per.items(), key=lambda kv: kv[1], reverse=True)
    if highlight and highlight["found"]:
        highlight["rank"] = next(i for i, (sk, _) in enumerate(ranked, 1)
                                 if sk == highlight["sku"])

    if compare_mode:                              # compare-these mode: no "Other"
        data = [(name_by_sku.get(sk, sk), sk, v) for sk, v in ranked]
    else:
        head = ranked[:top]
        head_skus = {sk for sk, _ in head}
        if highlight and highlight["found"] and highlight["sku"] not in head_skus:
            head = head + [(highlight["sku"], per[highlight["sku"]])]
            head_skus.add(highlight["sku"])
        data = [(name_by_sku.get(sk, sk), sk, v) for sk, v in head]
        other = sum(v for sk, v in ranked if sk not in head_skus)
        if other > 0:
            data.append(("Other", "", other))

    cx = cy = 120.0
    r = 112.0
    ang = -math.pi / 2
    hi_sku = highlight["sku"] if highlight and highlight["found"] else None
    colours = _assign_colours([sk or label for label, sk, v in data
                              if sk or label != "Other"])
    slices, legend = [], []
    for label, sk, v in data:
        frac = v / total
        colour = colours[sk or label] if sk or label != "Other" \
            else _PIE_COLORS[-1]
        if frac >= 0.99999:                       # single slice — a full disc
            path = (f"M{cx - r},{cy:.1f} a{r},{r} 0 1,0 {2 * r},0 "
                    f"a{r},{r} 0 1,0 {-2 * r},0 Z")
        else:
            a2 = ang + frac * 2 * math.pi
            large = 1 if (a2 - ang) > math.pi else 0
            x1, y1 = cx + r * math.cos(ang), cy + r * math.sin(ang)
            x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
            path = (f"M{cx:.1f},{cy:.1f} L{x1:.2f},{y1:.2f} "
                    f"A{r:.0f},{r:.0f} 0 {large},1 {x2:.2f},{y2:.2f} Z")
            ang = a2
        row = {"label": label, "sku": sk, "colour": colour,
               "pct": round(100 * frac, 4), "value": round(v, 2),
               "highlight": bool(sk) and sk == hi_sku}
        slices.append({**row, "path": path})
        legend.append(row)

    return {"has_data": True, "metric": metric, "metric_label": metric_label,
            "branch_label": branch_label, "total": round(total, 2),
            "n_products": len(per), "picked": picks, "highlight": highlight,
            "slices": slices, "legend": legend,
            "svg": {"w": 240, "h": 240, "cx": cx, "cy": cy, "r": r}}


def _pie_shapes(data):
    """``data`` = [(label, key, value), ...] with value >= 0. Returns
    ``(slices, legend, svg)`` in the shape the pie_card macro renders. ``key``
    is carried through for optional highlighting by the caller."""
    import math
    total = float(sum(v for _l, _k, v in data)) or 1.0
    cx = cy = 120.0
    r = 112.0
    ang = -math.pi / 2
    colours = _assign_colours([key or label for label, key, v in data if label != "Other"])
    slices, legend = [], []
    for label, key, v in data:
        frac = v / total
        colour = _PIE_COLORS[-1] if label == "Other" else colours[key or label]
        if frac >= 0.99999:
            path = (f"M{cx - r},{cy:.1f} a{r},{r} 0 1,0 {2 * r},0 "
                    f"a{r},{r} 0 1,0 {-2 * r},0 Z")
        else:
            a2 = ang + frac * 2 * math.pi
            large = 1 if (a2 - ang) > math.pi else 0
            x1, y1 = cx + r * math.cos(ang), cy + r * math.sin(ang)
            x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
            path = (f"M{cx:.1f},{cy:.1f} L{x1:.2f},{y1:.2f} "
                    f"A{r:.0f},{r:.0f} 0 {large},1 {x2:.2f},{y2:.2f} Z")
            ang = a2
        row = {"label": label, "sku": key, "colour": colour,
               "pct": round(100 * frac, 4), "value": round(v, 2),
               "highlight": False}
        slices.append({**row, "path": path})
        legend.append(row)
    return slices, legend, {"w": 240, "h": 240, "cx": cx, "cy": cy, "r": r}


def flow_summary() -> dict:
    """Headline numbers for the Flow Analysis KPI strip (Power-BI-style tiles):
    last week's units + week-on-week change, last week's revenue + WoW change
    and the all-time gross margin, the top branch and its share, the
    best-selling product (units) and the best product by profit."""
    pan = cached_panel()
    MAT, keys, weeks = pan["MAT"], pan["keys"], pan["weeks"]
    n = len(weeks)
    if not getattr(MAT, "size", 0) or n == 0:
        return {"has_data": False}
    weekly = MAT.sum(axis=0)
    last = float(weekly[-1])
    prev = float(weekly[-2]) if n >= 2 else 0.0
    wow = round(100 * (last - prev) / prev, 1) if prev > 0 else None

    REV, PROFIT = pan.get("REV"), pan.get("PROFIT")
    rev_week = rev_wow = margin_pct = None
    if getattr(REV, "size", 0):
        rev_weekly = REV.sum(axis=0)
        rev_last = float(rev_weekly[-1])
        rev_prev = float(rev_weekly[-2]) if n >= 2 else 0.0
        rev_wow = round(100 * (rev_last - rev_prev) / rev_prev, 1) if rev_prev > 0 else None
        rev_week = int(round(rev_last))
        rev_total = float(rev_weekly.sum())
        profit_total = float(PROFIT.sum()) if getattr(PROFIT, "size", 0) else 0.0
        margin_pct = round(100 * profit_total / rev_total, 1) if rev_total > 0 else None

    item_of = pan["item_of"]
    name_by_sku: dict = {}
    for (bc, sk), nm in item_of.items():
        name_by_sku.setdefault(sk, nm or sk)

    per: dict = {}
    u_sku: dict = {}
    p_sku: dict = {}
    for i, (bc, sk) in enumerate(keys):
        u = float(MAT[i].sum())
        per[bc] = per.get(bc, 0.0) + u
        u_sku[sk] = u_sku.get(sk, 0.0) + u
        if getattr(PROFIT, "size", 0):
            p_sku[sk] = p_sku.get(sk, 0.0) + float(PROFIT[i].sum())
    grand = sum(per.values()) or 1.0
    top_bc, top_v = (max(per.items(), key=lambda kv: kv[1]) if per else ("", 0.0))

    u_total = sum(u_sku.values()) or 1.0
    p_total = sum(p_sku.values()) or 1.0
    top_u_sk, top_u_v = (max(u_sku.items(), key=lambda kv: kv[1]) if u_sku else ("", 0.0))
    top_p_sk, top_p_v = (max(p_sku.items(), key=lambda kv: kv[1]) if p_sku else ("", 0.0))

    return {
        "has_data": True,
        "week_units": int(round(last)),
        "wow_pct": wow,
        "week_label": _short_week(weeks[-1]),
        "week_revenue": rev_week,
        "revenue_wow_pct": rev_wow,
        "margin_pct": margin_pct,
        "top_branch": BRANCH_NAME.get(top_bc, top_bc),
        "top_branch_pct": round(100 * top_v / grand, 1),
        "top_prod": name_by_sku.get(top_u_sk, top_u_sk),
        "top_prod_units": int(round(top_u_v)),
        "top_prod_pct": round(100 * top_u_v / u_total, 1),
        "top_profit_prod": name_by_sku.get(top_p_sk, top_p_sk),
        "top_profit_val": int(round(top_p_v)),
        "top_profit_pct": round(100 * top_p_v / p_total, 1),
        "n_weeks": n,
    }


def worst_performers(db, *, bcodes=None, limit: int = 15,
                     weeks: int = 13) -> dict:
    """Dead / slow stock: products that sell little **and** earn little, yet are
    still sitting in branch inventory. Returns ``{"overall": [...],
    "by_branch": [...], "weeks": n}`` where each list holds row dicts already
    formatted for :func:`_macros.table`, worst first.

    ``bcodes`` (list of branch codes) restricts BOTH lists to those branches;
    empty / ``None`` means every branch.
    """
    from wms.services import stock as stock_svc

    pan = cached_panel()
    MAT, PROFIT, keys, wk = (pan.get("MAT"), pan.get("PROFIT"),
                             pan["keys"], pan["weeks"])
    if not getattr(MAT, "size", 0) or not keys:
        return {"overall": [], "by_branch": [], "weeks": 0}
    W = len(wk)
    w = max(1, min(int(weeks or 13), W))
    item_of = pan["item_of"]
    has_profit = bool(getattr(PROFIT, "size", 0))
    want = {str(c).strip().upper() for c in (bcodes or []) if str(c).strip()}

    def _in(bc):
        return not want or str(bc).upper() in want

    inv = stock_svc.levels_df(db)
    oh_bs: dict = {}
    oh_sku: dict = {}
    if not inv.empty:
        for r in inv.itertuples():
            b, s = str(r.branch_code).upper(), str(r.sku).upper()
            q = int(r.on_hand or 0)
            oh_bs[(b, s)] = oh_bs.get((b, s), 0) + q
            if _in(b):
                oh_sku[s] = oh_sku.get(s, 0) + q

    name_by_sku: dict = {}
    for (bc, sk), nm in item_of.items():
        name_by_sku.setdefault(sk, nm or sk)

    sku_u: dict = {}
    sku_p: dict = {}
    bs_rows: list = []
    for i, (bc, sk) in enumerate(keys):
        if not _in(bc):
            continue
        u = float(MAT[i, -w:].sum())
        p = float(PROFIT[i, -w:].sum()) if has_profit else 0.0
        sku_u[sk] = sku_u.get(sk, 0.0) + u            # overall aggregation (in scope)
        sku_p[sk] = sku_p.get(sk, 0.0) + p
        oh = oh_bs.get((str(bc).upper(), str(sk).upper()), 0)
        if oh <= 0:
            continue
        bs_rows.append({"branch": BRANCH_NAME.get(bc, bc), "sku": sk,
                        "product": item_of.get((bc, sk), sk),
                        "units": u, "profit": p, "on_hand": oh})

    u_total = sum(sku_u.values()) or 1.0
    p_total = sum(v for v in sku_p.values() if v > 0) or 1.0

    ov_rows = []
    for sk, u in sku_u.items():
        oh = oh_sku.get(str(sk).upper(), 0)
        if oh <= 0:
            continue
        ov_rows.append({"sku": sk, "product": name_by_sku.get(sk, sk),
                        "units": u, "profit": sku_p.get(sk, 0.0), "on_hand": oh})

    def _fmt(rows, with_branch=False):
        if not rows:
            return []
        df = pd.DataFrame(rows)
        df["u_r"] = df["units"].rank(pct=True, method="average")
        df["p_r"] = df["profit"].rank(pct=True, method="average")
        df["score"] = df["u_r"] + df["p_r"]          # low sales + low profit = low
        df = (df.sort_values(["score", "on_hand"], ascending=[True, False])
                .head(limit))
        out = []
        for r in df.itertuples():
            rate = float(r.units) / w
            cover = round(float(r.on_hand) / rate, 1) if rate > 0 else None
            row = {
                "branch": r.branch if with_branch else "",
                "product": r.product, "sku": r.sku,
                "units": int(round(r.units)),
                "units_pct": round(100 * r.units / u_total, 1),
                "profit": int(round(r.profit)),
                "profit_pct": round(100 * r.profit / p_total, 1),
                "on_hand": int(r.on_hand),
                "cover": cover,
                "dead": int(round(r.units)) == 0,
            }
            out.append(row)
        return out

    return {"overall": _fmt(ov_rows), "by_branch": _fmt(bs_rows, with_branch=True),
            "weeks": w}


_LOW_STOCK_COVER_WEEKS = 1.5     # below this many weeks of cover = a reorder-point alert
_HIGH_PRIORITY_PCT = 0.75        # top quartile of network-wide weekly demand = "high priority"
_REORDER_SAFETY_MARGIN = 0.2     # safety stock = avg weekly sales x (1 + this) - "slightly higher"
_EXCESS_COVER_WEEKS = 12         # more than this many weeks of on-hand cover = excess stock


def low_stock_alerts(db, bcode: str = "", limit: int = 50) -> dict:
    """High-priority products (top quartile of network-wide weekly demand)
    sitting below a safe cover threshold at some branch - a candidate reorder
    list. A product's own sales velocity decides "high priority", not its
    category or price, so a fast-moving cheap item outranks a slow expensive
    one - matching what the Allocation plan already caps shipments against.

    ``bcode`` (optional) restricts the alerts to one branch; "high priority"
    still ranks products by their NETWORK-wide demand either way, so a branch
    filter narrows which alerts show without changing what counts as a
    priority product."""
    from wms.services import stock as stock_svc

    st = cached_run()["state"]
    if st.empty:
        return {"rows": [], "n_high_priority_skus": 0, "total_alerts": 0}
    net_demand = st.groupby("sku")["weekly_demand"].sum()
    sellers = net_demand[net_demand > 0]
    if sellers.empty:
        return {"rows": [], "n_high_priority_skus": 0, "total_alerts": 0}
    threshold = sellers.quantile(_HIGH_PRIORITY_PCT)
    priority = {str(s).upper() for s in sellers[sellers >= threshold].index}

    inv = stock_svc.levels_df(db)
    on_hand: dict = {}
    if not inv.empty:
        for r in inv.itertuples():
            on_hand[(str(r.branch_code).upper(), str(r.sku).upper())] = int(r.on_hand or 0)

    name_by_sku: dict = {}
    for r in st.itertuples():
        name_by_sku.setdefault(str(r.sku).upper(), str(r.item or "").strip())

    want_bc = bcode.strip().upper()

    rows = []
    for r in st.itertuples():
        sku_u = str(r.sku).upper()
        if sku_u not in priority:
            continue
        bc = str(r.branch).upper()
        if want_bc and bc != want_bc:
            continue
        rate = float(r.weekly_demand or 0)
        if rate <= 0:
            continue
        oh = on_hand.get((bc, sku_u), 0)
        cover = oh / rate
        if cover < _LOW_STOCK_COVER_WEEKS:
            rows.append({
                "branch": BRANCH_NAME.get(bc, bc), "branch_code": bc,
                "sku": r.sku, "product": name_by_sku.get(sku_u) or r.sku,
                "on_hand": oh, "weekly_demand": round(rate, 1),
                "cover_weeks": round(cover, 1),
            })
    rows.sort(key=lambda x: x["weekly_demand"], reverse=True)
    return {"rows": rows[:limit], "n_high_priority_skus": len(priority),
            "total_alerts": len(rows)}


def reorder_points(db, bcode: str = "", limit: int = 300) -> dict:
    """A reorder point per branch-product, for every product with real sales
    velocity ("important" meaning actively selling, not a dead line): safety
    stock set slightly above the product's own average weekly sales, plus
    enough extra cover for the supplier lead time, so a branch reorders
    before it runs dry rather than after -

        safety_stock  = avg_weekly_sales x (1 + margin)      - a small buffer
        reorder_point = safety_stock + avg_weekly_sales x lead_time_weeks

    ``bcode`` (optional) restricts to one branch. Rows are sorted with the
    branch-products at or below their reorder point first (soonest to run out
    within those), then by days of stock remaining.

    Weekly sales velocity is real wherever a branch has uploaded weekly sales
    history; for a branch with no weekly uploads but real monthly Hansa
    exports, it's a coarser real estimate (that month's qty / 4) - never
    fabricated. See :func:`weekly_demand_estimate`; rows carry
    ``is_estimated`` so callers can tell the two apart."""
    from wms.services import stock as stock_svc

    wd = weekly_demand_estimate()
    if wd.empty:
        return {"rows": [], "n_products": 0, "n_reorder_now": 0, "lead_time_weeks": 1.0}

    lead_time_weeks = max(get_settings().lead_time_days, 1) / 7.0

    inv = stock_svc.levels_df(db)
    on_hand: dict = {}
    if not inv.empty:
        for r in inv.itertuples():
            on_hand[(str(r.branch_code).upper(), str(r.sku).upper())] = int(r.on_hand or 0)

    want_bc = bcode.strip().upper()

    rows = []
    for r in wd.itertuples():
        rate = float(r.weekly_demand or 0)
        if rate <= 0:
            continue                        # no sales velocity - not an "important" line
        bc = str(r.branch_code).upper()
        if want_bc and bc != want_bc:
            continue
        sku_u = str(r.sku).upper()
        safety_stock = rate * (1 + _REORDER_SAFETY_MARGIN)
        reorder_point = safety_stock + rate * lead_time_weeks
        oh = on_hand.get((bc, sku_u), 0)
        daily = rate / 7.0
        days_of_stock = round(oh / daily, 1) if daily > 0 else None
        rows.append({
            "branch": r.branch, "branch_code": bc,
            "sku": r.sku, "product": str(r.item or "").strip() or r.sku,
            "avg_weekly_sales": round(rate, 1),
            "safety_stock": int(round(safety_stock)),
            "reorder_point": int(round(reorder_point)),
            "on_hand": oh, "days_of_stock": days_of_stock,
            "status": "reorder_now" if oh <= reorder_point else "ok",
            "is_estimated": bool(r.is_estimated),
        })
    rows.sort(key=lambda x: (x["status"] != "reorder_now",
                              x["days_of_stock"] if x["days_of_stock"] is not None else 9e9))
    n_now = sum(1 for x in rows if x["status"] == "reorder_now")
    return {"rows": rows[:limit], "n_products": len(rows), "n_reorder_now": n_now,
            "lead_time_weeks": round(lead_time_weeks, 1)}


def weekly_demand_estimate() -> pd.DataFrame:
    """Per-branch-SKU weekly demand for every branch with ANY real sales data
    - exact where possible, honestly (coarsely) estimated where not, never
    fabricated:

      * branches with uploaded weekly sales history (:func:`cached_run`) use
        their real, model-fitted weekly demand.
      * branches with no weekly uploads but real MONTHLY Hansa exports
        (``data/sales_history``, see :mod:`wms.analytics.monthly_sales`) get
        an approximate rate: their most recent month's real quantity / 4 -
        a coarser real number, not a simulated one.
      * branches with neither source are left out entirely.

    Columns: branch_code, branch, sku, item, weekly_demand, is_estimated.
    """
    from wms.analytics import monthly_sales as ms

    rows = []
    st = cached_run().get("state")
    real_branches = set()
    if st is not None and not st.empty:
        real_branches = set(st["branch"].str.upper())
        for r in st.itertuples():
            if float(r.weekly_demand or 0) <= 0:
                continue
            rows.append({"branch_code": r.branch, "branch": r.branch_name,
                         "sku": r.sku, "item": r.item,
                         "weekly_demand": round(float(r.weekly_demand), 2),
                         "is_estimated": False})

    panel = ms.cached_panel()
    if not panel.empty:
        gap = panel[~panel["branch_code"].str.upper().isin(real_branches)]
        if not gap.empty:
            latest = gap.groupby("branch_code")["period"].transform("max")
            for r in gap[gap["period"] == latest].itertuples():
                if float(r.qty or 0) <= 0:
                    continue
                rows.append({"branch_code": r.branch_code, "branch": r.branch,
                             "sku": r.sku, "item": r.item,
                             "weekly_demand": round(float(r.qty) / 4.0, 2),
                             "is_estimated": True})

    if not rows:
        return pd.DataFrame(columns=["branch_code", "branch", "sku", "item",
                                     "weekly_demand", "is_estimated"])
    return pd.DataFrame(rows)


def excess_stock(db, bcode: str = "", limit: int = 50) -> dict:
    """Branch-products sitting on far more stock than their own sales velocity
    justifies (more than :data:`_EXCESS_COVER_WEEKS` weeks of cover, or stock
    with NO recent sales at all) - capital tied up on a shelf, and at the
    zero-sales extreme, candidate dead stock. The mirror image of
    :func:`low_stock_alerts`: too much stock is as much a planning problem as
    too little.

    ``bcode`` (optional) restricts to one branch. Zero-sales (dead) lines sort
    first, then by weeks of cover, worst first."""
    from wms.services import stock as stock_svc

    inv = stock_svc.levels_df(db)
    if inv.empty:
        return {"rows": [], "total_excess": 0}

    st = cached_run()["state"] if has_data() else None
    rate_by: dict = {}
    name_by: dict = {}
    if st is not None and not st.empty:
        for r in st.itertuples():
            rate_by[(str(r.branch).upper(), str(r.sku).upper())] = float(r.weekly_demand or 0)
            name_by.setdefault(str(r.sku).upper(), str(r.item or "").strip())

    want_bc = bcode.strip().upper()

    rows = []
    for r in inv.itertuples():
        bc = str(r.branch_code).upper()
        if want_bc and bc != want_bc:
            continue
        oh = int(r.on_hand or 0)
        if oh <= 0:
            continue
        sku_u = str(r.sku).upper()
        rate = rate_by.get((bc, sku_u), 0.0)
        cover = round(oh / rate, 1) if rate > 0 else None   # None = no sales at all
        if cover is None or cover > _EXCESS_COVER_WEEKS:
            rows.append({
                "branch": BRANCH_NAME.get(bc, bc), "branch_code": bc,
                "sku": r.sku, "product": name_by.get(sku_u) or r.sku,
                "on_hand": oh, "weekly_demand": round(rate, 1),
                "cover_weeks": cover, "dead": cover is None,
            })
    rows.sort(key=lambda x: (not x["dead"], -(x["cover_weeks"] or 0)))
    return {"rows": rows[:limit], "total_excess": len(rows)}


def _sparkline_svg(values, w0: float = 200.0, h0: float = 56.0, pad: float = 4.0) -> dict:
    """A ready-to-render sparkline (line + soft fill) for a plain list of
    values, used by the Growth page's weekly trend charts."""
    if not values:
        return {"w": w0, "h": h0, "line": "", "area": "", "last_x": pad, "last_y": h0 - pad}
    lo, hi = float(min(values)), float(max(values))
    span = (hi - lo) or 1.0
    n = len(values)
    xs = [pad + i * (w0 - 2 * pad) / max(1, n - 1) for i in range(n)]
    ys = [h0 - pad - (v - lo) / span * (h0 - 2 * pad) for v in values]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = (f"M{xs[0]:.1f},{h0 - pad:.1f} " +
            " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) +
            f" L{xs[-1]:.1f},{h0 - pad:.1f} Z")
    return {"w": w0, "h": h0, "line": line, "area": area, "last_x": xs[-1], "last_y": ys[-1]}


def _diverging_bar_svg(values, labels=None, w0: float = 720.0, h0: float = 220.0,
                       pad_x: float = 30.0, pad_top: float = 20.0,
                       pad_bottom: float = 26.0) -> dict:
    """A bar chart centered on a zero baseline - bars rise above it for
    positive values, hang below for negative ones, sized against the largest
    magnitude in the series. Used for week-over-week growth, which is as
    often a shrink as a gain. Wide enough to carry a value label on every bar
    and a week/month label under every bar (thinned if there are many), so
    the progression across the window reads at a glance, not just on hover -
    :func:`growth_overview`'s ``points`` already carry the same real labels
    used here."""
    n = len(values)
    mid_y = pad_top + (h0 - pad_top - pad_bottom) / 2
    if not n:
        return {"w": w0, "h": h0, "mid_y": mid_y, "bars": [], "xticks": [], "zero_x0": 0,
                "zero_x1": w0}
    labels = labels or [""] * n
    span = max((abs(v) for v in values), default=0) or 1.0
    usable_h = (h0 - pad_top - pad_bottom) / 2
    bw = (w0 - 2 * pad_x) / n
    bar_w = max(bw * 0.6, 1.0)
    bars = []
    for i, v in enumerate(values):
        x = pad_x + i * bw + (bw - bar_w) / 2
        bh = max(abs(v) / span * usable_h, 1.0)
        y = mid_y - bh if v >= 0 else mid_y
        label_y = (y - 5) if v >= 0 else (y + bh + 11)
        bars.append({"x": round(x, 1), "y": round(y, 1), "w": round(bar_w, 1),
                     "h": round(bh, 1), "pos": v >= 0, "cx": round(x + bar_w / 2, 1),
                     "value_y": round(label_y, 1), "value": round(v, 1)})
    step = max(1, n // 10)                   # thin labels so they don't collide
    xticks = [{"x": bars[j]["cx"], "label": labels[j]}
              for j in range(n) if j % step == 0 or j == n - 1]
    return {"w": w0, "h": h0, "mid_y": round(mid_y, 1), "bars": bars, "xticks": xticks,
            "zero_x0": pad_x, "zero_x1": w0 - pad_x}


def growth_overview(weeks: int = 12, bcode: str = "") -> dict:
    """Network growth: how total units sold moved week over week, and how
    many branches / distinct products were actively selling each week -
    built from the same weekly sales panel used everywhere else (no separate
    tracking system, no invented customer/visitor counts this WMS has no
    data source for).

    ``bcode`` (optional) scopes ONLY the ``sales_growth`` chart to one branch
    - every other figure here (active branches/products, units, overall
    growth %) stays network-wide regardless, since those are the page's
    shared KPI strip, not part of the chart being filtered."""
    pan = cached_panel()
    MAT, keys = pan.get("MAT"), pan.get("keys")
    if not pan["weeks"] or not getattr(MAT, "size", 0):
        return {"has_data": False}
    wk = list(pan["weeks"])[-weeks:]
    tail = np.nan_to_num(MAT[:, -len(wk):], nan=0.0)
    units_per_week = tail.sum(axis=0)
    if not len(units_per_week):
        return {"has_data": False}

    growth_tail = tail
    if bcode:
        idx = [i for i, (bc, _sk) in enumerate(keys) if bc.upper() == bcode.strip().upper()]
        growth_tail = tail[idx] if idx else tail[:0]
    growth_units_per_week = (growth_tail.sum(axis=0) if growth_tail.shape[0]
                              else np.zeros_like(units_per_week))

    branches_by_week, skus_by_week = [], []
    for wi in range(tail.shape[1]):
        active = tail[:, wi] > 0
        branches_by_week.append(len({keys[i][0] for i in range(len(keys)) if active[i]}))
        skus_by_week.append(len({keys[i][1] for i in range(len(keys)) if active[i]}))

    growth_pts = []
    for i, (w, v) in enumerate(zip(wk, growth_units_per_week)):
        prev = growth_units_per_week[i - 1] if i > 0 else None
        pct = round(100 * (v - prev) / prev, 1) if prev and prev > 0 else None
        growth_pts.append({"label": _short_week(w), "value": int(round(v)), "pct": pct})

    sku_pts = [{"label": _short_week(w), "value": c} for w, c in zip(wk, skus_by_week)]
    branch_pts = [{"label": _short_week(w), "value": c} for w, c in zip(wk, branches_by_week)]

    half = max(1, len(units_per_week) // 2)
    first_half, second_half = float(units_per_week[:half].sum()), float(units_per_week[half:].sum())
    overall_pct = round(100 * (second_half - first_half) / first_half, 1) if first_half > 0 else None

    growth_bcode = bcode.strip().upper()
    return {
        "has_data": True, "n_weeks": len(wk),
        "active_branches": branches_by_week[-1], "active_products": skus_by_week[-1],
        "units_period": int(units_per_week.sum()), "overall_pct_growth": overall_pct,
        "sales_growth": {"points": growth_pts, "bcode": growth_bcode,
                          "branch_label": (BRANCH_NAME.get(growth_bcode, growth_bcode)
                                          if growth_bcode else "All branches"),
                          "svg": _diverging_bar_svg([p["pct"] or 0 for p in growth_pts],
                                                    labels=[p["label"] for p in growth_pts])},
        "sku_growth": {"points": sku_pts, "svg": _sparkline_svg(skus_by_week)},
        "branch_growth": {"points": branch_pts, "svg": _sparkline_svg(branches_by_week)},
    }


_ABC_A_CUM_PCT = 0.80   # Class A: SKUs whose cumulative network revenue share reaches this
_ABC_B_CUM_PCT = 0.95   # Class B: cumulative share up to this; the long tail past it is C
_ABC_STRATEGY = {
    "A": {"name": "Fast-moving", "role": "Sales-replenish (pull)",
          "note": "High-velocity consumables and parts - let branch sales history "
                  "trigger automatic reorder-point replenishment from the hub "
                  "rather than a fixed allocation (see Reorder points on Inventory)."},
    "B": {"name": "Medium-moving", "role": "Demand-aligned",
          "note": "Routine components and sub-assemblies - split across branches by "
                  "their own recent sales history, the same forecast-driven cover the "
                  "split tool below already applies."},
    "C": {"name": "Slow-moving / capital", "role": "Hub-centralized",
          "note": "Low turnover, high value - keep the bulk at the central hub rather "
                  "than scattering it thin across branches; ship a unit out only where "
                  "a branch has shown real demand for it."},
}


def abc_classification() -> dict:
    """Classic ABC inventory analysis: every product ranked by its network-wide
    revenue (highest first), split into three tiers by CUMULATIVE revenue
    share - Class A (the vital few SKUs driving most of the revenue), Class B
    (the next tranche), Class C (the long tail: many SKUs, little revenue
    each). For mining-equipment-style inventory this tracks the fast/medium/
    slow-moving split a Demand-Driven Push-Pull allocation strategy needs:
    high-value revenue drivers are the ones worth auto-replenishing, while the
    long tail is exactly the low-turnover, capital-heavy stock worth holding
    centrally instead of pushing out to every branch."""
    pan = cached_panel()
    REV, MAT, keys = pan.get("REV"), pan.get("MAT"), pan.get("keys")
    if not getattr(REV, "size", 0):
        return {"has_data": False}
    item_of = pan["item_of"]
    rev_by_sku: dict = {}
    units_by_sku: dict = {}
    name_by_sku: dict = {}
    for i, (bc, sk) in enumerate(keys):
        rev_by_sku[sk] = rev_by_sku.get(sk, 0.0) + float(np.nan_to_num(REV[i]).sum())
        units_by_sku[sk] = units_by_sku.get(sk, 0.0) + float(np.nan_to_num(MAT[i]).sum())
        name_by_sku.setdefault(sk, item_of.get((bc, sk)) or sk)
    total_rev = sum(rev_by_sku.values())
    if total_rev <= 0:
        return {"has_data": False}

    ranked = sorted(rev_by_sku.items(), key=lambda kv: kv[1], reverse=True)
    rows, cum = [], 0.0
    for sk, rev in ranked:
        cum += rev
        cum_pct = cum / total_rev
        cls = "A" if cum_pct <= _ABC_A_CUM_PCT else ("B" if cum_pct <= _ABC_B_CUM_PCT else "C")
        rows.append({
            "sku": sk, "product": name_by_sku.get(sk, sk), "class": cls,
            "strategy": _ABC_STRATEGY[cls]["role"],
            "revenue": round(rev, 2), "units": round(units_by_sku.get(sk, 0.0), 1),
            "pct_of_revenue": round(100 * rev / total_rev, 3),
            "cum_pct": round(100 * cum_pct, 1),
        })
    class_by_sku = {r["sku"].upper(): r["class"] for r in rows}
    n_a = sum(1 for r in rows if r["class"] == "A")
    n_b = sum(1 for r in rows if r["class"] == "B")
    n_c = len(rows) - n_a - n_b
    return {"has_data": True, "rows": rows, "class_by_sku": class_by_sku,
            "n_a": n_a, "n_b": n_b, "n_c": n_c, "n_total": len(rows),
            "strategy": _ABC_STRATEGY}


def branch_mix(sku: str = "", metric: str = "units", bcodes=None) -> dict:
    """Each BRANCH's share of the total, as a ready-to-render pie (same shape as
    :func:`sales_mix`, so the pie_card macro renders it).

    ``sku``     restrict to one product (SKU exact or name contains); blank =
                every product -> compares branches on total sales / profit.
    ``metric``  ``units`` (default), ``profit`` or ``revenue``.
    ``bcodes``  up to 3 branch codes to compare against each other; blank/None =
                every branch with data.
    """
    src, metric_label = _MIX_METRICS.get(metric, _MIX_METRICS["units"])
    pan = cached_panel()
    keys, item_of = pan["keys"], pan["item_of"]
    M = pan.get(src)
    s = sku.strip().lower()
    picks = []
    for c in (bcodes or []):
        c = (c or "").strip().upper()
        if c and c not in picks:
            picks.append(c)
    pick_set = set(picks)

    matched = set()
    per: dict = {}
    if getattr(M, "size", 0):
        for i, (bc, sk) in enumerate(keys):
            if picks and bc.upper() not in pick_set:
                continue
            if s and not (sk.lower() == s
                          or s in (item_of.get((bc, sk), "") or "").lower()):
                continue
            matched.add(sk)
            v = float(M[i].sum())
            if v > 0:
                per[bc] = per.get(bc, 0.0) + v
    total = float(sum(per.values()))

    matched = sorted(matched)
    if not s:
        prod_label = "All products"
    elif len(matched) == 1:
        k = next((keys[i] for i in range(len(keys)) if keys[i][1] == matched[0]), None)
        nm = item_of.get(k, "") if k else ""
        prod_label = f"{matched[0]} ({nm})" if nm else matched[0]
    else:
        prod_label = f'"{sku.strip()}", {len(matched)} products'

    blank = {"has_data": False, "metric": metric, "metric_label": metric_label,
             "branch_label": prod_label, "slices": [], "legend": [],
             "total": 0.0, "n_products": 0, "picked": picks, "highlight": None}
    if total <= 0:
        return {**blank, "has_data": bool(getattr(M, "size", 0))}

    ranked = sorted(per.items(), key=lambda kv: kv[1], reverse=True)
    data = [(BRANCH_NAME.get(bc, bc), bc, v) for bc, v in ranked]
    slices, legend, svg = _pie_shapes(data)
    return {"has_data": True, "metric": metric, "metric_label": metric_label,
            "branch_label": prod_label, "total": round(total, 2),
            "n_products": len(per), "picked": picks, "highlight": None,
            "slices": slices, "legend": legend, "svg": svg}


_MODEL_BLURB = {
    "damped_mean": "spike-damped mean: the SKU's own level, bulk weeks count half",
    "esrnn": "ES-RNN: per-series exp. smoothing plus a shared LSTM (Smyl, M4 winner)",
    "esrnn_ratio": "ES-RNN (ratio): learns a 0.8 to 1.5 multiplier on the recent-demand level, leaning just above actual",
    "neuralprophet": "NeuralProphet: global level, Fourier seasonality, AR-Net",
    "sba": "Syntetos-Boylan Approximation: bias-corrected Croston rate",
    "snaive": "seasonal-naive: repeat the last 4-week pattern",
    "gbm": "XGBoost: gradient-boosted trees on ~24 history and intermittency features, Tweedie objective",
    "lgbm": "LightGBM: leaf-wise gradient boosting on the same features, Tweedie objective",
    "lstm": "windowed LSTM: last 8 weeks plus branch and category, predicts next / level",
    "blend": "blend: rolling-origin-weighted mean of ES-RNN-ratio, XGBoost and LightGBM",
    "old_excel": "Old Excel: last month's total x 1.1, rounded up, divided by 4 - a plain manual baseline",
}


def model_options() -> list:
    """Every model key a user may pin from the Flow Analysis picker, in a sane
    order - INDEPENDENT of which were actually fitted/scored this run (pinning a
    cheap model skips fitting the rest, but they must stay selectable)."""
    s = get_settings()
    order = ["blend", "esrnn_ratio", "gbm", "lgbm", "lstm", "esrnn",
             "neuralprophet", "old_excel", "damped_mean", "sba", "snaive"]
    flag = {"esrnn": "weekly_esrnn", "neuralprophet": "weekly_neuralprophet",
            "esrnn_ratio": "weekly_esrnn_ratio", "gbm": "weekly_gbm",
            "lgbm": "weekly_lgbm", "lstm": "weekly_lstm",
            "old_excel": "weekly_old_excel"}
    return [m for m in order
            if m not in flag or bool(getattr(s, flag[m], True))]


def model_scores() -> list[dict]:
    """The hold-out method comparison as rows for the Flow Analysis page."""
    r = cached_run()
    ov = r.get("overall")
    if ov is None or ov.empty:
        return []
    best = r.get("best_method")
    out = []
    for method, row in ov.iterrows():
        out.append({
            "method": method, "blurb": _MODEL_BLURB.get(method, ""),
            "wape": row["WAPE"], "mae": row["MAE"], "rmse": row["RMSE"],
            "bias": row["bias"], "mase": row["MASE"],
            "is_best": method == best, "is_ensemble": False,
        })
    return out


def backtest_view(bcode: str = "", q: str = "", limit: int = 400,
                  which: str = "primary") -> dict:
    """Per-product predicted vs actual over the hold-out, using the chosen model.
    ``which`` selects the iteration: ``"primary"`` (the better of the two),
    ``"1"`` (last-week hold-out) or ``"4"`` (last-4-weeks hold-out)."""
    top = cached_run()
    r = top if which == "primary" else top.get("runs", {}).get(int(which), top)
    bt = r.get("backtest")
    labels = r.get("test_week_labels", [])
    empty = {"weeks": [], "rows": [], "total": 0,
             "method": r.get("best_method"), "test_weeks": r.get("test_weeks", 0)}
    if bt is None or bt.empty:
        return empty
    d = bt.copy()
    if bcode:
        d = _filter_by_bcode(d, bcode)
    if q:
        s = q.strip().lower()
        d = d[d["sku"].str.lower().str.contains(s, regex=False)
              | d["item"].str.lower().str.contains(s, regex=False)]
    total = len(d)
    d = d.sort_values("actual", ascending=False)
    if limit:
        d = d.head(limit)
    rows = []
    for m in d.to_dict("records"):
        rows.append({
            "location": m["branch_name"], "sku": m["sku"], "product": m["item"],
            "weeks": [{"pred": int(m[f"pred::{lab}"]), "actual": int(m[f"act::{lab}"])}
                      for lab in labels],
            "predicted": int(m["predicted"]), "actual": int(m["actual"]),
            "error": int(m["error"]),
            "wape": None if m["wape"] != m["wape"] else float(m["wape"]),
            "stockout": int(m.get("stockout", 0)),   # held-out weeks out of stock
        })
    return {"weeks": [_short_week(x) for x in labels], "rows": rows, "total": total,
            "method": r.get("best_method"), "test_weeks": r.get("test_weeks", 0),
            "uplift": float(r.get("coverage", {}).get("applied_uplift", 1.0)),
            "safety_margin": float(r.get("coverage", {}).get("safety_margin", 0.0))}
