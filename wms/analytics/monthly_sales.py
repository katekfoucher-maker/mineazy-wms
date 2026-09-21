"""Load the branch monthly *Item Statistics* exports into one tidy panel.

Each source file is one HansaWorld "Item Statistics" sheet for one branch and
one calendar month:

    Item No | Item | Qty | (junk) | Profit | GP % | Turnover
    ...one row per SKU sold that month, last row = totals (Item No blank)

Filenames look like ``JULY BM SALES 28.xlsx`` / ``MAY GWA SALES.xlsx`` — the
month name and the branch token (BM / GWA) are parsed out of the name.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from wms.config import get_settings

# branch token in the filename -> (code, display name).  The upload writes files
# as "<MONTH> <CODE> SALES.xlsx"; hand-dropped files may use any of these tokens.
_BRANCH_DEFS = [
    ("BM", "Belmont Shop"), ("BTA", "Botswana"), ("DC", "DISTRIBUTION CENTER"),
    ("ES", "Esigodini"), ("ES2", "Esigodini 2"), ("FL", "Filabusi Mswela"),
    ("FLM", "Filabusi Mthwakazi"), ("FMS", "Filabusi Main Shop"),
    ("FWH", "Filabusi Warehouse"), ("GW", "Gweru"), ("GWA", "Gwanda VID"),
    ("GWL", "Gweru Luton Rd"), ("GWT", "Gwanda Thobelani"), ("JS", "Junkshop"),
    ("MP", "Maphisa"), ("TG", "Tongogara"), ("ZMA", "Zambia"),
]
_BRANCHES = {c: (c, n) for c, n in _BRANCH_DEFS}
_BRANCHES.update({
    "BELMONT": ("BM", "Belmont Shop"),
    "GWANDA": ("GWA", "Gwanda VID"), "VID": ("GWA", "Gwanda VID"),
    "GWANDA VID": ("GWA", "Gwanda VID"),
    # full display-name tokens for the branches whose backup folders/files
    # spell out the name rather than the short Hansa code
    "ESIGODINI": ("ES", "Esigodini"),
    "GWERU": ("GW", "Gweru"),
    "MAPHISA": ("MP", "Maphisa"),
    "TONGOGARA": ("TG", "Tongogara"),
    "ZAMBIA": ("ZMA", "Zambia"),
    "JUNKSHOP": ("JS", "Junkshop"),
    "BOTSWANA": ("BTA", "Botswana"),
    # "Filabusi" alone is ambiguous across 4 branches - the filenames spell
    # out which one ("FILABUSI MTHWAKAZI SALES.xlsx"), so key off that second,
    # unambiguous word instead; bare "FILABUSI" is deliberately not mapped
    "MTHWAKAZI": ("FLM", "Filabusi Mthwakazi"),
    "MSWELA": ("FL", "Filabusi Mswela"),
    "WAREHOUSE": ("FWH", "Filabusi Warehouse"),
    "MAIN": ("FMS", "Filabusi Main Shop"),
})
BRANCH_NAME = {c: n for c, n in _BRANCH_DEFS}          # code -> display name
_MONTHS = {
    "JAN": 1, "JANUARY": 1, "FEB": 2, "FEBRUARY": 2, "MAR": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4, "MAY": 5, "JUN": 6, "JUNE": 6, "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8, "SEP": 9, "SEPT": 9, "SEPTEMBER": 9, "OCT": 10,
    "OCTOBER": 10, "NOV": 11, "NOVEMBER": 11, "DEC": 12, "DECEMBER": 12,
}
_DEFAULT_YEAR = 2026
# a filename may carry an explicit day range for a month that is only partly
# reported yet, e.g. "1 TO 12 SEPTEMBER FILABUSI MSWELA SALES.xlsx" (a
# mid-month export) - both numbers must be plausible calendar days
_DAY_RANGE_RX = re.compile(r"\b(\d{1,2})\s*(?:TO|-)\s*(\d{1,2})\b")


def _parse_day_range(fname: str):
    """-> (day_from, day_to) if the name carries an explicit day range, else
    None (the whole month is assumed)."""
    m = _DAY_RANGE_RX.search(os.path.basename(fname).upper())
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if 1 <= a <= 31 and 1 <= b <= 31 and a <= b:
        return a, b
    return None


def history_dir() -> Path:
    s = get_settings()
    p = Path(getattr(s, "sales_history_dir", "./data/sales_history"))
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p


def data_signature() -> tuple:
    """A fingerprint of the CURRENT monthly sales data, for cache-busting
    demand_forecast.cached_run(). Uses local file names+mtimes when there are
    any (same as the original design - unaffected locally/in tests); falls
    back to a database fingerprint (row count + latest update) when there are
    none, which is what a real deploy - or a direct DB import that never
    touched a file - actually has."""
    d = history_dir()
    try:
        files = tuple(sorted((p.name, p.stat().st_mtime)
                             for p in d.glob("*.xls*") if not p.name.startswith("~$")))
    except OSError:
        files = ()
    if files:
        return files
    from sqlalchemy import func
    from wms.db import SessionLocal
    from wms.models import MonthlySalesLine
    db = SessionLocal()
    try:
        n = db.query(func.count(MonthlySalesLine.id)).scalar() or 0
        mx = db.query(func.max(MonthlySalesLine.updated_at)).scalar()
    finally:
        db.close()
    return ("db", int(n), mx.isoformat() if mx else None)


def short_month(lab: str) -> str:
    """A month-end period label -> ``"Apr 2025"``. Unlike weekly_forecast's
    ``_short_week`` this always carries the year - months repeat across the
    (multi-year) monthly history, so "Apr" alone would be ambiguous between
    e.g. 2025 and 2026."""
    try:
        return pd.Timestamp(lab).strftime("%b %Y")
    except Exception:
        return lab


_MATRIX_PANEL_CACHE: dict = {}


def cached_matrix_panel() -> dict:
    """The monthly sales panel reshaped into the same
    ``{MAT, PROFIT, REV, keys, weeks, item_of}`` matrix shape
    ``weekly_forecast.cached_panel()`` returns (series x period, one row per
    (branch, sku)), memoised on :func:`data_signature`. Flow Analysis uses
    this in place of the weekly panel wherever a branch has no real weekly
    upload history yet - real monthly totals, never a fabricated weekly
    split. The dict key stays ``"weeks"`` (not "periods") purely so it drops
    straight into the weekly chart/KPI helpers unchanged. Distinct from the
    plain-DataFrame :func:`cached_panel` used elsewhere (exports, reports)."""
    sig = data_signature()
    if _MATRIX_PANEL_CACHE.get("sig") == sig:
        return _MATRIX_PANEL_CACHE["val"]
    val = _build_matrix_panel()
    _MATRIX_PANEL_CACHE["sig"], _MATRIX_PANEL_CACHE["val"] = sig, val
    return val


def _build_matrix_panel() -> dict:
    panel = load_panel()
    empty = {"MAT": np.zeros((0, 0), np.float32), "PROFIT": np.zeros((0, 0), np.float32),
             "REV": np.zeros((0, 0), np.float32), "keys": [], "weeks": [], "item_of": {}}
    if panel.empty:
        return empty
    periods = sorted(panel["period"].unique())
    p_ix = {p: i for i, p in enumerate(periods)}
    keys = sorted(set(zip(panel["branch_code"], panel["sku"])))
    k_ix = {k: i for i, k in enumerate(keys)}
    MAT = np.zeros((len(keys), len(periods)), np.float32)
    PROFIT = np.zeros((len(keys), len(periods)), np.float32)
    REV = np.zeros((len(keys), len(periods)), np.float32)
    item_of: dict = {}
    for r in panel.itertuples():
        i, j = k_ix[(r.branch_code, r.sku)], p_ix[r.period]
        MAT[i, j] += float(r.qty or 0.0)
        REV[i, j] += float(r.turnover or 0.0)
        if r.profit is not None:
            PROFIT[i, j] += float(r.profit)
        item_of[(r.branch_code, r.sku)] = r.item
    weeks = [pd.Timestamp(p).date().isoformat() for p in periods]
    return {"MAT": MAT, "PROFIT": PROFIT, "REV": REV, "keys": keys,
            "weeks": weeks, "item_of": item_of}


# ---------------------------------------------------------------- categories
_CAT_RULES = [
    ("Cable & Electrical", r"CABLE|ELECTRIC|BREAKER|CONTACTOR|STARTER|ALTERNATOR|RECTIFIER|GENERATOR|SWITCH|SOCKET|LAMP|TORCH|WIRE ARMOUR"),
    ("Fencing & Wire", r"FENCE|BARBED|TYING WIRE|BAILING|FIELD FENCE|DROPPER|GABION|POST"),
    ("Fasteners", r"BOLT|NUT|WASHER|SCREW|ANCHOR|RIVET|THREADED"),
    ("Pumps & Water", r"PUMP|SUBPUMP|BOREHOLE|SEWAGE|POLY PIPE|PVC|HDPE|VALVE|HOSE|IRRIGAT"),
    ("Milling & Crushing", r"STAMPMILL|MILL BALL|HAMMERMILL|JAW CRUSHER|BEATER|SCREEN|LINER|DIES|SHOE|ROLLER|BUNGA"),
    ("Lubricants & Chemicals", r"OIL|GREASE|HYSPIN|LUBRI|DEGREAS|FLUX|COOLANT|BRAKE FLUID|PAINT|THINNER|SAE|15W40|SUPER SERIES"),
    ("Hand Tools", r"SPANNER|WRENCH|HAMMER|PICK|SHOVEL|CHISEL|PLIER|SCREWDRIVER|SAW|AXE|TROWEL|TAPE MEASURE|COMBINATION"),
    ("Power Tools & Accessories", r"DRILL|GRINDER|DISC|BLADE|BIT|ANGLE GRIND|WELDER|WELDING|ELECTRODE"),
    ("PPE & Workwear", r"GUMBOOT|HELMET|GLOVE|OVERALL|BOOT|GOGGLE|RESPIRATOR|MASK|EAR MUFF|VEST|MUTTON CLOTH"),
    ("Bearings & Drives", r"BEARING|P/BLOCK|PLUMMER|V BELT|PULLEY|SPROCKET|CHAIN|COUPLING|SHAFT"),
]
_CAT_RX = [(name, re.compile(rx, re.I)) for name, rx in _CAT_RULES]


def categorise(name: str) -> str:
    n = str(name or "")
    for cat, rx in _CAT_RX:
        if rx.search(n):
            return cat
    return "General / Other"


# ---------------------------------------------------------------- parsing
def _parse_name(fname: str):
    """-> (month, branch, year). ``year`` is the 4-digit year token in the
    filename (e.g. "JULY 2025 BELMONT SALES.xlsx") if there is one, else
    ``None`` - the older upload convention has no year in the name at all and
    falls back to ``_DEFAULT_YEAR``, see ``_read_file``/``parse_upload``."""
    toks = re.split(r"[\s_.-]+", os.path.basename(fname).upper())
    month = branch = year = None
    for i, t in enumerate(toks):
        if month is None and t in _MONTHS:
            month = _MONTHS[t]
        if branch is None:
            # "ESIGODINI 2" / "GWANDA THOBELANI" are DIFFERENT branches from
            # bare "ESIGODINI" / "GWANDA" - the single-token match below would
            # otherwise resolve them on the first word alone, before ever
            # seeing the disambiguating second word
            if t == "ESIGODINI" and i + 1 < len(toks) and toks[i + 1] == "2":
                branch = ("ES2", "Esigodini 2")
            elif t == "GWANDA" and i + 1 < len(toks) and toks[i + 1] == "THOBELANI":
                branch = ("GWT", "Gwanda Thobelani")
            elif t in _BRANCHES:
                branch = _BRANCHES[t]
        if year is None and re.fullmatch(r"(19|20)\d{2}", t):
            year = int(t)
    return month, branch, year


_PANEL_COLUMNS = ["branch_code", "branch", "sku", "item", "category", "period",
                  "month_label", "qty", "turnover", "profit", "gp_pct",
                  "day_from", "day_to"]


def _read_file(f: str) -> pd.DataFrame:
    """One file -> rows with columns sku, item, qty, turnover, profit, gp_pct,
    branch_code, branch, period, month_label, day_from, day_to. Empty frame if
    the filename doesn't carry a recognised month + branch."""
    month, branch, year = _parse_name(f)
    if not month or not branch:
        return pd.DataFrame()
    try:
        raw = pd.read_excel(f, sheet_name="Item Statistics", header=0)
    except ValueError:
        raw = pd.read_excel(f, header=0)
    raw = raw.rename(columns={raw.columns[0]: "sku", raw.columns[1]: "item",
                              raw.columns[2]: "qty"})
    raw = raw[raw["sku"].notna()].copy()          # drop the totals row
    raw["sku"] = raw["sku"].astype(str).str.strip()
    raw = raw[raw["sku"].str.len() > 0]
    raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce").fillna(0.0)
    tcol = next((c for c in raw.columns if str(c).lower().startswith("turnover")), None)
    gcol = next((c for c in raw.columns if str(c).lower().startswith("gp")), None)
    pcol = next((c for c in raw.columns if str(c).strip().lower() == "profit"), None)
    raw["turnover"] = pd.to_numeric(raw[tcol], errors="coerce") if tcol else 0.0
    raw["gp_pct"] = pd.to_numeric(raw[gcol], errors="coerce") if gcol else None
    raw["profit"] = pd.to_numeric(raw[pcol], errors="coerce") if pcol else None
    code, disp = branch
    period = pd.Timestamp(year=year or _DEFAULT_YEAR, month=month, day=1) + pd.offsets.MonthEnd(0)
    day_from, day_to = _parse_day_range(f) or (1, int(period.day))
    day_to = min(day_to, int(period.day))
    out = raw[["sku", "item", "qty", "turnover", "profit", "gp_pct"]].copy()
    out["branch_code"] = code
    out["branch"] = disp
    out["period"] = period
    out["month_label"] = period.strftime("%b %Y")
    out["day_from"] = day_from
    out["day_to"] = day_to
    return out


def _finish_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=_PANEL_COLUMNS)
    panel = pd.concat(frames, ignore_index=True)
    # one SKU can appear twice in a month (name variants) -> sum; day_from/to
    # come from the file, uniform within a (branch, month) since the upload
    # route keeps at most one file per branch-month
    panel = (panel.groupby(["branch_code", "branch", "sku", "period", "month_label"],
                           as_index=False)
                  .agg(item=("item", "first"), qty=("qty", "sum"),
                       turnover=("turnover", "sum"), profit=("profit", "sum"),
                       gp_pct=("gp_pct", "mean"),
                       day_from=("day_from", "first"), day_to=("day_to", "first")))
    panel["category"] = panel["item"].map(categorise)
    return panel.sort_values(["branch_code", "sku", "period"]).reset_index(drop=True)


def parse_upload(raw: bytes, filename: str, branch_code: str | None = None) -> pd.DataFrame:
    """Parse one uploaded file's bytes the same way a file on disk would be -
    used by the upload route to write straight into MonthlySalesLine. The
    month always comes from the filename; the branch does too UNLESS
    ``branch_code`` is given explicitly (the upload form lets the user pick
    the branch, so the filename itself only has to carry the month)."""
    import io
    month, branch, year = _parse_name(filename)
    if not month:
        return pd.DataFrame()
    if branch_code:
        branch = (branch_code.strip().upper(),
                 BRANCH_NAME.get(branch_code.strip().upper(), branch_code.strip().upper()))
    if not branch:
        return pd.DataFrame()
    try:
        raw_df = pd.read_excel(io.BytesIO(raw), sheet_name="Item Statistics", header=0)
    except ValueError:
        raw_df = pd.read_excel(io.BytesIO(raw), header=0)
    raw_df = raw_df.rename(columns={raw_df.columns[0]: "sku", raw_df.columns[1]: "item",
                                    raw_df.columns[2]: "qty"})
    raw_df = raw_df[raw_df["sku"].notna()].copy()
    raw_df["sku"] = raw_df["sku"].astype(str).str.strip()
    raw_df = raw_df[raw_df["sku"].str.len() > 0]
    raw_df["qty"] = pd.to_numeric(raw_df["qty"], errors="coerce").fillna(0.0)
    tcol = next((c for c in raw_df.columns if str(c).lower().startswith("turnover")), None)
    gcol = next((c for c in raw_df.columns if str(c).lower().startswith("gp")), None)
    pcol = next((c for c in raw_df.columns if str(c).strip().lower() == "profit"), None)
    raw_df["turnover"] = pd.to_numeric(raw_df[tcol], errors="coerce") if tcol else 0.0
    raw_df["gp_pct"] = pd.to_numeric(raw_df[gcol], errors="coerce") if gcol else None
    raw_df["profit"] = pd.to_numeric(raw_df[pcol], errors="coerce") if pcol else None
    code, disp = branch
    period = pd.Timestamp(year=year or _DEFAULT_YEAR, month=month, day=1) + pd.offsets.MonthEnd(0)
    day_from, day_to = _parse_day_range(filename) or (1, int(period.day))
    day_to = min(day_to, int(period.day))
    out = raw_df[["sku", "item", "qty", "turnover", "profit", "gp_pct"]].copy()
    out["branch_code"] = code
    out["branch"] = disp
    out["period"] = period
    out["month_label"] = period.strftime("%b %Y")
    out["day_from"] = day_from
    out["day_to"] = day_to
    return _finish_panel([out])


def save_month(branch_code: str, period, rows: pd.DataFrame) -> int:
    """Replace one (branch_code, period)'s rows in MonthlySalesLine. Returns
    the number of line rows saved. A single bulk INSERT (not one row at a
    time) - this matters a lot once the target is a remote database instead
    of local SQLite: hundreds of individual round trips per file adds up to
    minutes per file, a bulk statement is one round trip regardless of size."""
    from sqlalchemy import insert
    from wms.db import SessionLocal
    from wms.models import MonthlySalesLine

    period_date = pd.Timestamp(period).date()
    db = SessionLocal()
    try:
        db.query(MonthlySalesLine).filter(
            MonthlySalesLine.branch_code == branch_code,
            MonthlySalesLine.period == period_date).delete()
        values = [{
            "branch_code": branch_code, "sku": r.sku, "item": r.item,
            "period": period_date,
            "qty": float(r.qty), "turnover": float(r.turnover or 0),
            "profit": (float(r.profit) if pd.notna(r.profit) else None),
            "gp_pct": (float(r.gp_pct) if pd.notna(r.gp_pct) else None),
            "day_from": int(r.day_from), "day_to": int(r.day_to),
        } for r in rows.itertuples()]
        n = len(values)
        if values:
            db.execute(insert(MonthlySalesLine), values)
        db.commit()
        return n
    finally:
        db.close()


def merge_month(branch_code: str, period, rows: pd.DataFrame) -> int:
    """Like :func:`save_month`, but for a branch-month that may already have
    an earlier PARTIAL upload (e.g. "1 to 12 SEPTEMBER" already saved, now
    "13 to 20 SEPTEMBER" arrives). ``save_month`` always replaces the whole
    month wholesale, which is correct for a fresh/cumulative re-export but
    would silently delete the earlier days' real sales if ``rows`` only
    covers a later slice - exactly the destructive case this guards against.

    * no existing rows for this (branch, period)   -> plain save.
    * new day-range covers (>=) the existing one    -> a fresher cumulative
      export supersedes the old one; plain replace.
    * new day-range sits fully inside the existing one -> stale re-upload of
      an already-covered slice; nothing to do, existing data kept as-is.
    * new day-range is adjacent/non-overlapping     -> genuinely new days;
      per-SKU qty/turnover/profit are SUMMED with the existing rows (both
      are real, disjoint day-windows of the same month) and day_from/day_to
      widen to cover both.
    * any other (genuine) overlap                   -> ambiguous without
      per-day data; refuses to guess and raises instead of risking silent
      double-counting or loss.
    """
    from wms.db import SessionLocal
    from wms.models import MonthlySalesLine

    period_date = pd.Timestamp(period).date()
    db = SessionLocal()
    try:
        existing = db.query(MonthlySalesLine).filter(
            MonthlySalesLine.branch_code == branch_code,
            MonthlySalesLine.period == period_date).all()
    finally:
        db.close()

    if not existing:
        return save_month(branch_code, period, rows)

    new_from = int(rows["day_from"].iloc[0]) if len(rows) else 1
    new_to = int(rows["day_to"].iloc[0]) if len(rows) else 1
    old_from = min(int(r.day_from or 1) for r in existing)
    old_to = max(int(r.day_to or 1) for r in existing)

    if new_from <= old_from and new_to >= old_to:
        return save_month(branch_code, period, rows)          # fresher cumulative export
    if new_from >= old_from and new_to <= old_to:
        return 0                                               # already covered, nothing to do
    if new_from > old_to + 1 or new_to < old_from - 1:
        raise ValueError(
            f"{branch_code} {period_date}: new day range {new_from}-{new_to} doesn't "
            f"connect to the existing {old_from}-{old_to} - refusing to guess how they "
            "combine.")
    if not (new_from == old_to + 1 or new_to == old_from - 1):
        raise ValueError(
            f"{branch_code} {period_date}: new day range {new_from}-{new_to} partially "
            f"overlaps the existing {old_from}-{old_to} - refusing to guess the split "
            "(would risk double-counting or losing real sales).")

    old_df = pd.DataFrame([{
        "sku": r.sku, "item": r.item, "qty": float(r.qty or 0),
        "turnover": float(r.turnover or 0),
        "profit": (float(r.profit) if r.profit is not None else None),
        "gp_pct": (float(r.gp_pct) if r.gp_pct is not None else None),
    } for r in existing])
    merged = pd.concat([old_df, rows[["sku", "item", "qty", "turnover", "profit", "gp_pct"]]],
                       ignore_index=True)
    merged = (merged.groupby("sku", as_index=False)
                    .agg(item=("item", "first"), qty=("qty", "sum"),
                         turnover=("turnover", "sum"), profit=("profit", "sum"),
                         gp_pct=("gp_pct", "mean")))
    merged["day_from"] = min(new_from, old_from)
    merged["day_to"] = max(new_to, old_to)
    return save_month(branch_code, period, merged)


def _load_panel_from_db() -> pd.DataFrame:
    from wms.db import SessionLocal
    from wms.models import MonthlySalesLine

    db = SessionLocal()
    try:
        rows = db.query(MonthlySalesLine).all()
        if not rows:
            return pd.DataFrame(columns=_PANEL_COLUMNS)
        panel = pd.DataFrame([{
            "branch_code": r.branch_code,
            "branch": BRANCH_NAME.get(r.branch_code, r.branch_code),
            "sku": r.sku, "item": r.item or "",
            "period": pd.Timestamp(r.period),
            "month_label": pd.Timestamp(r.period).strftime("%b %Y"),
            "qty": float(r.qty or 0), "turnover": float(r.turnover or 0),
            "profit": (float(r.profit) if r.profit is not None else None),
            "gp_pct": (float(r.gp_pct) if r.gp_pct is not None else None),
            "day_from": r.day_from, "day_to": r.day_to,
        } for r in rows])
        panel["category"] = panel["item"].map(categorise)
        return panel.sort_values(["branch_code", "sku", "period"]).reset_index(drop=True)
    finally:
        db.close()


def load_panel(directory: str | os.PathLike | None = None) -> pd.DataFrame:
    """Return one row per (branch, sku, month).

    Columns: branch_code, branch, sku, item, category, period (Timestamp,
    month-end), month_label, qty, turnover, profit, gp_pct, day_from, day_to.
    ``profit`` is read directly from a "Profit" column when the export has one
    (HansaWorld's own margin figure); the last two columns default to the whole
    month (1..days-in-month) - a filename carrying an explicit day range (a
    mid-month, not-yet-complete export) narrows them, so a downstream reader
    knows the qty covers only part of the month.

    Reads uploaded Excel files under ``directory`` (or the configured
    ``sales_history_dir``) - the on-disk path this always used to take, still
    used by tests that populate a directory directly. These files only ever
    hold the most recent few months (whatever has been hand-dropped/uploaded
    since); any OLDER (branch, month) the local files don't cover - e.g. a
    bulk historical import that went straight to the database and never wrote
    a file - is pulled in from MonthlySalesLine and merged in, so imported
    history is never silently shadowed by a handful of recent local files.
    """
    d = Path(directory) if directory else history_dir()
    files = sorted(glob.glob(str(d / "*.xls*")))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    file_panel = _finish_panel([_read_file(f) for f in files]) if files else pd.DataFrame(columns=_PANEL_COLUMNS)
    db_panel = _load_panel_from_db()
    if file_panel.empty:
        return db_panel
    if db_panel.empty:
        return file_panel
    covered = set(zip(file_panel["branch_code"], file_panel["period"]))
    extra = db_panel[~db_panel.apply(lambda r: (r["branch_code"], r["period"]) in covered, axis=1)]
    if extra.empty:
        return file_panel
    return (pd.concat([file_panel, extra], ignore_index=True)
              .sort_values(["branch_code", "sku", "period"]).reset_index(drop=True))


def coverage(panel: pd.DataFrame) -> dict:
    if panel.empty:
        return {"files": 0, "months": 0, "branches": 0, "sku_month_rows": 0}
    return {
        "months": int(panel["period"].nunique()),
        "month_range": " - ".join(sorted(panel["month_label"].unique(),
                                         key=lambda s: pd.to_datetime(s))),
        "branches": ", ".join(sorted(panel["branch"].unique())),
        "distinct_skus": int(panel.groupby("branch_code")["sku"].nunique().sum()),
        "sku_month_rows": int(len(panel)),
    }


_PANEL_CACHE: dict = {}


def cached_panel() -> pd.DataFrame:
    """:func:`load_panel`, memoised on the sales-history directory's file
    signature (path + mtime of every export) so repeated calls in one request
    - or across the several Reports & Exports workbooks that all need this
    same real monthly panel - don't each re-read every .xlsx from disk."""
    d = history_dir()
    try:
        sig = tuple(sorted((f, os.path.getmtime(f)) for f in glob.glob(str(d / "*.xls*"))))
    except OSError:
        sig = ()
    if _PANEL_CACHE.get("sig") != sig:
        _PANEL_CACHE["sig"] = sig
        _PANEL_CACHE["val"] = load_panel()
    return _PANEL_CACHE["val"]


def recent_panel(panel: pd.DataFrame, months: int, branch_code: str = "") -> pd.DataFrame:
    d = panel
    if branch_code:
        d = d[d["branch_code"].str.upper() == branch_code.strip().upper()]
    periods = sorted(d["period"].unique())[-months:]
    return d[d["period"].isin(periods)]


def branch_sales_summary(months: int = 3, branch_code: str = "") -> pd.DataFrame:
    """Per-branch sales totals + trend across the last ``months`` months of
    real monthly Hansa "Item Statistics" exports - the real-data equivalent
    of the old SalesRecord-seeded ``statistics.branch_sales_summary``."""
    panel = recent_panel(cached_panel(), months, branch_code)
    if panel.empty:
        return pd.DataFrame(columns=["branch", "sales_qty", "sales_value",
                                     "distinct_skus", "avg_monthly_qty", "trend_pct"])
    rows = []
    for br, g in panel.groupby("branch"):
        by_month = (g.groupby("period")
                     .agg(qty=("qty", "sum"), day_from=("day_from", "first"),
                          day_to=("day_to", "first"))
                     .sort_index())
        # a filename carrying a day range (e.g. "1 TO 12 SEPTEMBER ... SALES")
        # means that month is still in progress - comparing its raw total
        # against a FULL prior month would read as a sales crash that isn't
        # real, so the trend compares a DAILY RATE (qty / days actually
        # covered), not the raw monthly total.
        days = (by_month["day_to"] - by_month["day_from"] + 1).clip(lower=1)
        daily_rate = by_month["qty"] / days
        half = max(1, len(by_month) // 2)
        first, last = daily_rate.iloc[:half].mean(), daily_rate.iloc[-half:].mean()
        rows.append({
            "branch": br,
            "sales_qty": int(g["qty"].sum()),
            "sales_value": round(float(g["turnover"].sum()), 2),
            "distinct_skus": int(g["sku"].nunique()),
            "avg_monthly_qty": round(float(by_month["qty"].mean()), 1),
            "trend_pct": round((last - first) / first * 100, 1) if first else None,
            "latest_month_partial": bool(days.iloc[-1] < 28),
        })
    return pd.DataFrame(rows).sort_values("sales_value", ascending=False)


def abc_classification(months: int = 3, branch_code: str = "") -> pd.DataFrame:
    """A/B/C SKUs by cumulative sales-VALUE share (80 / 95), from real monthly
    turnover - the real-data equivalent of the old
    ``statistics.abc_classification`` (which ranked by SalesRecord's fabricated
    seed data). See also :func:`wms.analytics.weekly_forecast.abc_classification`,
    the weekly-panel version the live Allocation plan page uses."""
    panel = recent_panel(cached_panel(), months, branch_code)
    if panel.empty:
        return pd.DataFrame(columns=["sku", "item", "qty", "value", "cum_share", "class"])
    g = (panel.groupby("sku").agg(item=("item", "first"), qty=("qty", "sum"),
                                  value=("turnover", "sum"))
              .sort_values("value", ascending=False).reset_index())
    total = g["value"].sum() or 1.0
    g["cum_share"] = (g["value"].cumsum() / total).round(4)
    g["class"] = np.where(g.cum_share <= 0.8, "A", np.where(g.cum_share <= 0.95, "B", "C"))
    return g


def sales_trend(months: int = 6, branch_code: str = "") -> pd.DataFrame:
    """Monthly qty/value trend, network-wide or for one branch - real monthly
    data, replacing ``statistics.sales_trend``'s weekly buckets of fabricated
    daily sales. ``partial`` flags a month whose export doesn't yet cover the
    whole month (e.g. "1 TO 12 SEPTEMBER ... SALES") - its qty/value are real
    but not directly comparable to a full month's total."""
    panel = recent_panel(cached_panel(), months, branch_code)
    if panel.empty:
        return pd.DataFrame(columns=["period", "month_label", "qty", "value", "partial"])
    out = (panel.groupby(["period", "month_label"])
                .agg(qty=("qty", "sum"), value=("turnover", "sum"),
                     day_from=("day_from", "min"), day_to=("day_to", "max"))
                .reset_index().sort_values("period"))
    out["partial"] = (out["day_to"] - out["day_from"] + 1) < 28
    return out.drop(columns=["day_from", "day_to"])
