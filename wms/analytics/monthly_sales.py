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
_BRANCHES.update({"BELMONT": ("BM", "Belmont Shop"), "GWANDA": ("GWA", "Gwanda VID"),
                  "VID": ("GWA", "Gwanda VID"), "GWANDA VID": ("GWA", "Gwanda VID")})
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
    toks = re.split(r"[\s_.-]+", os.path.basename(fname).upper())
    month = branch = None
    for t in toks:
        if month is None and t in _MONTHS:
            month = _MONTHS[t]
        if branch is None and t in _BRANCHES:
            branch = _BRANCHES[t]
    return month, branch


def load_panel(directory: str | os.PathLike | None = None) -> pd.DataFrame:
    """Return one row per (branch, sku, month).

    Columns: branch_code, branch, sku, item, category, period (Timestamp,
    month-end), month_label, qty, turnover, profit, gp_pct, day_from, day_to.
    ``profit`` is read directly from a "Profit" column when the export has one
    (HansaWorld's own margin figure); the last two columns default to the whole
    month (1..days-in-month) - a filename carrying an explicit day range (a
    mid-month, not-yet-complete export) narrows them, so a downstream reader
    knows the qty covers only part of the month.
    """
    d = Path(directory) if directory else history_dir()
    files = sorted(glob.glob(str(d / "*.xls*")))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    frames = []
    for f in files:
        month, branch = _parse_name(f)
        if not month or not branch:
            continue
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
        period = pd.Timestamp(year=_DEFAULT_YEAR, month=month, day=1) + pd.offsets.MonthEnd(0)
        day_from, day_to = _parse_day_range(f) or (1, int(period.day))
        day_to = min(day_to, int(period.day))
        out = raw[["sku", "item", "qty", "turnover", "profit", "gp_pct"]].copy()
        out["branch_code"] = code
        out["branch"] = disp
        out["period"] = period
        out["month_label"] = period.strftime("%b %Y")
        out["day_from"] = day_from
        out["day_to"] = day_to
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["branch_code", "branch", "sku", "item", "category",
                                     "period", "month_label", "qty", "turnover", "profit",
                                     "gp_pct", "day_from", "day_to"])

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
