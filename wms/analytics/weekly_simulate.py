"""Fill the weekly-forecast gap with **simulated** weekly sales, derived from
the monthly "Item Statistics" history, for whichever (branch, calendar month)
pairs have no real weekly upload yet.

Real weekly exports are laborious to prepare, so a branch's weekly history
often starts later than its monthly history does (monthly exports can be
pulled for any past month in bulk). This module spreads each month's
qty / turnover / profit across Monday-anchored weekly buckets, proportional to
how many days of that week fall inside the month, and writes them as ordinary
weekly sales files into a dedicated sub-folder of the weekly-sales directory -
so the weekly forecast model picks them up exactly like a real upload, without
ever touching or replacing the source monthly files (those stay put; nothing
here reads or writes ``monthly_sales.history_dir()`` besides the initial load).

Real weekly uploads are authoritative day by day, not month by month: a real
week file often starts partway into a month (weekly exports only began once
the branch had time to prepare them) or has a hole in the middle (an export
that was skipped one week), so this only simulates the calendar days a real
weekly file has not already reported, never the ones it has - both keeping
every real day and never double-counting one. Call :func:`regenerate` after
every monthly or weekly upload; it wipes and rebuilds the simulated set from
scratch each time, so a day that later gains real weekly coverage stops being
simulated on its own, automatically.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

from wms.analytics import monthly_sales
from wms.analytics import weekly_forecast as wfc


def simulated_dir() -> Path:
    return wfc.weekly_dir() / "_simulated_from_monthly"


def _real_week_ranges() -> dict:
    """branch_code -> [(week_start, week_end), ...] from every REAL weekly file
    (top level of ``weekly_dir()`` only - the simulated sub-folder is excluded,
    so a previous simulated run is never mistaken for real coverage)."""
    out: dict = {}
    d = wfc.weekly_dir()
    files = [f for f in sorted(glob.glob(str(d / "*.xls*")))
             if not os.path.basename(f).startswith("~$")]
    for f in files:
        code, ws = wfc.parse_name(f)
        if not code or ws is None:
            continue
        out.setdefault(code, []).append((ws, ws + pd.Timedelta(days=6)))
    return out


def _real_covered_days() -> dict:
    """branch_code -> {calendar day, ...} already reported by a REAL weekly
    upload (every day in its ``[week_start, week_start + 6]`` span) - the
    exact set of days :func:`_week_buckets` must leave untouched. Checks both
    real files (the on-disk path tests / a hand-populated directory use) and
    WeeklySalesLine in the database (what a real deploy actually has) -
    unioned, never either/or, so this can only ever under-simulate a real
    day's coverage, never double-count it."""
    out: dict = {}
    for code, ranges in _real_week_ranges().items():
        days: set = set()
        for ws, we in ranges:
            days.update(pd.date_range(ws, we, freq="D"))
        out[code] = days

    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine
    db = SessionLocal()
    try:
        rows = (db.query(WeeklySalesLine.branch_code, WeeklySalesLine.week_start)
                  .filter(WeeklySalesLine.is_simulated.is_(False))
                  .distinct().all())
    finally:
        db.close()
    for code, ws in rows:
        ws = pd.Timestamp(ws)
        days = out.setdefault(code, set())
        days.update(pd.date_range(ws, ws + pd.Timedelta(days=6), freq="D"))
    return out


def _week_buckets(year: int, month: int, day_from: int = 1, day_to: int | None = None,
                   covered: set | None = None):
    """``([(week_start, days_to_simulate)], full_days)`` - Monday-anchored
    weeks overlapping the ``[day_from, day_to]`` span of the calendar month
    (default: the whole month), where ``days_to_simulate`` counts only the
    days of that span **not** already in ``covered`` (a branch's real-reported
    calendar days). ``full_days`` is the day count of the whole span - the
    denominator for the month's average daily rate - independent of how many
    of those days end up simulated, so a real week in the middle of the month
    does not inflate the daily rate of the days around it. A file that only
    reports part of the month (a mid-month export) narrows the span, so the
    unreported remainder of the month is left alone rather than guessed at."""
    covered = covered or set()
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    if day_to is None:
        day_to = int(month_end.day)
    span_start = month_start + pd.Timedelta(days=max(1, day_from) - 1)
    span_end = min(month_start + pd.Timedelta(days=day_to - 1), month_end)
    if span_end < span_start:
        return [], 0
    full_days = (span_end - span_start).days + 1
    w = span_start - pd.Timedelta(days=int(span_start.weekday()))     # Monday on/before
    out = []
    while w <= span_end:
        we = w + pd.Timedelta(days=6)
        lo, hi = max(w, span_start), min(we, span_end)
        if hi >= lo:
            uncovered = sum(1 for d in pd.date_range(lo, hi, freq="D") if d not in covered)
            if uncovered > 0:
                out.append((w, uncovered))
        w += pd.Timedelta(days=7)
    return out, full_days


def clear() -> int:
    """Delete every simulated file (the on-disk path only - see
    clear_simulated_weeks() in weekly_forecast.py for the database rows a
    real deploy actually has). Returns how many were removed."""
    d = simulated_dir()
    if not d.exists():
        return 0
    n = 0
    for f in d.glob("*.xls*"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


def regenerate() -> dict:
    """Rebuild every simulated weekly row from the current monthly history,
    in WeeklySalesLine (is_simulated=True). Safe to call after any monthly or
    weekly upload; idempotent. The old set is only cleared right before the
    write loop below, not up front - a concurrent page load or upload that
    reads the simulated rows while this is running should never see them
    empty during the (comparatively slow) monthly-panel load and per-SKU
    apportioning that happens first."""
    panel = monthly_sales.load_panel()
    if panel.empty:
        clear()
        wfc.clear_simulated_weeks()
        wfc._CACHE.clear()
        wfc._PANEL_CACHE.clear()
        return {"weeks_written": 0, "branches": [], "months_skipped": 0}

    covered_days = _real_covered_days()

    # (branch_code, week_start) -> {sku: {"item", "qty", "turnover", "profit"}}
    contrib: dict = {}
    months_skipped = 0
    touched: set = set()

    for (bc, period), grp in panel.groupby(["branch_code", "period"]):
        year, month = int(period.year), int(period.month)
        day_from = int(grp["day_from"].iloc[0]) if "day_from" in grp else 1
        day_to = int(grp["day_to"].iloc[0]) if "day_to" in grp else None
        buckets, full_days = _week_buckets(year, month, day_from, day_to,
                                            covered_days.get(bc))
        if not buckets:
            months_skipped += 1
            continue
        total_days = full_days or 1
        for row in grp.itertuples():
            qty = float(row.qty or 0.0)
            turnover = float(row.turnover or 0.0)
            # the export's own Profit figure when there is one; gp_pct's scale
            # (points vs a fraction) varies by source, so it is never guessed at
            rprofit = getattr(row, "profit", None)
            profit = float(rprofit) if pd.notna(rprofit) else 0.0
            for w_start, ndays in buckets:
                frac = ndays / total_days
                if frac <= 0:
                    continue
                cell = (contrib.setdefault((bc, w_start), {})
                               .setdefault(row.sku, {"item": row.item, "qty": 0.0,
                                                     "turnover": 0.0, "profit": 0.0}))
                cell["qty"] += qty * frac
                cell["turnover"] += turnover * frac
                cell["profit"] += profit * frac
                touched.add(bc)

    clear()
    wfc.clear_simulated_weeks()
    written = 0
    for (bc, w_start), skus in contrib.items():
        rows = [{"sku": sku, "item": c["item"], "qty": round(c["qty"], 2),
                "revenue": round(c["turnover"], 2), "profit": round(c["profit"], 2)}
               for sku, c in skus.items() if c["qty"] > 0]
        if not rows:
            continue
        wfc.save_simulated_week(bc, w_start, pd.DataFrame(rows))
        written += 1

    wfc._CACHE.clear()
    wfc._PANEL_CACHE.clear()
    return {"weeks_written": written, "branches": sorted(touched),
            "months_skipped": months_skipped}
