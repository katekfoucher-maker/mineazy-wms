"""Simulated weekly sales derived from monthly history: fills the gap for a
branch/month with no real weekly upload yet, and steps aside the moment real
weekly data covers that month (never double-counted)."""
import io

import pandas as pd
import pytest

from wms.analytics import monthly_sales
from wms.analytics import weekly_forecast as wf
from wms.analytics import weekly_simulate as wsim


def _month_file(path, rows):
    """rows: (sku, item, qty, turnover, gp_pct) -> an 'Item Statistics' xlsx like
    the HansaWorld monthly export monthly_sales.load_panel() reads."""
    recs = [{"Item No": s, "Item": it, "Qty": q, "Unnamed: 3": None,
            "Turnover": tv, "GP %": gp} for s, it, q, tv, gp in rows]
    recs.append({"Item No": None, "Item": "TOTAL", "Qty": sum(r[2] for r in rows)})
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame(recs).to_excel(xl, sheet_name="Item Statistics", index=False)


def _upload_week(branch_code, filename, rows):
    """Push one real weekly-sales file straight into WeeklySalesLine, the same
    way the /allocation/upload-weekly route does - real weekly data is
    database-only now, same as the simulated rows it sits alongside."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        pd.DataFrame([{"Item No": s, "Item": it, "Qty": q} for s, it, q in rows]
                    ).to_excel(xl, sheet_name="Item Statistics", index=False)
    parsed = wf.parse_upload_sales(buf.getvalue(), filename, branch_code=branch_code)
    assert parsed, f"could not parse {filename}"
    code, ws, df = parsed
    wf.save_week(code, ws, df)
    return ws


@pytest.fixture(autouse=True)
def _isolate(tmp_path_factory, monkeypatch, seeded):
    """history_dir points at an empty temp dir for the monthly SOURCE files
    these tests write directly (monthly_sales.load_panel()'s "files if
    present" path picks them up). weekly_dir is monkeypatched too, purely so
    weekly_simulate.clear()'s legacy file cleanup never touches the real
    project's data/ directory - real and simulated weekly rows are both
    database-only now (WeeklySalesLine), so each test clears that table (and
    MonthlySalesLine) itself rather than relying on directory isolation."""
    wd = tmp_path_factory.mktemp("weekly")
    hd = tmp_path_factory.mktemp("monthly")
    monkeypatch.setattr(wf, "weekly_dir", lambda: wd)
    monkeypatch.setattr(monthly_sales, "history_dir", lambda: hd)
    wf._CACHE.clear()
    wf._PANEL_CACHE.clear()
    _clear_weekly_sales_lines()
    _clear_monthly_sales_lines()
    yield wd, hd
    _clear_weekly_sales_lines()
    _clear_monthly_sales_lines()


def _clear_weekly_sales_lines():
    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine
    db = SessionLocal()
    try:
        db.query(WeeklySalesLine).delete()
        db.commit()
    finally:
        db.close()


def _clear_monthly_sales_lines():
    from wms.db import SessionLocal
    from wms.models import MonthlySalesLine
    db = SessionLocal()
    try:
        db.query(MonthlySalesLine).delete()
        db.commit()
    finally:
        db.close()


def _simulated_weeks(branch_code: str | None = None):
    """[(branch_code, week_start date), ...] for every simulated row - the
    DB equivalent of the old ``simulated_dir().glob("*.xlsx")`` check."""
    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine
    db = SessionLocal()
    try:
        q = db.query(WeeklySalesLine.branch_code, WeeklySalesLine.week_start).filter(
            WeeklySalesLine.is_simulated.is_(True))
        if branch_code:
            q = q.filter(WeeklySalesLine.branch_code == branch_code)
        return sorted(set(q.all()))
    finally:
        db.close()


def test_regenerate_fills_every_week_of_a_month_with_no_real_data(_isolate):
    wd, hd = _isolate
    _month_file(hd / "MAY BM SALES.xlsx",
               [("SKU1", "Widget", 100, 500.0, 20.0),
                ("SKU2", "Gadget", 40, 200.0, 10.0)])

    summary = wsim.regenerate()
    assert summary["weeks_written"] > 0
    assert summary["branches"] == ["BM"]

    weeks = _simulated_weeks("BM")
    assert weeks
    for _code, ws in weeks:
        ws = pd.Timestamp(ws)
        assert (ws.year, ws.month) in {(2026, 4), (2026, 5)}   # month + its edge week

    # the simulated weeks reconstruct the month's per-SKU total
    panel = wf.load_panel()
    i1 = panel["keys"].index(("BM", "SKU1"))
    i2 = panel["keys"].index(("BM", "SKU2"))
    assert panel["MAT"][i1].sum() == pytest.approx(100, abs=0.5)
    assert panel["MAT"][i2].sum() == pytest.approx(40, abs=0.5)


def test_a_month_with_partial_real_weekly_coverage_fills_only_the_gap_days(_isolate):
    """Real weekly data rarely covers a full month end-to-end - it usually
    starts partway through (weekly exports began later than monthly history)
    or has a hole in the middle. The days a real week does not reach must
    still come from the monthly average, never dropped, and the real week's
    own figures are never touched or blended with the simulated estimate."""
    wd, hd = _isolate
    _month_file(hd / "MAY BM SALES.xlsx", [("SKU1", "Widget", 310, 0.0, None)])   # 31 days -> 10/day
    _upload_week("BM", "BM 2026-05-04 week.xlsx", [("SKU1", "Widget", 49)])       # Mon 4 - Sun 10, real

    summary = wsim.regenerate()
    assert "BM" in summary["branches"]                    # the rest of May still needs filling

    for _code, ws in _simulated_weeks("BM"):
        assert pd.Timestamp(ws) != pd.Timestamp("2026-05-04")   # the real week is never re-simulated

    panel = wf.load_panel()
    i = panel["keys"].index(("BM", "SKU1"))
    # the real week's own (different) total, plus the other 24 days at 10/day
    assert panel["MAT"][i].sum() == pytest.approx(49 + 24 * 10, abs=0.5)


def test_a_fully_covered_month_is_never_simulated(_isolate):
    wd, hd = _isolate
    _month_file(hd / "MAY BM SALES.xlsx", [("SKU1", "Widget", 140, 0.0, None)])
    wsim.regenerate()
    assert _simulated_weeks("BM")                          # May got simulated first

    # five real weeks covering every day of May 2026 end to end
    for ws in ("2026-04-27", "2026-05-04", "2026-05-11", "2026-05-18", "2026-05-25"):
        _upload_week("BM", f"BM {ws} week.xlsx", [("SKU1", "Widget", 7)])
    summary = wsim.regenerate()
    assert "BM" not in summary["branches"]                # nothing left uncovered

    panel = wf.load_panel()
    i = panel["keys"].index(("BM", "SKU1"))
    assert panel["MAT"][i].sum() == pytest.approx(35, abs=0.01)   # only the 5 real weeks, not 175


def test_other_branches_and_months_are_unaffected(_isolate):
    wd, hd = _isolate
    _month_file(hd / "MAY BM SALES.xlsx", [("SKU1", "Widget", 100, 0.0, None)])   # 31 days
    _month_file(hd / "JUNE BM SALES.xlsx", [("SKU1", "Widget", 80, 0.0, None)])
    _month_file(hd / "MAY GWA SALES.xlsx", [("SKU9", "Gizmo", 60, 0.0, None)])

    _upload_week("BM", "BM 2026-05-04 week.xlsx", [("SKU1", "Widget", 20)])
    summary = wsim.regenerate()

    # BM's other May days still need filling; June for BM and May for GWA are untouched by real data
    assert "GWA" in summary["branches"]
    assert "BM" in summary["branches"]
    panel = wf.load_panel()
    i_bm = panel["keys"].index(("BM", "SKU1"))
    i_gwa = panel["keys"].index(("GWA", "SKU9"))
    # real week (20) + the other 24 days of May at 100/31 per day + all of June
    assert panel["MAT"][i_bm].sum() == pytest.approx(20 + 100 * 24 / 31 + 80, abs=0.5)
    assert panel["MAT"][i_gwa].sum() == pytest.approx(60, abs=0.5)


def test_week_buckets_leaves_only_uncovered_days(_isolate):
    # no real coverage: the whole month is up for simulation
    buckets, full_days = wsim._week_buckets(2026, 5, covered=set())
    assert full_days == 31
    assert sum(n for _w, n in buckets) == 31

    # the whole month covered by real data leaves nothing to simulate
    all_days = set(pd.date_range("2026-05-01", "2026-05-31", freq="D"))
    buckets, full_days = wsim._week_buckets(2026, 5, covered=all_days)
    assert buckets == []
    assert full_days == 31

    # only a middle week covered: the days before and after it remain
    mid_week = set(pd.date_range("2026-05-04", "2026-05-10", freq="D"))
    buckets, full_days = wsim._week_buckets(2026, 5, covered=mid_week)
    assert full_days == 31
    assert sum(n for _w, n in buckets) == 31 - 7
    assert pd.Timestamp("2026-05-04") not in {w for w, _n in buckets}


def test_regenerate_is_idempotent_and_clears_stale_weeks(_isolate):
    wd, hd = _isolate
    _month_file(hd / "MAY BM SALES.xlsx", [("SKU1", "Widget", 50, 0.0, None)])
    wsim.regenerate()
    before = set(_simulated_weeks())

    _month_file(hd / "JUNE BM SALES.xlsx", [("SKU1", "Widget", 30, 0.0, None)])
    wsim.regenerate()
    after = set(_simulated_weeks())
    assert before < after                                 # June's weeks were added

    # remove the monthly source entirely -> a rebuild leaves nothing simulated
    for f in hd.glob("*.xlsx"):
        f.unlink()
    summary = wsim.regenerate()
    assert summary["weeks_written"] == 0
    assert not _simulated_weeks()


def test_no_monthly_history_is_a_no_op(_isolate):
    summary = wsim.regenerate()
    assert summary == {"weeks_written": 0, "branches": [], "months_skipped": 0}


def test_a_partial_month_export_only_simulates_the_days_it_covers(_isolate):
    """A mid-month snapshot like '1 TO 12 SEPTEMBER ... SALES.xlsx' must not be
    treated as if it were the whole month - the unreported rest of the month
    is left alone rather than diluted / guessed at."""
    wd, hd = _isolate
    _month_file(hd / "1 TO 12 SEPTEMBER FL SALES.xlsx",
               [("SKU1", "Widget", 120, 0.0, None)])   # 12 days -> 10 units/day
    summary = wsim.regenerate()
    assert "FL" in summary["branches"]

    for _code, ws in _simulated_weeks("FL"):
        # every simulated week must fall within the 1-12 September span
        assert pd.Timestamp(ws).date() <= pd.Timestamp("2026-09-12").date()

    panel = wf.load_panel()
    i = panel["keys"].index(("FL", "SKU1"))
    # exactly the reported total - nothing invented for days 13-30
    assert panel["MAT"][i].sum() == pytest.approx(120, abs=0.5)


def test_upload_monthly_sales_route_works_without_a_ui_card(_isolate, seeded):
    """The 'Monthly sales' upload form was dropped from the Sales & Forecasting
    page (the weekly-sales upload is the front door now), but the underlying
    POST /allocation/upload (kind=sales) route must keep working - a branch's
    history still arrives this way whenever a real weekly export isn't ready
    yet. The route writes straight into MonthlySalesLine now, not a file - see
    monthly_sales.parse_upload / save_month. It deliberately does NOT trigger
    weekly_simulate.regenerate() any more - simulated/estimated weekly figures
    were judged too unreliable to feed the Allocation plan, so the weekly
    model now only ever trains on real weekly uploads (weekly_simulate.py
    itself still works and is exercised directly by the tests above, it is
    just no longer wired into the live upload routes)."""
    wd, hd = _isolate
    from fastapi.testclient import TestClient
    from wms.api.main import app

    c = TestClient(app)
    c.post("/login", data={"username": "controller", "password": "wms1234"},
           follow_redirects=False)

    buf = io.BytesIO()
    df = pd.DataFrame([{"Item No": "AAA", "Item": "Widget", "Qty": 100,
                        "Turnover": 500.0, "GP %": 0.2}])
    with pd.ExcelWriter(buf) as xl:
        df.to_excel(xl, sheet_name="Item Statistics", index=False)

    r = c.post("/allocation/upload",
              data={"kind": "sales", "branch_code": "BM"},
              files={"file": ("MAY BM SALES.xlsx", buf.getvalue(),
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
              follow_redirects=False)
    assert r.status_code == 303

    from wms.db import SessionLocal
    from wms.models import MonthlySalesLine
    db = SessionLocal()
    try:
        rows = db.query(MonthlySalesLine).filter(MonthlySalesLine.branch_code == "BM").all()
    finally:
        db.close()
    assert rows and rows[0].sku == "AAA"
    assert not _simulated_weeks("BM")                       # no auto gap-filling any more
