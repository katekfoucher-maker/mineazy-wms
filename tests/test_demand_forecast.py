"""Monthly demand forecast from the branch Item-Statistics exports."""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from wms.analytics import monthly_sales as ms
from wms.analytics import demand_forecast as dfc


@pytest.fixture(scope="module")
def panel(seeded):
    # load_panel() merges in any DB-only historical months, so the DB schema
    # must exist before it runs
    p = ms.load_panel()
    if p.empty:
        pytest.skip("no sales_history files present")
    return p


def test_panel_shape(panel):
    # the real sales_history keeps growing as branches upload more months, so
    # these check the invariants a fresh snapshot from May 2026 must hold, not
    # an exact frozen count
    assert panel["period"].nunique() >= 4          # at least May..Aug 2026
    assert {"Belmont Shop", "Gwanda VID"} <= set(panel["branch"])
    assert len(panel) > 2000
    assert (panel["qty"].dtype.kind in "if")
    cov = ms.coverage(panel)
    assert cov["months"] >= 4 and "Belmont Shop" in cov["branches"]


def test_forecast_table(panel):
    periods = sorted(panel["period"].unique())
    history_lbl = [pd.Timestamp(p).strftime("%b %Y") for p in periods[:-1]]
    target_lbl = pd.Timestamp(periods[-1]).strftime("%b %Y")

    fc = dfc.forecast_table(panel)
    assert not fc.empty
    assert {"forecast_qty", "actual_qty", "forecast_error", "low80", "high80",
            "order_up_to", "confidence", "pattern", "forecast_month"} <= set(fc.columns)
    assert (fc["forecast_qty"] >= 0).all()
    assert (fc["low80"] <= fc["high80"]).all()
    assert (fc["order_up_to"] >= fc["forecast_qty"]).all()
    assert set(fc["confidence"]) <= {"High", "Medium", "Low"}
    # forecast target is the latest month, sitting next to its real actuals
    assert fc["forecast_month"].iloc[0] == target_lbl
    assert (fc["actual_qty"] >= 0).all()
    assert (fc["forecast_error"] == fc["forecast_qty"] - fc["actual_qty"]).all()
    # history columns are every month *before* the target, not the target itself
    assert set(history_lbl) <= set(fc.columns)
    assert target_lbl not in fc.columns


def test_backtest_scores_latest_month(panel):
    periods = sorted(panel["period"].unique())
    holdout_period = pd.Timestamp(periods[-1])
    # a mid-month export (e.g. "1 TO 12 SEPTEMBER ... SALES.xlsx") reports only
    # part of the month; scoring a naive "repeat last month" baseline against
    # such a structurally-depressed actual is not a fair forecast-quality
    # comparison, so the skill check only applies once the latest month is
    # fully reported
    holdout_rows = panel[panel["period"] == holdout_period]
    days_in_month = (holdout_period + pd.offsets.MonthEnd(0)).day
    holdout_complete = (holdout_rows["day_to"] >= days_in_month).all() if "day_to" in holdout_rows else True

    train_lbl = [pd.Timestamp(p).strftime("%b %Y") for p in periods[:-1]]
    holdout_lbl = holdout_period.strftime("%b %Y")

    bt = dfc.backtest(panel)
    assert bt["ok"]
    assert bt["train_months"] == train_lbl
    assert bt["holdout_month"] == holdout_lbl
    o = bt["overall"]
    assert o["wape_pct"] is not None and o["bias_pct"] is not None
    if holdout_complete:
        # must not do worse than "repeat last month" on a fully-reported hold-out
        assert o["skill_vs_last_pct"] is not None and o["skill_vs_last_pct"] >= -1


def test_single_month_item_is_damped(panel):
    # an item sold only in the latest month -> 0.6x that qty (regresses toward 0)
    row = dfc._forecast_row([0.0, 0.0, 10.0])
    assert 5 <= row[0] <= 7
    # went quiet last month -> heavily shrunk
    assert dfc._forecast_row([8.0, 6.0, 0.0])[0] < 3


@pytest.fixture()
def web(seeded):
    from wms.api.main import app
    return TestClient(app)


def _login(c, u="analyst", p="wms1234"):
    r = c.post("/login", data={"username": u, "password": p}, follow_redirects=False)
    assert r.status_code == 303
    return c


def test_web_demand_tab_is_the_weekly_forecast(web):
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        pytest.skip("no weekly_sales files present")
    _login(web)
    # bare /analytics now defaults to the Allocation plan, not this tab
    assert "Split by predicted sales" in web.get("/allocation").text

    r = web.get("/allocation?tab=demand")
    assert r.status_code == 200
    assert "Forecast by product" in r.text
    assert 'class="fit"' in r.text
    # weekly per-SKU columns, not monthly forecast/actual/error
    assert ">Next week</th>" in r.text
    for gone in ("<th>Segment</th>", "<th>Method</th>", ">Weeks sold</th>"):
        assert gone not in r.text
    assert "May 2026" not in r.text and ">actual</span>" not in r.text
    # the model / backtest section was removed from the page
    assert "Weekly demand model" not in r.text and "SEG-CHAMP" not in r.text
    assert "Demand forecast" in r.text and "Allocation plan" in r.text
    # removed/unrecognised tabs still resolve (fall through to the
    # allocation-plan default), never 500
    assert web.get("/allocation?tab=stats").status_code == 200

    r2 = web.get("/allocation?tab=demand&q=cable")
    assert r2.status_code == 200
    # data-layer filter: bcode restricts rows to the one branch (the exact
    # code, not a name prefix - "gwanda" alone now also matches Gwanda
    # Thobelani, since branch names are matched by prefix)
    only_gwa = wfc.display_frame(bcode="GWA")
    assert set(only_gwa["Location"]) <= {"Gwanda VID"}


def test_demand_forecast_excel_matches_the_page(web):
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        pytest.skip("no weekly_sales files present")
    _login(web)
    r = web.get("/download/demand-forecast")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    xl = pd.ExcelFile(io.BytesIO(r.content))
    assert xl.sheet_names == ["Forecast by product"]          # one sheet only
    sheet = xl.parse("Forecast by product")
    disp = wfc.display_frame()
    assert list(sheet.columns) == list(disp.columns)
    assert list(sheet.columns) == ["Location", "SKU", "Product", "Next week"]
    assert len(sheet) == len(disp)                            # every row

    # respects the branch filter, same as the page (exact code: "gwanda" alone
    # now also matches Gwanda Thobelani via the branch-name-prefix match)
    only_gwa = web.get("/download/demand-forecast?bcode=GWA")
    g = pd.ExcelFile(io.BytesIO(only_gwa.content)).parse("Forecast by product")
    assert 0 < len(g) < len(sheet)
    assert set(g["Location"]) <= {"Gwanda VID"}
