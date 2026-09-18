"""Backorder processing flow: 3-stage machine, analytics, export."""
import pytest

from wms.analytics import loaders, backorder_flow as bof
from wms.enums import STAGE_NEXT, BackOrderStage
from wms.errors import WMSError
from wms.models import Branch, BackOrder, BackOrderEvent
from wms.services import backorder_entry as bo_entry
from wms.services import backorder_stages as bo_stage


def _bo(db, code="GWA"):
    bid = db.query(Branch).filter(Branch.code == code).first().id
    return bo_entry.create_back_order(db, branch_id=bid,
                                      items=[{"sku": "SFC1269", "qty": 100}], user_id=1)


def test_stage_model_is_open_closed():
    assert [s.value for s in BackOrderStage] == ["OPEN", "CLOSED"]
    assert STAGE_NEXT[BackOrderStage.OPEN] == {BackOrderStage.CLOSED}
    assert STAGE_NEXT[BackOrderStage.CLOSED] == set()


def test_forward_only_transitions_are_enforced(db):
    bo = _bo(db)
    assert bo.stage == "OPEN"
    bo_stage.advance(db, bo_no=bo.bo_no, to_stage="CLOSED", user_id=1)
    assert bo_entry.get(db, bo.bo_no).stage == "CLOSED"
    with pytest.raises(WMSError):                       # already terminal
        bo_stage.advance(db, bo_no=bo.bo_no, to_stage="CLOSED", user_id=1)


def test_close_records_fulfilment(db):
    bo = _bo(db, "MP")
    item_id = bo.items[0].id
    bo_stage.advance(db, bo_no=bo.bo_no, to_stage="CLOSED", user_id=1,
                     items={item_id: 80})              # only 80 of 100 fulfilled

    bo = bo_entry.get(db, bo.bo_no)
    assert bo.stage == "CLOSED" and bo.status == "CLOSED"
    assert bo.items[0].qty_fulfilled == 80
    assert bo.fulfil_pct == 0.8
    # one event for the initial OPEN + one for the close
    assert db.query(BackOrderEvent).filter(BackOrderEvent.back_order_id == bo.id).count() == 2


def _flow_history(db):
    """Build a handful of back orders across a few branches, some closed -
    the seed no longer ships fabricated flow history."""
    made = []
    for code in ("GWA", "MP", "GW"):
        for _ in range(3):
            made.append(_bo(db, code))
    for i, bo in enumerate(made):
        if i % 3 == 1:
            bo_stage.advance(db, bo_no=bo.bo_no, to_stage="CLOSED", user_id=1)
    return made


def test_flow_analytics_shapes(db):
    _flow_history(db)
    hdr = loaders.back_orders_df(db)
    items = loaders.back_order_items_df(db)
    events = loaders.back_order_events_df(db)
    fm = bof.fulfilment_metrics(hdr, items)
    assert fm["back_orders"] >= 9
    assert 0 <= fm["fill_rate_qty"] <= 1
    assert not bof.stage_funnel(hdr).empty
    assert not bof.cycle_times(events).empty
    assert bof.bottleneck_stage(events) is not None
    bb = bof.by_branch(hdr, items)
    assert "fill_rate_qty" in bb.columns and len(bb) >= 3
    vs = bof.branch_backorders_vs_sales(db)
    assert "demand_met_pct" in vs.columns
    tr = bof.trend(hdr, events)
    assert {"raised", "closed", "outstanding"} <= set(tr.columns)


def test_product_performance_separates_supply_from_demand(db):
    pp = bof.product_performance(db)
    assert not pp.empty
    assert {"sku", "sold_qty", "fill_rate", "unmet_demand_ratio", "stockout_line_pct",
            "recovery_rate", "demand_captured_pct", "lost_sales_value",
            "verdict"} <= set(pp.columns)
    assert set(pp["verdict"]) <= {"Supply-constrained", "Chronic shortage",
                                  "Genuinely low demand", "Healthy", "No demand signal"}
    # a product requested but never sent -> flagged as a supply problem, not low demand
    short = pp[pp["sku"] == "KJ1262"].iloc[0]
    assert short["sent_qty"] == 0 and short["fill_rate"] == 0
    assert short["verdict"] in ("Supply-constrained", "Chronic shortage")

    k = bof.product_performance_kpis(pp)
    assert k["products"] == len(pp)
    assert k["supply_constrained"] >= 1 and k["lost_sales_value"] > 0


def test_flow_analysis_api(client):
    r = client.get("/api/outbound/back-orders-analysis")
    assert r.status_code == 200
    body = r.json()
    assert "fulfilment" in body and "stage_funnel" in body and "trend" in body


def test_flow_workbook(db):
    from wms.exports import excel
    p = excel.backorder_flow_workbook(db)
    assert p.exists() and p.stat().st_size > 5000


def test_low_sales_diagnosis_separates_supply_from_demand(db, monkeypatch):
    """No false 'supply problem' when stock is on hand and nobody back-ordered it."""
    import pandas as pd
    from wms.analytics import weekly_forecast as wfc

    fake_state = pd.DataFrame([
        # predicted 10/wk, selling 1/wk -> "low"; stock on shelf, no back order
        {"branch": "GWA", "branch_name": "Gwanda VID", "sku": "AAA", "item": "Widget A",
         "segment": "smooth", "method": "sba", "weeks_sold": 8,
         "weekly_demand": 10, "recent_sales": 1},
        # predicted 10/wk, selling 0 -> "low"; a branch back-ordered it
        {"branch": "GWA", "branch_name": "Gwanda VID", "sku": "BBB", "item": "Widget B",
         "segment": "smooth", "method": "sba", "weeks_sold": 6,
         "weekly_demand": 10, "recent_sales": 0},
        # selling at forecast -> not listed
        {"branch": "GWA", "branch_name": "Gwanda VID", "sku": "CCC", "item": "Widget C",
         "segment": "smooth", "method": "sba", "weeks_sold": 9,
         "weekly_demand": 5, "recent_sales": 5},
    ])
    monkeypatch.setattr(wfc, "cached_run", lambda: {"state": fake_state})

    gwa = db.query(Branch).filter(Branch.code == "GWA").first().id
    bo_entry.create_back_order(db, branch_id=gwa, items=[{"sku": "BBB", "qty": 40}],
                               user_id=1)
    from wms.services import stock as stock_svc
    stock_svc.set_branch_stock(db, branch_id=gwa,
                               items=[{"sku": "AAA", "qty": 25}], user_id=1)
    try:
        d = bof.low_sales_diagnosis(db).set_index("sku")
        assert "CCC" not in d.index                       # selling on target -> not flagged
        assert d.loc["AAA", "cause"] == "Low demand, stock available"   # NOT a false alarm
        assert d.loc["BBB", "cause"] == "Stockout, back ordered"
        assert d.loc["BBB", "backordered"] == 40
    finally:
        from wms.models import BackOrder, StockOnHand, Product
        db.query(StockOnHand).filter(StockOnHand.branch_id == gwa).delete()
        for b in db.query(BackOrder).filter(BackOrder.branch_id == gwa).all():
            db.delete(b)
        db.query(Product).filter(Product.sku.in_(["AAA", "BBB", "CCC"])).delete(
            synchronize_session=False)
        db.commit()
