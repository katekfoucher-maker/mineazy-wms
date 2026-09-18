"""Dispatch flow: one OPEN back order per branch (loads every dispatch until
closed) + stock-on-hand + full reversal."""
from datetime import date

from wms.models import Branch, BackOrder, DeliveryNote, SalesRecord, StockOnHand
from wms.services import backorders as dn_svc
from wms.services import backorder_entry as bo_entry
from wms.services import backorder_stages as bo_stage
from wms.services import stock as stock_svc


def _dispatch(db, branch_id, doc_no, d, lines):
    dn = dn_svc.enter_delivery_note(
        db, branch_id=branch_id, doc_no=doc_no, doc_date=d, lines=lines,
        raise_backorder=False, commit=False, user_id=1)
    stock_svc.add_stock(db, branch_id=branch_id,
                        items=[{"sku": ln["sku"], "qty": ln.get("sent_qty", 0)}
                               for ln in lines if ln.get("sent_qty", 0) > 0],
                        user_id=1, commit=False)
    bo = bo_entry.attach_dispatch(db, dn, user_id=1)
    db.commit()
    return bo


def _cleanup(db, code):
    bid = db.query(Branch).filter(Branch.code == code).first().id
    db.query(StockOnHand).filter(StockOnHand.branch_id == bid).delete()
    db.query(SalesRecord).filter(SalesRecord.branch_id == bid,
                                 SalesRecord.source == "DELIVERY").delete()
    for m in (BackOrder, DeliveryNote):
        for row in db.query(m).filter(m.branch_id == bid).all():
            db.delete(row)
    db.commit()


def test_every_dispatch_loads_onto_one_open_backorder(db):
    es2 = db.query(Branch).filter(Branch.code == "ES2").first()
    try:
        bo1 = _dispatch(db, es2.id, "D-1", date(2026, 3, 2), [
            {"sku": "SFC1269", "requested_qty": 20, "sent_qty": 5}])   # short 15
        assert bo1.stage == "OPEN" and bo1.status == "OPEN"
        assert bo1.bo_no == "BO-ES2-2026W10"

        # a dispatch two weeks later still merges (not weekly any more)
        bo2 = _dispatch(db, es2.id, "D-2", date(2026, 3, 18), [
            {"sku": "SFC1269", "requested_qty": 10, "sent_qty": 0},    # +10
            {"sku": "SFC1270", "requested_qty": 4, "sent_qty": 1}])    # new line, short 3
        assert bo2.id == bo1.id
        assert {i.sku: i.qty_ordered for i in bo2.items} == {"SFC1269": 25, "SFC1270": 3}
        assert set(bo2.source_ref.split(",")) == {"D-1", "D-2"}

        # close it -> next dispatch opens a fresh back order
        bo_stage.advance(db, bo_no=bo1.bo_no, to_stage="CLOSED", user_id=1)
        bo3 = _dispatch(db, es2.id, "D-3", date(2026, 3, 20), [
            {"sku": "SFC1269", "requested_qty": 5, "sent_qty": 0}])
        assert bo3.id != bo1.id and bo3.status == "OPEN"
        assert bo3.bo_no == "BO-ES2-2026W12"

        lv = stock_svc.levels_df(db).set_index(["branch_code", "sku"])["on_hand"]
        assert lv.get(("ES2", "SFC1269")) == 5        # 5 + 0 + 0 dispatched
        assert lv.get(("ES2", "SFC1270")) == 1
    finally:
        _cleanup(db, "ES2")


def test_reverse_dispatch_undoes_everything(db):
    tg = db.query(Branch).filter(Branch.code == "TG").first()
    try:
        bo = _dispatch(db, tg.id, "R-1", date(2026, 4, 6), [
            {"sku": "SFC1269", "requested_qty": 30, "sent_qty": 20},   # short 10
            {"sku": "SFC1270", "requested_qty": 5, "sent_qty": 5}])    # fully sent
        _dispatch(db, tg.id, "R-2", date(2026, 4, 8), [
            {"sku": "SFC1269", "requested_qty": 12, "sent_qty": 4}])   # short 8 -> merges

        lv = stock_svc.levels_df(db).set_index(["branch_code", "sku"])["on_hand"]
        assert lv.get(("TG", "SFC1269")) == 24                        # 20 + 4
        assert {i.sku: i.qty_ordered for i in bo.items} == {"SFC1269": 18}
        assert db.query(SalesRecord).filter(SalesRecord.source_ref == "R-1").count() == 2

        res = bo_entry.reverse_dispatch(db, "R-2", user_id=1)
        assert res["units_pulled"] == 4
        lv = stock_svc.levels_df(db).set_index(["branch_code", "sku"])["on_hand"]
        assert lv.get(("TG", "SFC1269")) == 20                        # R-2's 4 pulled back
        bo = db.query(BackOrder).filter(BackOrder.bo_no == bo.bo_no).first()
        assert {i.sku: i.qty_ordered for i in bo.items} == {"SFC1269": 10}   # back to R-1 only
        assert bo.source_ref == "R-1"
        assert db.query(DeliveryNote).filter(DeliveryNote.dn_no == "R-2").first() is None

        # reversing the last dispatch deletes the (now empty) back order
        bo_no = bo.bo_no
        res = bo_entry.reverse_dispatch(db, "R-1", user_id=1)
        assert res["units_pulled"] == 25 and res["sales_deleted"] == 2   # 20 + 5 sent
        assert db.query(BackOrder).filter(BackOrder.bo_no == bo_no).first() is None
        tg_lv = stock_svc.levels_df(db).query("branch_code == 'TG'")
        assert (tg_lv["on_hand"] == 0).all()          # every dispatched unit pulled back
    finally:
        _cleanup(db, "TG")


def test_reverse_refused_once_backorder_closed(db):
    import pytest
    from wms.errors import WMSError
    mp = db.query(Branch).filter(Branch.code == "MP").first()
    try:
        bo = _dispatch(db, mp.id, "C-1", date(2026, 5, 4), [
            {"sku": "SFC1269", "requested_qty": 9, "sent_qty": 2}])
        bo_stage.advance(db, bo_no=bo.bo_no, to_stage="CLOSED", user_id=1)
        with pytest.raises(WMSError):
            bo_entry.reverse_dispatch(db, "C-1", user_id=1)
    finally:
        _cleanup(db, "MP")


def test_dispatch_updates_allocation_plan_on_hand(db):
    from wms.analytics import allocation, weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")

    import math
    from wms.models import Product
    st = wfc.cached_run()["state"]
    row = st[(st["branch"] == "GWA") & (st["weekly_demand"] >= 5)].iloc[0]
    sku, demand = row["sku"], int(row["weekly_demand"])
    target = math.ceil(demand * 10 / 7)                       # 7d week + 3d transit
    gwa = db.query(Branch).filter(Branch.code == "GWA").first()
    pre_sku = db.query(Product).filter(Product.sku == sku).first() is not None
    try:
        _dispatch(db, gwa.id, "D-ALLOC", date(2026, 6, 1), [
            {"sku": sku, "requested_qty": target + 10, "sent_qty": 3}])
        after = allocation.weekly_allocation_plan(db, branch_code="gwanda")
        arow = after[after["sku"] == sku].iloc[0]
        assert int(arow["on_hand"]) == 3
        assert int(arow["target"]) == target
        assert int(arow["to_transport"]) == max(0, target - 3)
    finally:
        _cleanup(db, "GWA")
        if not pre_sku:                                       # don't leak an auto-created product
            db.query(Product).filter(Product.sku == sku).delete()
            db.commit()
