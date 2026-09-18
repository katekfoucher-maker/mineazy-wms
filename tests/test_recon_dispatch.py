"""Recon: stock leaving the warehouse for a branch - a pure stock movement
(warehouse -> branch), no requested-vs-sent split and no back order, unlike
the older DeliveryNote/BackOrder flow this leaves untouched."""
from datetime import date

import pytest

from wms.errors import WMSError
from wms.models import Branch, DispatchOrder, Product, StockOnHand
from wms.services import dispatch as dispatch_svc
from wms.services import receiving as recv_svc


def _dc(db):
    return db.query(Branch).filter(Branch.code == "DC").first()


def _bm(db):
    return db.query(Branch).filter(Branch.code == "BM").first()


def _cleanup(db, do_nos, ro_nos, sku):
    for do_no in do_nos:
        row = db.query(DispatchOrder).filter(DispatchOrder.do_no == do_no).first()
        if row:
            db.delete(row)
    from wms.models import ReceivingOrder
    for ro_no in ro_nos:
        row = db.query(ReceivingOrder).filter(ReceivingOrder.ro_no == ro_no).first()
        if row:
            db.delete(row)
    db.query(StockOnHand).filter(StockOnHand.sku == sku).delete()
    db.query(Product).filter(Product.sku == sku).delete()
    db.commit()


def test_dispatch_moves_stock_from_warehouse_to_branch(db):
    dc, bm = _dc(db), _bm(db)
    sku = "ZZDISPTEST1"
    try:
        recv_svc.enter_receiving_order(
            db, branch_id=dc.id, doc_no="RO-DISP-1", doc_date=date(2026, 1, 1),
            lines=[{"sku": sku, "received_qty": 100}], user_id=1)

        do, warnings = dispatch_svc.enter_dispatch_order(
            db, branch_id=bm.id, doc_no="DO-DISP-1", doc_date=date(2026, 1, 2),
            lines=[{"sku": sku, "dispatched_qty": 40}], user_id=1)
        assert do.do_no == "DO-DISP-1" and do.total_dispatched == 40
        assert warnings == []

        dc_soh = db.query(StockOnHand).filter(StockOnHand.branch_id == dc.id,
                                               StockOnHand.sku == sku).first()
        bm_soh = db.query(StockOnHand).filter(StockOnHand.branch_id == bm.id,
                                              StockOnHand.sku == sku).first()
        assert dc_soh.qty_on_hand == 60          # 100 received - 40 dispatched
        assert bm_soh.qty_on_hand == 40
    finally:
        _cleanup(db, ["DO-DISP-1"], ["RO-DISP-1"], sku)


def test_dispatch_caps_at_warehouse_stock_never_goes_negative(db):
    """Asking to dispatch more than the warehouse has must cap the line (and
    warn), not push the warehouse balance below zero - the whole point is an
    accurate warehouse inventory."""
    dc, bm = _dc(db), _bm(db)
    sku = "ZZDISPTEST2"
    try:
        recv_svc.enter_receiving_order(
            db, branch_id=dc.id, doc_no="RO-DISP-2", doc_date=date(2026, 1, 1),
            lines=[{"sku": sku, "received_qty": 30}], user_id=1)

        do, warnings = dispatch_svc.enter_dispatch_order(
            db, branch_id=bm.id, doc_no="DO-DISP-2", doc_date=date(2026, 1, 2),
            lines=[{"sku": sku, "dispatched_qty": 500}], user_id=1)
        assert do.total_dispatched == 30                # capped, not 500
        assert any("capped from 500 to 30" in w for w in warnings)

        dc_soh = db.query(StockOnHand).filter(StockOnHand.branch_id == dc.id,
                                               StockOnHand.sku == sku).first()
        assert dc_soh.qty_on_hand == 0                   # never negative
    finally:
        _cleanup(db, ["DO-DISP-2"], ["RO-DISP-2"], sku)


def test_dispatch_with_nothing_on_hand_is_rejected_and_rolled_back(db):
    dc, bm = _dc(db), _bm(db)
    sku = "ZZDISPTEST3"
    try:
        with pytest.raises(WMSError, match="Nothing could be dispatched"):
            dispatch_svc.enter_dispatch_order(
                db, branch_id=bm.id, doc_no="DO-DISP-3", doc_date=date(2026, 1, 2),
                lines=[{"sku": sku, "dispatched_qty": 10}], user_id=1)
        assert db.query(DispatchOrder).filter(
            DispatchOrder.do_no == "DO-DISP-3").first() is None
    finally:
        _cleanup(db, ["DO-DISP-3"], [], sku)


def test_dispatch_to_the_warehouse_itself_is_rejected(db):
    dc = _dc(db)
    with pytest.raises(WMSError, match="can't be the warehouse itself"):
        dispatch_svc.enter_dispatch_order(
            db, branch_id=dc.id, doc_no="DO-DISP-SELF",
            lines=[{"sku": "ANYTHING", "dispatched_qty": 1}], user_id=1)


def test_reverse_dispatch_returns_units_to_the_warehouse(db):
    dc, bm = _dc(db), _bm(db)
    sku = "ZZDISPTEST4"
    try:
        recv_svc.enter_receiving_order(
            db, branch_id=dc.id, doc_no="RO-DISP-4", doc_date=date(2026, 1, 1),
            lines=[{"sku": sku, "received_qty": 50}], user_id=1)
        dispatch_svc.enter_dispatch_order(
            db, branch_id=bm.id, doc_no="DO-DISP-4", doc_date=date(2026, 1, 2),
            lines=[{"sku": sku, "dispatched_qty": 20}], user_id=1)

        res = dispatch_svc.reverse_dispatch_order(db, "DO-DISP-4", user_id=1)
        assert res["units_pulled"] == 20
        assert db.query(DispatchOrder).filter(
            DispatchOrder.do_no == "DO-DISP-4").first() is None

        dc_soh = db.query(StockOnHand).filter(StockOnHand.branch_id == dc.id,
                                               StockOnHand.sku == sku).first()
        bm_soh = db.query(StockOnHand).filter(StockOnHand.branch_id == bm.id,
                                              StockOnHand.sku == sku).first()
        assert dc_soh.qty_on_hand == 50            # back to what was received
        assert bm_soh.qty_on_hand == 0
    finally:
        _cleanup(db, ["DO-DISP-4"], ["RO-DISP-4"], sku)


def test_reverse_unknown_dispatch_order_raises(db):
    with pytest.raises(WMSError):
        dispatch_svc.reverse_dispatch_order(db, "DO-DOES-NOT-EXIST", user_id=1)
