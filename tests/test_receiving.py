"""Receiving Orders: stock arriving into the warehouse (mirrors Recon's
dispatch flow, but only ever adds to stock - no back order / shortfall)."""
from datetime import date

import pytest

from wms.errors import WMSError
from wms.models import Branch, Product, ReceivingOrder, StockOnHand
from wms.services import receiving as recv_svc
from wms.services import stock as stock_svc


def _dc(db):
    return db.query(Branch).filter(Branch.code == "DC").first()


def _cleanup(db, ro_nos, sku):
    for ro_no in ro_nos:
        row = db.query(ReceivingOrder).filter(ReceivingOrder.ro_no == ro_no).first()
        if row:
            db.delete(row)
    dc = _dc(db)
    if dc:
        db.query(StockOnHand).filter(StockOnHand.branch_id == dc.id,
                                     StockOnHand.sku == sku).delete()
    db.query(Product).filter(Product.sku == sku).delete()
    db.commit()


def test_enter_receiving_order_adds_to_warehouse_stock(db):
    dc = _dc(db)
    assert dc is not None, "the DISTRIBUTION CENTER branch must exist"
    sku = "ZZRECVTEST1"
    try:
        before = stock_svc.levels_df(db)
        before_qty = 0
        if not before.empty:
            row = before[(before.branch_code == "DC") & (before.sku == sku)]
            before_qty = int(row["on_hand"].iloc[0]) if not row.empty else 0
        assert before_qty == 0

        ro = recv_svc.enter_receiving_order(
            db, branch_id=dc.id, doc_no="RO-TEST-1", doc_date=date(2026, 1, 5),
            supplier="Test Supplier", lines=[{"sku": sku, "description": "Test widget",
                                              "received_qty": 40}],
            user_id=1)
        assert ro.ro_no == "RO-TEST-1"
        assert ro.total_received == 40
        assert ro.supplier == "Test Supplier"

        soh = (db.query(StockOnHand)
              .filter(StockOnHand.branch_id == dc.id, StockOnHand.sku == sku).first())
        assert soh is not None and soh.qty_on_hand == 40

        # a second receipt for the same SKU adds on top, it does not replace
        ro2 = recv_svc.enter_receiving_order(
            db, branch_id=dc.id, doc_no="RO-TEST-2", doc_date=date(2026, 1, 6),
            lines=[{"sku": sku, "received_qty": 15}], user_id=1)
        db.refresh(soh)
        assert soh.qty_on_hand == 55
        assert ro2.ro_no == "RO-TEST-2"
    finally:
        _cleanup(db, ["RO-TEST-1", "RO-TEST-2"], sku)


def test_enter_receiving_order_rejects_a_duplicate_reference(db):
    dc = _dc(db)
    sku = "ZZRECVTEST2"
    try:
        recv_svc.enter_receiving_order(
            db, branch_id=dc.id, doc_no="RO-DUP-1", doc_date=date(2026, 1, 5),
            lines=[{"sku": sku, "received_qty": 5}], user_id=1)
        with pytest.raises(WMSError):
            recv_svc.enter_receiving_order(
                db, branch_id=dc.id, doc_no="RO-DUP-1", doc_date=date(2026, 1, 6),
                lines=[{"sku": sku, "received_qty": 3}], user_id=1)
    finally:
        _cleanup(db, ["RO-DUP-1"], sku)


def test_reverse_receiving_order_pulls_stock_back_and_deletes_it(db):
    dc = _dc(db)
    sku = "ZZRECVTEST3"
    try:
        recv_svc.enter_receiving_order(
            db, branch_id=dc.id, doc_no="RO-REV-1", doc_date=date(2026, 1, 5),
            lines=[{"sku": sku, "received_qty": 30}], user_id=1)
        soh = (db.query(StockOnHand)
              .filter(StockOnHand.branch_id == dc.id, StockOnHand.sku == sku).first())
        assert soh.qty_on_hand == 30

        res = recv_svc.reverse_receiving_order(db, "RO-REV-1", user_id=1)
        assert res["units_pulled"] == 30
        assert db.query(ReceivingOrder).filter(
            ReceivingOrder.ro_no == "RO-REV-1").first() is None
        db.refresh(soh)
        assert soh.qty_on_hand == 0
    finally:
        _cleanup(db, ["RO-REV-1"], sku)


def test_reverse_receiving_order_unknown_reference_raises(db):
    with pytest.raises(WMSError):
        recv_svc.reverse_receiving_order(db, "RO-DOES-NOT-EXIST", user_id=1)
