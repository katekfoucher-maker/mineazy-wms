"""Delivery-note fill analysis + backorder generation from a DN shortfall."""
from wms.analytics import loaders
from wms.analytics import backorders as dn_an
from wms.models import Branch, BackOrder, Product
from wms.services import backorders as dn_svc
from wms.services import backorder_entry as bo_entry


def test_seed_matches_document_26503244(db):
    """requested 41,980 / sent 6,326 -> 35,654 backordered over 74 lines."""
    dl = loaders.dn_lines_df(db)
    s = dn_an.overall_summary(dl)
    assert s["lines"] == 90
    assert s["total_requested"] == 41980
    assert s["total_sent"] == 6326
    assert s["total_backorder_qty"] == 35654
    assert s["backordered_lines"] == 74
    assert (s["lines_full"], s["lines_partial"], s["lines_nil"]) == (16, 29, 45)


def test_delivery_note_shortfall_creates_one_back_order(db):
    bid = db.query(Branch).filter(Branch.code == "GWA").first().id
    dn = dn_svc.enter_delivery_note(db, branch_id=bid, lines=[
        {"sku": "SFC1269", "requested_qty": 100, "sent_qty": 40},   # short 60
        {"sku": "SFC1274", "requested_qty": 50, "sent_qty": 0},     # short 50 (blank == 0)
        {"sku": "SFC1276", "requested_qty": 10, "sent_qty": 10},    # fully sent
    ], user_id=1)

    bo = (db.query(BackOrder)
          .filter(BackOrder.source == "DELIVERY_NOTE", BackOrder.source_ref == dn.dn_no)
          .one())
    assert bo.stage == "OPEN" and bo.status == "OPEN"
    by_sku = {i.sku: i for i in bo.items}
    assert set(by_sku) == {"SFC1269", "SFC1274"}          # only the short lines
    assert by_sku["SFC1269"].qty_ordered == 60
    assert by_sku["SFC1274"].qty_ordered == 50
    # first event is the OPEN creation
    assert bo.events[0].to_stage == "OPEN"


def test_dn_by_branch_fill_analysis(db):
    dl = loaders.dn_lines_df(db)
    bb = dn_an.by_branch(dl).set_index("branch")
    assert "Belmont Shop" in bb.index
    assert bb.loc["Belmont Shop", "backorder_qty"] == 35654
    assert 0 < bb.loc["Belmont Shop", "fill_rate_qty"] < 1


def test_manual_back_order_entry(db):
    bid = db.query(Branch).filter(Branch.code == "ES").first().id
    bo = bo_entry.create_back_order(db, branch_id=bid, priority="HIGH",
                                    items=[{"sku": "SFC1269", "qty": 25},
                                           {"sku": "SFC1270", "qty": 10}], user_id=1)
    assert bo.stage == "OPEN" and bo.qty_ordered == 35
    assert {i.sku for i in bo.items} == {"SFC1269", "SFC1270"}


def test_unknown_sku_is_auto_registered(db):
    import pytest
    from wms.errors import WMSError
    from wms.services.catalogue import resolve_product

    assert db.query(Product).filter(Product.sku == "MEIW1254").first() is None

    p = resolve_product(db, "MEIW1254", name="MYSTERY WIDGET", user_id=1)
    assert p.id and p.sku == "MEIW1254"
    assert p.name == "MYSTERY WIDGET" and float(p.unit_price or 0) == 0

    # blank name -> fall back to the SKU; existing SKU -> same row, not a duplicate
    q = resolve_product(db, "ZZBRANDNEW01", user_id=1)
    assert q.name == "ZZBRANDNEW01"
    assert resolve_product(db, "MEIW1254").id == p.id

    with pytest.raises(WMSError):          # a missing numeric id is still an error
        resolve_product(db, 999999)

    db.rollback()                          # don't leak the new rows into other tests
