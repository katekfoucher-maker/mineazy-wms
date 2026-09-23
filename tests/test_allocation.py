"""Unit tests for the sales-capped split helper used by 'Split by predicted
sales' - the part that stops a branch hoarding stock it cannot move."""
import pytest

from wms.analytics.allocation import _split_capped


def test_levels_df_for_allocation_zeroes_belmont_but_not_plain_levels_df(seeded, monkeypatch):
    """Belmont's stock-on-hand isn't reliable enough to trust for allocation
    (see UNRELIABLE_FOR_ALLOCATION) - the allocation-flavoured lookup must
    zero it, while the plain lookup (Inventory page, coverage()) keeps
    showing the real figure. Tested with the network-wide
    allocation_use_inventory switch explicitly back on, since the current
    default (off) would zero every branch and make this Belmont-specific
    check meaningless - see the dedicated test for that switch below."""
    from wms.config import get_settings
    from wms.db import get_session
    from wms.models import Branch, StockOnHand
    from wms.services import stock as stock_svc

    monkeypatch.setattr(get_settings(), "allocation_use_inventory", True)
    db = next(get_session())
    bm = db.query(Branch).filter(Branch.code == "BM").first()
    other = db.query(Branch).filter(Branch.code != "BM").first()
    for b in (bm, other):
        db.query(StockOnHand).filter(StockOnHand.branch_id == b.id,
                                     StockOnHand.sku == "TEST-ALLOC-SKU").delete()
    db.add(StockOnHand(branch_id=bm.id, sku="TEST-ALLOC-SKU", qty_on_hand=42))
    db.add(StockOnHand(branch_id=other.id, sku="TEST-ALLOC-SKU", qty_on_hand=17))
    db.commit()

    real = stock_svc.levels_df(db)
    alloc = stock_svc.levels_df_for_allocation(db)

    real_row = real[(real.branch_code == "BM") & (real.sku == "TEST-ALLOC-SKU")]
    alloc_row = alloc[(alloc.branch_code == "BM") & (alloc.sku == "TEST-ALLOC-SKU")]
    assert int(real_row["on_hand"].iloc[0]) == 42
    assert int(alloc_row["on_hand"].iloc[0]) == 0

    # an unrelated branch's figure is untouched in both
    real_other = real[(real.branch_code == other.code) & (real.sku == "TEST-ALLOC-SKU")]
    alloc_other = alloc[(alloc.branch_code == other.code) & (alloc.sku == "TEST-ALLOC-SKU")]
    assert int(real_other["on_hand"].iloc[0]) == 17
    assert int(alloc_other["on_hand"].iloc[0]) == 17

    for b in (bm, other):
        db.query(StockOnHand).filter(StockOnHand.branch_id == b.id,
                                     StockOnHand.sku == "TEST-ALLOC-SKU").delete()
    db.commit()


def test_allocation_use_inventory_off_zeroes_every_branch(seeded):
    """The current default: stock-on-hand isn't trusted network-wide, so
    allocation runs on sales/demand alone - EVERY branch's on-hand reads as
    0 for allocation math, not just Belmont's. The plain lookup (Inventory
    page) is unaffected."""
    from wms.config import get_settings
    from wms.db import get_session
    from wms.models import Branch, StockOnHand
    from wms.services import stock as stock_svc

    assert get_settings().allocation_use_inventory is False    # the current default

    db = next(get_session())
    a = db.query(Branch).filter(Branch.code == "BM").first()
    b = db.query(Branch).filter(Branch.code != "BM").first()
    for br in (a, b):
        db.query(StockOnHand).filter(StockOnHand.branch_id == br.id,
                                     StockOnHand.sku == "TEST-NOINV-SKU").delete()
    db.add(StockOnHand(branch_id=a.id, sku="TEST-NOINV-SKU", qty_on_hand=42))
    db.add(StockOnHand(branch_id=b.id, sku="TEST-NOINV-SKU", qty_on_hand=17))
    db.commit()

    real = stock_svc.levels_df(db)
    alloc = stock_svc.levels_df_for_allocation(db)
    for code, want_real in ((a.code, 42), (b.code, 17)):
        r = real[(real.branch_code == code) & (real.sku == "TEST-NOINV-SKU")]
        al = alloc[(alloc.branch_code == code) & (alloc.sku == "TEST-NOINV-SKU")]
        assert int(r["on_hand"].iloc[0]) == want_real     # plain lookup: real figure
        assert int(al["on_hand"].iloc[0]) == 0             # allocation lookup: zeroed

    for br in (a, b):
        db.query(StockOnHand).filter(StockOnHand.branch_id == br.id,
                                     StockOnHand.sku == "TEST-NOINV-SKU").delete()
    db.commit()


def test_split_capped_is_proportional_when_under_caps():
    # caps are far above what's available -> pure proportional split by weight
    got = _split_capped(100, {"A": 9.0, "B": 1.0}, {"A": 1000, "B": 1000})
    assert got == {"A": 90, "B": 10}
    assert sum(got.values()) == 100


def test_split_capped_never_exceeds_a_cap_and_spills_to_others():
    # A would take the lot by weight, but its cap is 3 -> the rest goes to B
    got = _split_capped(20, {"A": 9.0, "B": 1.0}, {"A": 3, "B": 50})
    assert got["A"] == 3
    assert got["B"] == 17
    assert sum(got.values()) == 20


def test_split_capped_holds_back_when_every_branch_is_capped():
    # only 6 units of room in total -> 6 handed out, 4 left for the caller to hold
    got = _split_capped(10, {"A": 2.0, "B": 1.0}, {"A": 4, "B": 2})
    assert got == {"A": 4, "B": 2}
    assert sum(got.values()) == 6          # caller holds the remaining 4


def test_split_capped_zero_weights_falls_back_to_even():
    got = _split_capped(4, {"A": 0.0, "B": 0.0}, {"A": 10, "B": 10})
    assert got["A"] == 2 and got["B"] == 2


def test_allocation_reports_every_considered_branch_even_at_zero_alloc(seeded):
    """A branch that already has ample on-hand stock for a product - so it
    needs none of the split - must still be reported at 0 units, not silently
    dropped from the results. Dropping it reads as a missing/buggy branch,
    which is exactly what surfaced once the split started covering many
    branches at once instead of just one or two well-stocked ones."""
    from wms.analytics import weekly_forecast as wfc
    from wms.analytics import allocation
    from wms.db import get_session
    from wms.services import stock as stock_svc

    if not wfc.has_data():
        pytest.skip("no weekly_sales files present")
    db = next(get_session())
    st = wfc.cached_run()["state"]
    inv = stock_svc.levels_df(db)
    if inv.empty:
        pytest.skip("no inventory on hand in the sample data")

    st = st.assign(_sku=st["sku"].str.upper())
    inv = inv.assign(_sku=inv["sku"].str.upper())
    merged = st.merge(inv, left_on=["branch", "_sku"], right_on=["branch_code", "_sku"])
    # a branch sitting on more than 10 weeks of its own demand for this SKU -
    # a split should not push it any more
    over = merged[(merged["weekly_demand"] > 0) &
                  (merged["on_hand"] > 10 * merged["weekly_demand"])]
    if over.empty:
        pytest.skip("no over-stocked branch/SKU pair in the sample data")
    row = over.iloc[0]
    sku, branch_name = row["sku_x"], row["branch_name"]

    res = allocation.allocate_by_forecast(db, sku=sku, qty=30, branch_codes=None)
    match = next((a for a in res["allocations"] if a["branch"] == branch_name), None)
    assert match is not None, f"{branch_name} should still be reported for {sku}"
    assert match["allocated"] == 0
    assert match["kind"] == "held"


def test_allocation_probes_an_unstocked_branch_sized_by_branch_and_product(seeded):
    """A branch with no sales history for a product, and none on hand, still
    gets a real test quantity - not a flat token unit and not "no demand",
    since an untested branch is simply unproven, not proven-uninterested.
    The probe is sized off how well the product already performs at the
    branches that DO carry it (yield per unit of branch size) times the
    untested branch's OWN size (its total weekly demand across every
    product) - a branch that generally sells a lot gets a bigger probe than
    a branch that generally sells little, for the exact same product."""
    from wms.analytics import weekly_forecast as wfc
    from wms.analytics import allocation
    from wms.db import get_session
    from wms.services import stock as stock_svc

    if not wfc.has_data():
        pytest.skip("no weekly_sales files present")
    db = next(get_session())
    st = wfc.cached_run()["state"]
    inv = stock_svc.levels_df(db)
    name_by_code = dict(st[["branch", "branch_name"]].drop_duplicates().values)
    all_codes = set(name_by_code)
    if len(all_codes) < 2:
        pytest.skip("not enough branches in the sample data")
    branch_size = {bc.upper(): tot for bc, tot in
                  st.groupby(st["branch"].str.upper())["weekly_demand"].sum().items()}

    def _on_hand(code, sku):
        if inv.empty:
            return 0
        sub = inv[(inv["branch_code"].str.upper() == code) &
                  (inv["sku"].str.lower() == str(sku).lower())]
        return int(sub["on_hand"].sum()) if not sub.empty else 0

    target_sku = target_code = None
    for sku, grp in st.groupby("sku"):
        if grp["weekly_demand"].sum() <= 0:
            continue                                    # only real movers here
        sellers = set(grp.loc[grp["weekly_demand"] > 0, "branch"].str.upper())
        for code in sorted(all_codes - sellers, key=lambda c: -branch_size.get(c, 0.0)):
            if _on_hand(code, sku) == 0:
                target_sku, target_code = sku, code
                break
        if target_sku:
            break
    if target_sku is None:
        pytest.skip("no real-moving SKU with an un-stocked branch gap in the sample data")

    res = allocation.allocate_by_forecast(db, sku=target_sku, qty=100, branch_codes=None)
    match = next((a for a in res["allocations"] if a["branch"] == name_by_code[target_code]), None)
    assert match is not None
    assert match["kind"] == "probe"
    assert match["allocated"] >= 1


def test_allocation_probe_is_not_labelled_no_demand(seeded):
    """A branch that's never sold a product is unproven, not proven to have
    no demand for it - there is no 'none'/'no demand' kind any more."""
    from wms.analytics import weekly_forecast as wfc
    from wms.analytics import allocation
    from wms.db import get_session

    if not wfc.has_data():
        pytest.skip("no weekly_sales files present")
    db = next(get_session())
    st = wfc.cached_run()["state"]
    sku = None
    for s, grp in st.groupby("sku"):
        if grp["weekly_demand"].sum() > 0:
            sku = s
            break
    if sku is None:
        pytest.skip("no selling SKU in the sample data")
    res = allocation.allocate_by_forecast(db, sku=sku, qty=1, branch_codes=None)
    kinds = {a["kind"] for a in res["allocations"]}
    assert "none" not in kinds and "seed" not in kinds
    assert kinds <= {"probe", "cover", "held", "untested"}
