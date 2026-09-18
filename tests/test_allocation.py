"""Unit tests for the sales-capped split helper used by 'Split by predicted
sales' - the part that stops a branch hoarding stock it cannot move."""
import pytest

from wms.analytics.allocation import _split_capped


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
    assert match["kind"] in ("held", "none")


def test_allocation_seeds_an_unstocked_branch_with_a_real_amount_for_a_fast_mover(seeded):
    """A branch with no sales history for a product, and none on hand, should
    still get a real amount to test demand there when the product sells
    briskly at other branches - a token 1-unit probe makes no sense for a
    fast mover with stock left over after covering its existing branches.
    The seed is sized off the network's own lowest-selling (but actively
    selling) branch: roughly half to three quarters of what that branch got,
    per ``_SEED_FRACTION_OF_LOWEST``. Only a genuinely barely-moving product
    (see the very-slow-mover test below) gets the old token 1-2 unit probe."""
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

    def _on_hand(code, sku):
        if inv.empty:
            return 0
        sub = inv[(inv["branch_code"].str.upper() == code) &
                  (inv["sku"].str.lower() == str(sku).lower())]
        return int(sub["on_hand"].sum()) if not sub.empty else 0

    target_sku = target_code = None
    for sku, grp in st.groupby("sku"):
        if grp["weekly_demand"].sum() <= allocation._VERY_SLOW_NET_WEEKLY:
            continue                                    # only real movers here
        sellers = set(grp.loc[grp["weekly_demand"] > 0, "branch"].str.upper())
        for code in sorted(all_codes - sellers):
            if _on_hand(code, sku) == 0:
                target_sku, target_code = sku, code
                break
        if target_sku:
            break
    if target_sku is None:
        pytest.skip("no real-moving SKU with an un-stocked branch gap in the sample data")

    res = allocation.allocate_by_forecast(db, sku=target_sku, qty=100, branch_codes=None)
    assert res["very_slow"] is False
    match = next((a for a in res["allocations"] if a["branch"] == name_by_code[target_code]), None)
    assert match is not None
    assert match["kind"] == "seed"
    covered = [a["allocated"] for a in res["allocations"]
              if a["kind"] == "cover" and a["allocated"] > 0]
    assert covered
    low = min(covered)
    expected = max(1, round(low * allocation._SEED_FRACTION_OF_LOWEST))
    assert match["allocated"] == expected
    # meaningfully more than the old token 1-unit probe, unless the lowest
    # covered branch itself only got 1-2 units
    assert match["allocated"] >= 1


def test_allocation_probes_only_a_very_slow_mover(seeded):
    """A genuinely barely-moving product (well below the ordinary "slow
    mover" bar) still gets only a token 1-2 unit probe at an un-stocked
    branch - not the bigger seed a real mover gets."""
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

    def _on_hand(code, sku):
        if inv.empty:
            return 0
        sub = inv[(inv["branch_code"].str.upper() == code) &
                  (inv["sku"].str.lower() == str(sku).lower())]
        return int(sub["on_hand"].sum()) if not sub.empty else 0

    target_sku = target_code = None
    for sku, grp in st.groupby("sku"):
        net = grp["weekly_demand"].sum()
        if net <= 0 or net > allocation._VERY_SLOW_NET_WEEKLY:
            continue                                    # only very slow movers here
        sellers = set(grp.loc[grp["weekly_demand"] > 0, "branch"].str.upper())
        for code in sorted(all_codes - sellers):
            if _on_hand(code, sku) == 0:
                target_sku, target_code = sku, code
                break
        if target_sku:
            break
    if target_sku is None:
        pytest.skip("no very-slow-moving SKU with an un-stocked branch gap in the sample data")

    res = allocation.allocate_by_forecast(db, sku=target_sku, qty=100, branch_codes=None)
    assert res["very_slow"] is True
    match = next((a for a in res["allocations"] if a["branch"] == name_by_code[target_code]), None)
    assert match is not None
    assert match["kind"] == "probe"
    assert 1 <= match["allocated"] <= allocation._PROBE_UNITS
