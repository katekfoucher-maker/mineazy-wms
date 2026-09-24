"""Borrowing a little demand from same-name / different-spec siblings."""
import numpy as np

from wms.analytics import product_family as pf


def test_family_stem_strips_specs_but_keeps_the_name():
    stems = {pf.family_stem(n) for n in (
        "ELECTRIC CABLE FLEX 1.5MM 3CORE /M",
        "ELECTRIC CABLE FLEX 1.5MMX 4CORE /M",
        "ELECTRIC CABLE FLEX 2.5MM 4CORE /M",
        "ELECTRIC CABLE FLEX 4.MM 4CORE /M",
        "ELECTRIC CABLE FLEX 6MM 4C")}
    assert stems == {"ELECTRIC CABLE FLEX"}
    assert pf.family_stem("PVC TEE CONNECTOR 40MM") == pf.family_stem("PVC TEE CONNECTOR 63MM")
    assert pf.family_stem("ANDELI TERMINAL BLOCK TBS30A Black") == "ANDELI TERMINAL BLOCK"
    # too generic on its own - one word is not a family
    assert pf.family_stem("BEARING 6306 2RSC3") is None
    assert pf.family_stem("") is None and pf.family_stem(None) is None


def _panel(rows):
    return np.array(rows, float)


def _run(items, wd, mat, oos=None):
    branch = ["BM"] * len(items)
    return pf.borrow_from_siblings(branch, items, wd, _panel(mat), per_period_weeks=4.0, oos=oos)


NAME = "ELECTRIC CABLE FLEX {}MM"
STEADY = [80, 90, 100, 95, 85, 90, 100, 95]          # sells every month
GAPPY = [0, 100, 0, 0, 90, 0, 0, 100]                # supplied only now and then


def test_gappy_product_borrows_part_of_the_way_toward_steady_siblings():
    items = [NAME.format(1), NAME.format(2), NAME.format(3), NAME.format(4)]
    wd = [22, 22, 24, 4]                             # last one looks like it barely sells
    new, borrowed = _run(items, wd, [STEADY, STEADY, STEADY, GAPPY])
    assert new[3] > 4 and borrowed[3] == new[3] - 4
    assert new[3] < 22                                # never all the way to the siblings
    assert list(new[:3]) == [22, 22, 24] and not borrowed[:3].any()


def test_never_exceeds_a_multiple_of_its_own_best_period():
    items = [NAME.format(i) for i in range(1, 5)]
    big = [500, 600, 700, 650, 600, 620, 700, 680]
    gappy_small = [0, 8, 0, 0, 6, 0, 0, 8]
    new, _ = _run(items, [150, 150, 160, 2], [big, big, big, gappy_small])
    best_wk = max(gappy_small) / 4.0
    assert new[3] <= int(np.ceil(pf.CAP_MULT * best_wk))


def test_product_that_sells_most_months_is_trusted_as_is():
    items = [NAME.format(i) for i in range(1, 5)]
    steady_low = [5, 6, 4, 5, 6, 5, 4, 5]
    new, borrowed = _run(items, [22, 22, 24, 5], [STEADY, STEADY, STEADY, steady_low])
    assert new[3] == 5 and borrowed[3] == 0


def test_dead_product_is_not_revived_and_needs_enough_siblings():
    dead = [80, 90, 0, 0, 0, 0, 0, 0]
    items = [NAME.format(i) for i in range(1, 5)]
    new, borrowed = _run(items, [22, 22, 24, 1], [STEADY, STEADY, STEADY, dead])
    assert new[3] == 1 and borrowed[3] == 0
    # only one steady sibling -> nothing to borrow from
    new2, b2 = _run(items[:2], [22, 4], [STEADY, GAPPY])
    assert new2[1] == 4 and b2[1] == 0


def test_only_lifts_never_lowers_and_stays_within_a_branch():
    items = [NAME.format(i) for i in range(1, 5)]
    low_sibs = [10, 12, 9, 11, 10, 12, 9, 10]
    new, borrowed = _run(items, [3, 3, 3, 30], [low_sibs, low_sibs, low_sibs, GAPPY])
    assert new[3] == 30 and borrowed[3] == 0
    # siblings at another branch do not count
    branch = ["BM", "BM", "BM", "GW"]
    new2, _ = pf.borrow_from_siblings(branch, items, [22, 22, 24, 4],
                                      _panel([STEADY, STEADY, STEADY, GAPPY]), 4.0)
    assert new2[3] == 4


def test_a_product_not_seen_lately_borrows_less_than_a_fresh_one():
    items = [NAME.format(i) for i in range(1, 5)]
    fresh_gappy = [0, 100, 0, 0, 90, 0, 0, 100]         # last sale in the final period
    stale_gappy = [0, 100, 0, 90, 100, 0, 0, 0]         # last sale 4 periods back
    a, _ = _run(items, [22, 22, 24, 4], [STEADY, STEADY, STEADY, fresh_gappy])
    b, _ = _run(items, [22, 22, 24, 4], [STEADY, STEADY, STEADY, stale_gappy])
    assert 4 < b[3] < a[3]
