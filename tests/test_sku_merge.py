"""Joining the sales history of a product that was re-coded."""
import numpy as np

from wms.analytics import sku_merge as sm


def _panel(rows, W=10, names=None):
    """rows: {(branch, sku): monthly list}; names default to one shared name."""
    keys = list(rows)
    MAT = np.array([rows[k] for k in keys], float)
    item_of = {k: (names or {}).get(k[1], "P/BLOCK SN511 CHINESE LONG BASE") for k in keys}
    return {"keys": keys, "MAT": MAT, "PROFIT": MAT * 2, "REV": MAT * 10.0,
            "weeks": list(range(MAT.shape[1])), "item_of": item_of}


OLD = [10, 12, 9, 11, 13, 10, 0, 0, 0, 0]
NEW = [0, 0, 0, 0, 0, 0, 12, 8, 3, 0]


def test_old_code_is_folded_into_the_code_that_replaced_it():
    pan = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW})
    m = sm.merge_panel(pan, scale=False)
    assert m["keys"] == [("MP", "NEW1")]
    assert list(m["MAT"][0]) == [a + b for a, b in zip(OLD, NEW)]
    assert list(m["REV"][0]) == [10.0 * (a + b) for a, b in zip(OLD, NEW)]
    assert m["merges"][0]["old"] == "OLD1" and m["merges"][0]["new"] == "NEW1"
    assert ("MP", "OLD1") not in m["item_of"]
    # the shared/cached input panel is never modified
    assert len(pan["keys"]) == 2 and list(pan["MAT"][0]) == OLD


def test_two_live_codes_selling_together_are_not_merged():
    both = [5, 6, 7, 5, 6, 7, 5, 6, 7, 5]
    pan = _panel({("MP", "A"): both, ("MP", "B"): both})
    assert sm.merge_panel(pan)["merges"] == []


def test_different_names_or_prices_are_not_merged():
    pan = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW},
                 names={"NEW1": "SOMETHING ELSE ENTIRELY"})
    assert sm.merge_panel(pan)["merges"] == []
    pan = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW})
    pan["REV"][1] = pan["REV"][1] * 5                       # 5x the unit price
    assert sm.merge_panel(pan)["merges"] == []


def test_short_or_missing_names_never_match():
    pan = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW}, names={"OLD1": "TEST", "NEW1": "TEST"})
    assert sm.merge_panel(pan)["merges"] == []


def test_a_chain_of_recodes_lands_on_the_latest_code():
    a = [10, 10, 10, 10, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 10, 10, 10, 10, 0, 0]
    c = [0, 0, 0, 0, 0, 0, 0, 0, 10, 10]
    m = sm.merge_panel(_panel({("MP", "A"): a, ("MP", "B"): b, ("MP", "C"): c}))
    assert m["keys"] == [("MP", "C")]
    assert list(m["MAT"][0]) == [10] * 10
    assert {(x["old"], x["new"]) for x in m["merges"]} == {("A", "C"), ("B", "C")}


def test_a_company_wide_recode_is_applied_where_local_history_is_thin():
    rows = {}
    for br in ("BM", "ES", "FL"):                           # established at 3 branches
        rows[(br, "OLD1")], rows[(br, "NEW1")] = OLD, NEW
    thin_old = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0]               # only 1 selling month here
    rows[("MP", "OLD1")], rows[("MP", "NEW1")] = thin_old, NEW
    m = sm.merge_panel(_panel(rows))
    got = {(x["branch"], x["basis"]) for x in m["merges"]}
    assert ("MP", "company-wide") in got and ("BM", "branch") in got
    assert ("MP", "OLD1") not in m["keys"]
    # ...but a lone thin pair with no company-wide support is left alone
    lone = sm.merge_panel(_panel({("MP", "OLD1"): thin_old, ("MP", "NEW1"): NEW}))
    assert lone["merges"] == []


def test_aliases_map_old_to_live_codes_uppercase():
    pan = _panel({("MP", "old1"): OLD, ("MP", "new1"): NEW})
    assert sm.aliases(sm.merge_panel(pan)["merges"]) == {"OLD1": "NEW1"}


def test_load_panel_applies_the_merge_and_can_be_switched_off(tmp_path, monkeypatch):
    from wms.analytics import weekly_forecast as wf
    from wms.config import get_settings
    raw = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW})
    monkeypatch.setattr(wf, "_load_panel_raw", lambda directory=None: raw)
    wf._MERGE_CACHE.clear()
    assert wf.load_panel()["keys"] == [("MP", "NEW1")]
    monkeypatch.setattr(get_settings(), "weekly_merge_recoded", False)
    assert len(wf.load_panel()["keys"]) == 2
    assert len(raw["keys"]) == 2                            # never mutated


def test_old_history_is_added_at_the_share_the_new_code_has_taken_over():
    # old sold ~10.8/month; the new code has sold 5.75/month since it began
    pan = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW})
    m = sm.merge_panel(pan)
    share = m["merges"][0]["share"]
    assert abs(share - 5.75 / np.mean(OLD[:6][-6:])) < 0.01 and 0.25 <= share < 1
    assert np.allclose(m["MAT"][0], np.array(NEW) + share * np.array(OLD), atol=0.01)
    assert m["MAT"][0].sum() < np.sum(OLD) + np.sum(NEW)


def test_a_new_code_that_sells_as_much_as_the_old_gets_all_of_it():
    new_full = [0, 0, 0, 0, 0, 0, 12, 12, 11, 12]
    m = sm.merge_panel(_panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): new_full}))
    assert m["merges"][0]["share"] == 1.0
    assert list(m["MAT"][0]) == [a + b for a, b in zip(OLD, new_full)]


def test_share_never_falls_below_the_floor():
    new_tiny = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert sm.transfer_share(OLD, new_tiny) == sm.SCALE_LO


def test_names_match_ignoring_spacing_and_punctuation_but_not_sizes():
    a = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW},
               names={"OLD1": "V BELT B 2591", "NEW1": "V BELT B2591"})
    assert len(sm.merge_panel(a)["merges"]) == 1
    b = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW},
               names={"OLD1": "V BELT B2550", "NEW1": "V BELT B2150"})
    assert sm.merge_panel(b)["merges"] == []            # a different size is a different product



def test_merge_report_lists_each_join_with_its_months_and_volumes():
    pan = _panel({("MP", "OLD1"): OLD, ("MP", "NEW1"): NEW, ("BM", "OLD1"): OLD, ("BM", "NEW1"): NEW},
                 names={})
    pan["weeks"] = [f"2026-{m:02d}-28" for m in range(1, 11)]
    merged = sm.merge_panel(pan)
    prod, detail = sm.merge_report(pan, merged["merges"])
    assert len(prod) == 1 and len(detail) == 2
    row = prod.iloc[0]
    assert (row["Old code"], row["New code"], row["Branches"]) == ("OLD1", "NEW1", 2)
    assert row["Old code units (all branches)"] == 2 * int(sum(OLD))
    assert detail.iloc[0]["Old code sold"] == "Jan 2026 - Jun 2026"
    assert detail.iloc[0]["New code sold"] == "Jul 2026 - Sep 2026"
    empty_prod, empty_detail = sm.merge_report(pan, [])
    assert empty_prod.empty and empty_detail.empty
