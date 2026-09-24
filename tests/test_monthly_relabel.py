"""A re-coded product's old sales rows show up under the live code."""
import numpy as np
import pandas as pd

from wms.analytics import monthly_sales as ms


def _lines():
    per = pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"])
    rows = []
    for i, p in enumerate(per):
        if i < 4:                                           # old code sells Jan-Apr
            rows.append(("MP", "OLD1", "P/BLOCK SN511 LONG BASE", p, 10.0 + i, 100.0 + i, 20.0))
        else:                                               # new code from May
            rows.append(("MP", "NEW1", "P/BLOCK SN511 LONG BASE", p, 8.0, 84.0, 16.0))
    rows.append(("MP", "OTHER", "SOMETHING ELSE ENTIRELY", per[0], 5.0, 50.0, 10.0))
    df = pd.DataFrame(rows, columns=["branch_code", "sku", "item", "period", "qty", "turnover", "profit"])
    df["branch"] = "Maphisa"
    df["month_label"] = df["period"].dt.strftime("%b %Y")
    df["category"] = df["item"].map(ms.categorise)
    df["gp_pct"] = 20.0
    df["day_from"], df["day_to"] = 1, 28
    return df[ms._PANEL_COLUMNS]


def test_old_rows_appear_under_the_live_code_in_full():
    df = _lines()
    out = ms._relabel_recoded(df)
    assert "OLD1" not in set(out["sku"])
    mine = out[out["sku"] == "NEW1"].sort_values("period")
    assert list(mine["qty"]) == [10.0, 11.0, 12.0, 13.0, 8.0, 8.0]        # old months + new months, unscaled
    assert len(out) == len(df)
    assert out["qty"].sum() == df["qty"].sum() and out["turnover"].sum() == df["turnover"].sum()
    assert set(out.loc[out["sku"] == "OTHER", "item"]) == {"SOMETHING ELSE ENTIRELY"}
    assert list(out.columns) == ms._PANEL_COLUMNS


def test_a_month_both_codes_sold_is_added_together():
    df = _lines()
    extra = df[df["sku"] == "NEW1"].iloc[[0]].copy()
    extra["period"] = pd.Timestamp("2026-04-30"); extra["month_label"] = "Apr 2026"; extra["qty"] = 3.0
    extra["turnover"] = 30.0; extra["profit"] = 6.0
    out = ms._relabel_recoded(pd.concat([df, extra], ignore_index=True))
    apr = out[(out["sku"] == "NEW1") & (out["period"] == pd.Timestamp("2026-04-30"))]
    assert len(apr) == 1 and float(apr["qty"].iloc[0]) == 13.0 + 3.0
    assert not out.duplicated(["branch_code", "sku", "period"]).any()


def test_switching_the_merge_off_returns_the_data_as_recorded(monkeypatch):
    from wms.config import get_settings
    monkeypatch.setattr(get_settings(), "weekly_merge_recoded", False)
    df = _lines()
    assert ms._relabel_recoded(df) is df


def test_the_matrix_panel_carries_the_full_history_under_the_live_code():
    from wms.analytics import sku_merge as sm
    df = _lines()
    raw = ms._build_matrix_panel(df)
    assert ("MP", "OLD1") in raw["keys"]                    # the raw panel keeps both codes
    merged = sm.merge_panel(raw, scale=False)
    assert ("MP", "OLD1") not in merged["keys"]
    i = merged["keys"].index(("MP", "NEW1"))
    assert list(merged["MAT"][i][:6]) == [10, 11, 12, 13, 8, 8]
