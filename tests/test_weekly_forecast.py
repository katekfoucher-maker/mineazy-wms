"""Weekly per-SKU demand forecast: filename parsing, backtest, upload route."""
import io

import pandas as pd
import pytest

from wms.analytics import weekly_forecast as wf


@pytest.fixture(autouse=True)
def _isolate_model_choice(tmp_path_factory, monkeypatch, seeded):
    """Ignore any runtime-pinned model file / saved network so these tests use
    the config default and train fresh (fast / deterministic). ``seeded``
    ensures the WeeklySalesLine / WeeklyStockSnapshotLine tables exist for the
    "no files here -> read the database" fallback these loaders now have,
    regardless of what other test files have or haven't run yet."""
    d = tmp_path_factory.mktemp("mc")
    monkeypatch.setattr(wf, "_model_choice_path", lambda: d / "weekly_model.txt")
    monkeypatch.setattr(wf, "_ratio_ckpt_path", lambda: d / "esrnn_ratio.pt")
    monkeypatch.setattr(wf, "_ratio_meta_path", lambda: d / "esrnn_ratio.json")
    monkeypatch.setattr(wf, "_gbm_ckpt_path", lambda: d / "gbm.json")
    monkeypatch.setattr(wf, "_gbm_meta_path", lambda: d / "gbm.meta.json")
    monkeypatch.setattr(wf, "_lgbm_ckpt_path", lambda: d / "lgbm.txt")
    monkeypatch.setattr(wf, "_lgbm_meta_path", lambda: d / "lgbm.meta.json")
    monkeypatch.setattr(wf, "_blend_weights_path", lambda: d / "blend.json")
    # isolate from the real data/weekly_inventory dir (a test opts in by
    # pointing weekly_inventory_dir at its own fixture)
    empty_inv = tmp_path_factory.mktemp("wi_empty")
    monkeypatch.setattr(wf, "weekly_inventory_dir", lambda: empty_inv)
    wf._CACHE.clear()
    wf._CKPT_CACHE.clear()
    wf._BW_CACHE.clear()
    _clear_weekly_db_tables()
    yield
    _clear_weekly_db_tables()


def _clear_weekly_db_tables():
    """Keep WeeklySalesLine / WeeklyStockSnapshotLine empty around each test -
    most tests here isolate via an explicit ``directory=``/monkeypatched
    folder and never touch these tables, but the handful that exercise the
    upload routes (which write straight into them now) shouldn't leak into
    each other or into the "no files -> read the DB" fallback tests."""
    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine, WeeklyStockSnapshotLine
    db = SessionLocal()
    try:
        db.query(WeeklySalesLine).delete()
        db.query(WeeklyStockSnapshotLine).delete()
        db.commit()
    finally:
        db.close()


def _real_weeks(branch_code: str | None = None):
    """[(branch_code, week_start date), ...] for every REAL (non-simulated)
    WeeklySalesLine row - what the upload route writes now instead of a file."""
    from wms.db import SessionLocal
    from wms.models import WeeklySalesLine
    db = SessionLocal()
    try:
        q = db.query(WeeklySalesLine.branch_code, WeeklySalesLine.week_start).filter(
            WeeklySalesLine.is_simulated.is_(False))
        if branch_code:
            q = q.filter(WeeklySalesLine.branch_code == branch_code)
        return sorted(set(q.all()))
    finally:
        db.close()


def _week_file(path, rows):
    """rows: (sku, item, qty) or (sku, item, qty, profit, turnover) tuples ->
    an 'Item Statistics' xlsx like the HansaWorld export."""
    recs = []
    for r in rows:
        s, it, q = r[0], r[1], r[2]
        profit = r[3] if len(r) > 3 else 0
        turnover = r[4] if len(r) > 4 else 0
        recs.append({"Item No": s, "Item": it, "Qty": q, "Unnamed: 3": None,
                     "Profit": profit, "GP %": 0, "Turnover": turnover})
    recs.append({"Item No": None, "Item": "TOTAL", "Qty": sum(r[2] for r in rows)})
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame(recs).to_excel(xl, sheet_name="Item Statistics", index=False)


@pytest.mark.parametrize("name,code,week", [
    ("BELMONT  02-08-2026 to 08-08-2026 Sales.xlsx", "BM", "2026-08-02"),
    ("VID 23-08-2026 to 29-08-2026 Sales.xlsx", "GWA", "2026-08-23"),
    ("BM 2026-08-02 week.xlsx", "BM", "2026-08-02"),
    ("GWA 2026-08-02 week.xlsx", "GWA", "2026-08-02"),
])
def test_parse_name_branch(name, code, week):
    c, ws = wf.parse_name(name)
    assert c == code
    assert ws == pd.Timestamp(week)


def test_parse_name_rejects_nameless():
    assert wf.parse_name("random.xlsx") == (None, None)


def _make_panel(root, n_weeks=12):
    start = pd.Timestamp("2026-05-04")
    for w in range(n_weeks):
        ws = start + pd.Timedelta(days=7 * w)
        we = ws + pd.Timedelta(days=6)
        tag = f"{ws.strftime('%d-%m-%Y')} to {we.strftime('%d-%m-%Y')}"
        _week_file(root / f"BELMONT  {tag} Sales.xlsx", [
            ("AAA", "STEADY WIDGET", 10 + w % 3),          # smooth-ish
            ("BBB", "SPARSE BOLT", 5 if w % 4 == 0 else 0),  # intermittent
            ("CCC", "ONE-OFF PART", 7 if w == 2 else 0),     # new / dead
        ])
        _week_file(root / f"VID {tag} Sales.xlsx", [
            ("AAA", "STEADY WIDGET", 4),
            ("DDD", "GWANDA ONLY", 2 + w % 2),
        ])


def test_build_backtest_and_state(tmp_path):
    _make_panel(tmp_path)
    r = wf.build(directory=str(tmp_path), test_weeks=4)

    cov = r["coverage"]
    assert set(cov["branches"]) == {"BM", "GWA"}
    assert cov["weeks"] == 12
    assert r["test_weeks"] == 4

    st = r["state"]
    # 4 distinct (branch, sku) series: BM/AAA BM/BBB BM/CCC GWA/AAA GWA/DDD
    assert len(st) == 5
    assert set(zip(st["branch"], st["sku"])) == {
        ("BM", "AAA"), ("BM", "BBB"), ("BM", "CCC"), ("GWA", "AAA"), ("GWA", "DDD")}
    assert (st["weekly_demand"] >= 0).all()
    assert st["branch_name"].tolist().count("Belmont Shop") == 3

    # every classical candidate is scored on the strict hold-out (neural rows may
    # add to this on real, longer histories)
    assert list(r["overall"].columns) == ["MAE", "RMSE", "WAPE", "bias", "MASE"]
    assert set(r["overall"].index).issuperset(wf._MODELS)
    assert set(r["champ"]) <= {"smooth", "erratic", "intermittent", "lumpy", "new", "dead"}
    assert r["coverage"]["train_weeks"] == cov["weeks"] - r["test_weeks"]

    # ONE model for every product; chosen among the eligible (WAPE, prefers a
    # forecast that moves week to week, must not zero out still-selling SKUs)
    assert r["best_method"] in set(r["overall"].index)
    assert st["method"].nunique() == 1 and st["method"].iloc[0] == r["best_method"]
    assert r["coverage"]["method"] == r["best_method"]

    # the hold-out forecast is the chosen model fitted on the TRAIN weeks only
    # (nothing from the last test_weeks columns), then run through the shared
    # _finalise pipeline: uplift -> category pool -> asymmetric level clamp -> floor
    import numpy as np
    from wms.analytics import weekly_forecast as _wf
    from wms.analytics.weekly_forecast import load_panel, _MODELS, _finalise
    assert float(r["coverage"]["applied_uplift"]) >= 1.0    # live: never lowered
    uplift = float(r["coverage"]["holdout_uplift"])
    assert uplift >= 1.0
    if r["best_method"] in _MODELS:
        pan = load_panel(str(tmp_path))
        MAT = pan["MAT"]
        tr_end = MAT.shape[1] - r["test_weeks"]
        bt, labs = r["backtest"], r["test_week_labels"]
        fn = _MODELS[r["best_method"]]
        tr = MAT[:, :tr_end].astype(float)
        cats = np.array([_wf.categorise(pan["item_of"][k]) for k in pan["keys"]])
        raw = np.stack([fn(tr[i], r["test_weeks"]) for i in range(len(pan["keys"]))])
        want = np.round(_finalise(raw, tr, uplift, cats, _wf._SPIKE_INFLUENCE,
                                  _wf._CATEGORY_POOL, _wf._DOWN_BAND, _wf._UP_BAND,
                                  _wf._MIN_UNITS))
        for i, (b, s) in enumerate(pan["keys"]):
            got = [bt[(bt.branch == b) & (bt.sku == s)].iloc[0][f"pred::{l}"] for l in labs]
            # same pipeline; allow a ±1 rounding gap since coverage exposes the
            # uplift rounded to 3 dp
            assert all(abs(int(w) - int(g)) <= 1 for w, g in zip(want[i], got))

    # every backtest prediction stays within the asymmetric level window and is
    # never zero (a SKU only appears once it has sold at least once)
    for rec in r["backtest"].to_dict("records"):
        row = MAT[[i for i, k in enumerate(pan["keys"])
                   if k == (rec["branch"], rec["sku"])][0], :tr_end]
        ref = _wf._damped_mean(row)
        for lab in labs:
            if ref > 0:
                assert rec[f"pred::{lab}"] <= round(ref * (1 + _wf._UP_BAND)) + 1
            assert rec[f"pred::{lab}"] >= _wf._MIN_UNITS


def test_cached_run_busts_on_new_file(tmp_path, monkeypatch):
    _make_panel(tmp_path, n_weeks=12)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._CACHE.clear()
    assert wf.has_data()
    n0 = len(wf.cached_run()["state"])

    _week_file(tmp_path / "BELMONT  02-05-2027 to 08-05-2027 Sales.xlsx",
               [("ZZZ", "BRAND NEW SKU", 3)])
    n1 = len(wf.cached_run()["state"])
    assert n1 == n0 + 1
    wf._CACHE.clear()


def test_forced_model_choice(tmp_path, monkeypatch):
    _make_panel(tmp_path, n_weeks=14)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    monkeypatch.setattr(wf, "_model_choice_path", lambda: tmp_path / "weekly_model.txt")
    wf._CACHE.clear()

    # auto by default (no file, config may still pin one — just assert it runs)
    wf.set_forced_model("")
    assert wf.forced_model() == ""

    # pin a classical model -> every product uses it, cache rebuilt
    wf.set_forced_model("snaive")
    assert wf.forced_model() == "snaive"
    r = wf.cached_run()
    assert r["best_method"] == "snaive"
    assert set(r["state"]["method"]) == {"snaive"}
    assert r["coverage"]["method"] == "snaive"

    # switch again -> cache busts, new model in use
    wf.set_forced_model("sba")
    assert wf.cached_run()["best_method"] == "sba"

    # back to auto
    wf.set_forced_model("")
    assert wf.forced_model() == ""
    wf._CACHE.clear()


def test_train_and_save_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _make_panel(tmp_path, n_weeks=19)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    # this test specifically exercises the ES-RNN ratio network, which
    # train_and_save() otherwise skips - the suite disables it globally
    # (conftest's WEEKLY_ESRNN=false) so the rest of the suite stays fast
    monkeypatch.setattr(wf.get_settings(), "weekly_esrnn", True, raising=False)
    monkeypatch.setattr(wf.get_settings(), "weekly_esrnn_ratio", True, raising=False)
    wf._PANEL_CACHE.clear(); wf._CKPT_CACHE.clear(); wf._BW_CACHE.clear()

    # no checkpoint yet
    assert wf.load_ratio_checkpoint() is None
    assert wf.checkpoint_status()["exists"] is False

    meta = wf.train_and_save(epochs=15, val_weeks=2)
    assert wf._ratio_ckpt_path().exists() and wf._ratio_meta_path().exists()
    assert meta["n_series"] == 5 and meta["n_weeks"] == 19
    assert set(meta["branches"]) == {"BM", "GWA"}
    assert "holdout_wape" in meta and "holdout_bias" in meta
    # rolling-origin CV + learned blend weights were recorded and persisted
    assert meta["blend_mode"] in ("learned", "fixed")
    assert meta["blend_weights"] and wf._blend_weights_path().exists()
    assert wf.load_blend_weights() is None or isinstance(wf.load_blend_weights(), dict)
    ms = meta["model_scores"]
    assert "blend_fixed" in ms and "blend_learned" in ms

    # it can be loaded back and carries the network weights
    ck = wf.load_ratio_checkpoint()
    assert ck is not None and ck["hp"]["hidden"] == 16
    assert "rnn.weight_ih_l0" in ck["net_state"]

    st = wf.checkpoint_status()
    assert st["exists"] is True and st["mode"] == "saved network"
    assert st["stale"] is False                       # matches the files just trained on

    # add a new week -> the saved net is now stale vs the files on disk
    _week_file(tmp_path / "BELMONT  20-08-2026 to 26-08-2026 Sales.xlsx",
               [("AAA", "STEADY WIDGET", 11)])
    assert wf.checkpoint_status()["stale"] is True

    # a forecast build still works with the checkpoint in place (fast path)
    wf._CACHE.clear(); wf._PANEL_CACHE.clear()
    r = wf.build(directory=str(tmp_path), test_weeks=1)
    assert not r["state"].empty
    wf._CACHE.clear(); wf._PANEL_CACHE.clear(); wf._CKPT_CACHE.clear()


def test_cached_run_is_the_one_week_holdout(tmp_path, monkeypatch):
    _make_panel(tmp_path, n_weeks=14)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._CACHE.clear()
    r = wf.cached_run()

    assert r["iteration"] == "1-week"
    assert r["test_weeks"] == 1
    assert len(r["test_week_labels"]) == 1
    v = wf.backtest_view(which="1")
    assert v["test_weeks"] == 1 and len(v["weeks"]) == 1
    # "4" no longer exists — falls back to the primary (still the 1-week run)
    assert wf.backtest_view(which="4")["test_weeks"] == 1
    wf._CACHE.clear()


def test_empty_dir_is_safe(tmp_path):
    r = wf.build(directory=str(tmp_path))
    assert r["state"].empty
    assert r["coverage"]["series"] == 0


def test_weekly_sales_series(tmp_path, monkeypatch):
    _make_panel(tmp_path, n_weeks=12)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._PANEL_CACHE.clear()

    pan = wf.load_panel(str(tmp_path))
    total_all = int(pan["MAT"].sum())

    allb = wf.weekly_sales_series()
    assert allb["has_data"] and allb["n_weeks"] == 12
    assert allb["branch_label"] == "All branches"
    assert allb["sku_label"] == "All products"
    assert sum(allb["values"]) == total_all == allb["total"]
    assert len(allb["svg"]["dots"]) == 12 and len(allb["svg"]["grid"]) == 5
    assert allb["svg"]["line"]

    # one branch = that branch's rows only
    bm = wf.weekly_sales_series(bcode="BM")
    bm_rows = [i for i, (b, _s) in enumerate(pan["keys"]) if b == "BM"]
    assert bm["total"] == int(pan["MAT"][bm_rows].sum()) < total_all

    # one product, summed across branches (AAA sells in BM and GWA)
    aaa = wf.weekly_sales_series(sku="AAA")
    aaa_rows = [i for i, (_b, s) in enumerate(pan["keys"]) if s == "AAA"]
    assert aaa["total"] == int(pan["MAT"][aaa_rows].sum())
    assert aaa["sku_label"].startswith("AAA")
    assert aaa["n_skus"] == 1

    # product + branch
    ddd = wf.weekly_sales_series(bcode="GWA", sku="DDD")
    assert ddd["total"] > 0 and ddd["branch_label"].lower().startswith(("gwa", "gwanda", "vid"))

    # inventory view: no weekly stock files here -> no inventory line, but no crash
    inv = wf.weekly_sales_series(metric="inventory")
    assert inv["metric"] == "inventory" and inv["inv_available"] is False
    assert inv["svg"]["show_inv"] is False and not inv["svg"]["inv_line"]
    both = wf.weekly_sales_series(metric="both")
    assert both["svg"]["show_sales"] is True and both["svg"]["show_inv"] is False

    # with weekly stock files the inventory line renders and is dual-axis
    wi = tmp_path / "wi"; wi.mkdir()
    for w in range(12):
        ws = pd.Timestamp("2026-05-04") + pd.Timedelta(days=7 * w)
        _stock_file(wi / f"BM {ws.strftime('%d-%m-%Y')} Stock List.xlsx",
                    [("AAA", 50 + w), ("BBB", 20), ("CCC", 5)])
    monkeypatch.setattr(wf, "weekly_inventory_dir", lambda: wi)
    wf._INV_PANEL_CACHE.clear()
    iv = wf.weekly_sales_series(bcode="BM", metric="both")
    assert iv["inv_available"] and iv["svg"]["show_inv"] and iv["svg"]["dual"]
    assert iv["svg"]["inv_line"] and iv["svg"]["line"]
    assert iv["inv_peak"] >= 61 and iv["svg"]["grid"][0]["rlabel"]
    wf._PANEL_CACHE.clear(); wf._INV_PANEL_CACHE.clear()


def test_branch_mix_pie(tmp_path, monkeypatch):
    _make_panel(tmp_path, n_weeks=12)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._PANEL_CACHE.clear()

    # each BRANCH's slice of the total (same shape as sales_mix, for pie_card)
    r = wf.branch_mix()
    assert r["has_data"] and r["metric"] == "units"
    assert r["branch_label"] == "All products" and r["highlight"] is None
    labels = [s["label"] for s in r["legend"]]
    assert set(labels) == {"Belmont Shop", "Gwanda VID"}
    assert labels[0] == "Belmont Shop"                        # BM sells most
    assert abs(sum(s["pct"] for s in r["legend"]) - 100) < 0.5
    assert all(s["path"].startswith(("M", "m")) for s in r["slices"])

    # one product -> that product split by branch
    a = wf.branch_mix(sku="AAA")
    assert a["branch_label"].startswith("AAA")
    assert {s["label"] for s in a["legend"]} == {"Belmont Shop", "Gwanda VID"}

    # compare a subset of branches -> only those, renormalised
    two = wf.branch_mix(bcodes=["BM", "GWA"])
    assert two["picked"] == ["BM", "GWA"] and len(two["legend"]) == 2
    assert abs(sum(s["pct"] for s in two["legend"]) - 100) < 0.5

    # profit metric
    p = wf.branch_mix(metric="profit")
    assert p["metric"] == "profit" and p["metric_label"] == "profit"

    # unknown product -> no slices (pie card shows the empty message)
    assert wf.branch_mix(sku="ZZZ_NOPE")["slices"] == []
    wf._PANEL_CACHE.clear()


def test_sales_mix(tmp_path, monkeypatch):
    for wk in range(6):
        ws = pd.Timestamp("2026-05-04") + pd.Timedelta(days=7 * wk)
        tag = f"{ws.strftime('%d-%m-%Y')} to {(ws + pd.Timedelta(days=6)).strftime('%d-%m-%Y')}"
        # CHEAP: high units, low profit.  MACHINE: low units, high profit.
        _week_file(tmp_path / f"BELMONT  {tag} Sales.xlsx", [
            ("CHEAP", "CHEAP BOLT", 100, 20, 300),
            ("MACHINE", "BIG MACHINE", 1, 400, 1500),
        ])
        _week_file(tmp_path / f"VID {tag} Sales.xlsx", [
            ("CHEAP", "CHEAP BOLT", 10, 2, 30),
        ])
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._PANEL_CACHE.clear()

    units = wf.sales_mix(metric="units")
    assert units["has_data"] and units["branch_label"] == "All branches"
    assert units["metric_label"] == "units sold"
    top_u = {row["label"]: row["pct"] for row in units["legend"]}
    assert top_u["CHEAP BOLT"] > top_u["BIG MACHINE"]           # volume leader
    assert abs(sum(s["pct"] for s in units["legend"]) - 100) < 0.5
    assert all(s["path"].startswith(("M", "m")) for s in units["slices"])

    profit = wf.sales_mix(metric="profit")
    top_p = {row["label"]: row["pct"] for row in profit["legend"]}
    assert top_p["BIG MACHINE"] > top_p["CHEAP BOLT"]           # margin leader

    # branch filter narrows it (VID never sold MACHINE)
    vid = wf.sales_mix(bcode="GWA", metric="units")
    assert {row["label"] for row in vid["legend"]} == {"CHEAP BOLT"}

    # pick specific products to compare -> only those, no "Other"
    cmp = wf.sales_mix(metric="units", skus=["CHEAP", "MACHINE", ""])
    assert cmp["picked"] == ["CHEAP", "MACHINE"]
    assert {row["label"] for row in cmp["legend"]} == {"CHEAP BOLT", "BIG MACHINE"}
    assert "Other" not in {row["label"] for row in cmp["legend"]}
    assert abs(sum(s["pct"] for s in cmp["slices"]) - 100) < 0.5

    # a single SKU is SEARCH mode: still the full pie, but with highlight info
    one = wf.sales_mix(metric="profit", skus=["MACHINE"])
    assert {row["label"] for row in one["legend"]} == {"CHEAP BOLT", "BIG MACHINE"}
    assert one["highlight"]["found"] and one["highlight"]["sku"] == "MACHINE"
    assert one["highlight"]["name"] == "BIG MACHINE"
    assert one["highlight"]["rank"] == 1                # top by profit
    hi_rows = [row for row in one["legend"] if row["highlight"]]
    assert len(hi_rows) == 1 and hi_rows[0]["label"] == "BIG MACHINE"
    wf._PANEL_CACHE.clear()


def test_sales_mix_search_mode(tmp_path, monkeypatch):
    fillers = [(f"FILL{i}", f"FILLER {i}", 5, 1, 15) for i in range(9)]  # 9 mid sellers
    for wk in range(6):
        ws = pd.Timestamp("2026-05-04") + pd.Timedelta(days=7 * wk)
        tag = f"{ws.strftime('%d-%m-%Y')} to {(ws + pd.Timedelta(days=6)).strftime('%d-%m-%Y')}"
        _week_file(tmp_path / f"BELMONT  {tag} Sales.xlsx",
                  [("CHEAP", "CHEAP BOLT", 100, 20, 300)] + fillers)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._PANEL_CACHE.clear()

    # 10 products total (CHEAP + 9 fillers) > top=8, so a filler is normally
    # rolled into "Other" -> searching for one must pull it out as its own slice
    m = wf.sales_mix(metric="units", skus=["FILL8"])
    assert m["highlight"]["found"] and m["highlight"]["sku"] == "FILL8"
    assert m["highlight"]["name"] == "FILLER 8"
    assert m["highlight"]["n_total"] == 10
    assert m["highlight"]["rank"] > 8                    # below the natural top-8
    labels = {row["label"] for row in m["legend"]}
    assert "FILLER 8" in labels and "Other" in labels     # pulled out, Other still shown
    hi = [row for row in m["legend"] if row["highlight"]]
    assert len(hi) == 1 and hi[0]["sku"] == "FILL8"
    assert abs(sum(row["pct"] for row in m["legend"]) - 100) < 0.5

    # branch filter applies to search mode too
    empty = wf.sales_mix(bcode="GWA", metric="units", skus=["FILL8"])
    assert empty["highlight"]["found"] is False           # nothing sold at GWA

    # a SKU that isn't in the data at all: reported as not found, no crash
    missing = wf.sales_mix(metric="units", skus=["NOPE_NOT_A_SKU"])
    assert missing["highlight"] == {
        "sku": "NOPE_NOT_A_SKU", "name": "NOPE_NOT_A_SKU", "pct": 0.0,
        "rank": None, "n_total": 10, "found": False}
    wf._PANEL_CACHE.clear()


# ------------------------------------------------------- stockout unconstraining
import numpy as np


def _stock_file(path, rows):
    """rows: (sku, balance) -> a Hansa 'Stock List' xlsx (Item No / Balance)."""
    df = pd.DataFrame([{"Item No": s, "Name": s, "Unit": "IT", "Balance": b,
                        "Unit Cost": 1, "Value": b} for s, b in rows])
    with pd.ExcelWriter(path) as xl:
        df.to_excel(xl, sheet_name="Stock List", index=False)


def test_load_inventory_panel_negatives_to_zero(tmp_path):
    keys = [("BM", "AAA"), ("BM", "BBB")]
    weeks = [(pd.Timestamp("2026-05-04") + pd.Timedelta(days=7 * w)).date().isoformat()
             for w in range(4)]
    _stock_file(tmp_path / "BM 04-05-2026 Stock List.xlsx", [("AAA", 12), ("BBB", -7)])
    _stock_file(tmp_path / "BM 18-05-2026 Stock List.xlsx", [("AAA", 0), ("BBB", 4)])

    OH = wf.load_inventory_panel(keys, weeks, directory=str(tmp_path))
    assert OH is not None and OH.shape == (2, 4)
    assert OH[0, 0] == 12 and OH[1, 0] == 0.0            # -7 clamped to 0
    assert OH[0, 2] == 0.0 and OH[1, 2] == 4
    assert np.isnan(OH[0, 1]) and np.isnan(OH[0, 3])     # weeks with no reading


def test_load_inventory_panel_aligns_to_a_monthly_panel(tmp_path):
    """pan["weeks"] is month-end dates under the production default
    (weekly_data_source="monthly") - real weekly Hansa stock files must still
    land in the right MONTH bucket instead of being dropped/misaligned by
    the old (date - weeks[0]).days / 7 math, and several weekly readings in
    the same month take their lowest value (the most sensitive "stocked out
    at some point this month" signal)."""
    keys = [("BM", "AAA")]
    weeks = ["2026-04-30", "2026-05-31", "2026-06-30"]        # 3 monthly periods

    _stock_file(tmp_path / "BM 15-04-2026 Stock List.xlsx", [("AAA", 20)])
    _stock_file(tmp_path / "BM 04-05-2026 Stock List.xlsx", [("AAA", 50)])
    _stock_file(tmp_path / "BM 11-05-2026 Stock List.xlsx", [("AAA", 30)])
    _stock_file(tmp_path / "BM 25-05-2026 Stock List.xlsx", [("AAA", 0)])
    OH = wf.load_inventory_panel(keys, weeks, directory=str(tmp_path))

    assert OH is not None and OH.shape == (1, 3)
    assert OH[0, 0] == 20                     # April reading -> April bucket
    assert OH[0, 1] == 0                      # 3 May readings -> May bucket, lowest kept
    assert np.isnan(OH[0, 2])                 # no June reading


def test_load_inventory_panel_from_db_aligns_to_a_monthly_panel(seeded):
    from wms.db import get_session
    from wms.models import WeeklyStockSnapshotLine

    db = next(get_session())
    db.query(WeeklyStockSnapshotLine).filter(
        WeeklyStockSnapshotLine.branch_code == "ZZZTEST").delete()
    db.add_all([
        WeeklyStockSnapshotLine(branch_code="ZZZTEST", sku="Q1",
                                week_start=pd.Timestamp("2026-05-04").date(),
                                qty_on_hand=50),
        WeeklyStockSnapshotLine(branch_code="ZZZTEST", sku="Q1",
                                week_start=pd.Timestamp("2026-05-25").date(),
                                qty_on_hand=0),
    ])
    db.commit()

    keys = [("ZZZTEST", "Q1")]
    weeks = ["2026-04-30", "2026-05-31", "2026-06-30"]
    OH = wf._load_inventory_panel_from_db(keys, weeks)
    assert OH is not None
    assert OH[0, 1] == 0                      # both May readings -> May bucket, lowest kept

    db.query(WeeklyStockSnapshotLine).filter(
        WeeklyStockSnapshotLine.branch_code == "ZZZTEST").delete()
    db.commit()


def test_scaled_max_run_converts_weeks_to_months():
    weekly_weeks = [(pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * w)).date().isoformat()
                    for w in range(10)]
    assert wf._scaled_max_run(weekly_weeks) == wf._MAX_STOCKOUT_RUN     # unchanged for weekly

    monthly_weeks = [pd.Timestamp("2026-01-31") + pd.DateOffset(months=m)
                     for m in range(10)]
    monthly_weeks = [d.date().isoformat() for d in monthly_weeks]
    scaled = wf._scaled_max_run(monthly_weeks)
    assert 1 <= scaled < wf._MAX_STOCKOUT_RUN        # 8 weeks (~2 months), not 8 months


def test_stockout_mask_conservative():
    # AAA: material seller (~20/wk) with an INTERIOR dry gap weeks 4-6
    # BBB: naturally intermittent low seller — must NOT be touched
    # CCC: material seller but the "gap" is a LEADING run (not yet ranged)
    W = 12
    AAA = np.array([20, 22, 18, 21,  0,  0,  0, 19, 20, 23, 18, 21], float)
    BBB = np.array([0,  3,  0,  0,  2,  0,  0,  4,  0,  0,  1,  0], float)
    CCC = np.array([0,  0,  0,  0,  0, 15, 17, 16, 18, 15, 19, 16], float)
    MAT = np.vstack([AAA, BBB, CCC])
    # on-hand: 0 exactly on AAA's gap and on CCC's leading run
    OH = np.full((3, W), np.nan, np.float32)
    OH[0, 4:7] = 0.0
    OH[2, 0:5] = 0.0
    mask, lvl = wf._stockout_mask(MAT, OH, wf._SPIKE_INFLUENCE, heuristic=True)

    assert mask[0, 4:7].all() and mask[0].sum() == 3      # only the interior gap
    assert not mask[1].any()                              # intermittent seller left alone
    assert lvl[1] == 0.0
    assert not mask[2].any()                              # leading run is not interior
    assert lvl[0] >= wf._MIN_LIFT_LEVEL


def test_stockout_mask_long_run_and_cap():
    W = 16
    # a 10-week dry spell in an otherwise-steady seller: longer than _MAX_STOCKOUT_RUN
    row = np.array([20, 21, 19] + [0] * 10 + [22, 20, 21], float)
    MAT = row.reshape(1, -1)
    OH = np.full((1, W), np.nan, np.float32)
    OH[0, 3:13] = 0.0
    mask, lvl = wf._stockout_mask(MAT, OH, wf._SPIKE_INFLUENCE, heuristic=True)
    assert not mask.any()                                 # run trimmed -> nothing lifted


def test_unconstrain_lifts_to_level():
    MAT = np.array([[20.0, 0.0, 0.0, 22.0]])
    mask = np.array([[False, True, True, False]])
    U = wf._unconstrain(MAT, mask, np.array([21.0]))
    assert U[0, 1] == 21.0 and U[0, 2] == 21.0
    assert U[0, 0] == 20.0 and U[0, 3] == 22.0            # untouched
    # a week that somehow sold ABOVE the level keeps its higher value
    U2 = wf._unconstrain(np.array([[30.0, 0.0]]), np.array([[False, True]]),
                         np.array([10.0]))
    assert U2[0, 1] == 10.0


def test_old_excel_formula():
    import math
    # ceil(sum(last 4 weeks) * 1.1) / 4, held flat
    for last4 in ([10, 10, 10, 10], [0, 0, 5, 0], [25, 30, 20, 40], [0, 0, 0, 0]):
        y = np.array([1, 2, 3, 4, 5] + last4, float)
        f = wf.f_old_excel(y, 3)
        want = math.ceil(sum(last4) * 1.1) / 4
        assert list(f) == [want, want, want]
    # fewer than 4 weeks of history -> uses what's there
    assert wf.f_old_excel(np.array([6.0, 4.0]), 1)[0] == math.ceil(10 * 1.1) / 4


def test_old_excel_model_pinned(tmp_path, monkeypatch):
    import math
    _make_panel(tmp_path, n_weeks=14)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    monkeypatch.setattr(wf, "forced_model", lambda: "old_excel")
    wf._CACHE.clear()
    r = wf.build(directory=str(tmp_path), test_weeks=1)
    assert r["best_method"] == "old_excel"
    assert set(r["state"]["method"]) == {"old_excel"}
    assert "old_excel" in r["overall"].index

    # every live number is EXACTLY the spreadsheet rule on raw sales, no uplift,
    # no level clamp, no never-zero floor (round() only for display)
    pan = wf.load_panel(str(tmp_path))
    raw = pan["MAT"].astype(float)
    kx = {k: i for i, k in enumerate(pan["keys"])}
    for _, row in r["state"].iterrows():
        i = kx[(row["branch"], row["sku"])]
        want = round(math.ceil(raw[i, -4:].sum() * 1.1) / 4)
        assert row["weekly_demand"] == want
    wf._CACHE.clear()


def test_training_matrix_can_be_disabled(tmp_path, monkeypatch):
    _make_panel(tmp_path, n_weeks=12)
    pan = wf.load_panel(str(tmp_path))
    monkeypatch.setattr(wf.get_settings(), "weekly_unconstrain", False, raising=False)
    MAT, mask, meta = wf.training_matrix(pan)
    assert not mask.any() and meta["censored"] == 0
    assert np.array_equal(MAT, pan["MAT"].astype(float))


def test_blend_is_a_renormalised_weighted_mean():
    FC = {"esrnn_ratio": np.array([[10.0, 4.0]]), "gbm": np.array([[20.0, 8.0]])}
    b = wf._blend(FC, {"esrnn_ratio": 0.6, "gbm": 0.4})
    assert np.allclose(b, [[14.0, 5.6]])                  # .6*10 + .4*20 ; .6*4 + .4*8
    # a missing component is dropped and the rest renormalised
    b2 = wf._blend({"gbm": np.array([[20.0]])}, {"esrnn_ratio": 0.6, "gbm": 0.4})
    assert np.allclose(b2, [[20.0]])


def test_weekly_ml_forecasters_shapes(tmp_path, monkeypatch):
    _make_panel(tmp_path, n_weeks=22)
    pan = wf.load_panel(str(tmp_path))
    MAT = pan["MAT"]
    S, W = MAT.shape
    keys = pan["keys"]
    cats = np.array([wf.categorise(pan["item_of"][k]) for k in keys])
    from wms.analytics import weekly_ml as ml

    xgb = pytest.importorskip("xgboost")
    G = ml.gbm_forecast(MAT, pan["weeks"], W - 2, 2, keys=keys, cats=cats,
                        n_estimators=40)
    assert G is not None and G.shape == (S, 2)
    assert np.isfinite(G).all() and (G >= 0).all()
    # a saved booster round-trips and is reused (model_in short-circuits the fit)
    p = tmp_path / "gbm.json"
    ml.gbm_forecast(MAT, pan["weeks"], W, 1, keys=keys, cats=cats,
                    n_estimators=40, model_out=str(p))
    assert p.exists()
    G2 = ml.gbm_forecast(MAT, pan["weeks"], W, 1, keys=keys, cats=cats,
                         model_in=str(p))
    assert G2 is not None and G2.shape == (S, 1)

    lgb = pytest.importorskip("lightgbm")
    Lg = ml.lgbm_forecast(MAT, pan["weeks"], W - 2, 2, keys=keys, cats=cats,
                          n_estimators=40)
    assert Lg is not None and Lg.shape == (S, 2)
    assert np.isfinite(Lg).all() and (Lg >= 0).all()
    pl = tmp_path / "lgbm.txt"
    ml.lgbm_forecast(MAT, pan["weeks"], W, 1, keys=keys, cats=cats,
                     n_estimators=40, model_out=str(pl))
    assert pl.exists()
    Lg2 = ml.lgbm_forecast(MAT, pan["weeks"], W, 1, keys=keys, cats=cats,
                           model_in=str(pl))
    assert Lg2 is not None and Lg2.shape == (S, 1)

    pytest.importorskip("torch")
    L = ml.lstm_forecast(MAT, pan["weeks"], W - 1, 1, keys=keys, cats=cats,
                         epochs=8)
    # tiny panels may not have enough windows (returns None) — else check it
    if L is not None:
        assert L.shape == (S, 1) and np.isfinite(L).all() and (L >= 0).all()


def test_build_pins_blend_and_it_beats_neither_component_badly(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    pytest.importorskip("torch")
    _make_panel(tmp_path, n_weeks=22)
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    monkeypatch.setattr(wf, "forced_model", lambda: "blend")
    wf._CACHE.clear()
    r = wf.build(directory=str(tmp_path), test_weeks=1)
    assert r["best_method"] == "blend"
    assert set(r["state"]["method"]) == {"blend"}
    # blend row is present in the comparison and its forecast is between the
    # two components' totals (a mean cannot be outside them)
    ov = r["overall"]
    assert "blend" in ov.index
    wf._CACHE.clear()


def test_refresh_onhand_from_latest_weekly_file(tmp_path, monkeypatch, seeded):
    """The maintenance script pushes the newest week's Hansa balance into the
    stock_on_hand table that the Allocation plan / Low-sales reads."""
    from wms.scripts import refresh_onhand
    from wms.analytics import inventory as inv_mod
    from wms.services import stock as stock_svc
    from wms.db import SessionLocal

    wi = tmp_path / "wi"
    wi.mkdir()
    snap = tmp_path / "snap"
    _stock_file(wi / "BM 04-05-2026 Stock List.xlsx", [("AAA", 5), ("BBB", 1)])
    _stock_file(wi / "BM 18-05-2026 Stock List.xlsx", [("AAA", 40), ("BBB", -3),
                                                       ("CCC", 12)])   # newest wins
    monkeypatch.setattr(wf, "weekly_inventory_dir", lambda: wi)
    monkeypatch.setattr(inv_mod, "inventory_dir", lambda: snap)

    assert refresh_onhand.main([]) == 0

    db = SessionLocal()
    try:
        lv = stock_svc.levels_df(db)
        bm = lv[lv.branch_code == "BM"].set_index("sku")["on_hand"].to_dict()
    finally:
        db.close()
    assert bm == {"AAA": 40, "BBB": 0, "CCC": 12}         # -3 clamped, week 18 used
    assert (snap / "BM.xlsx").exists()                    # snapshot fallback refreshed


def test_build_reports_stockout_and_scores_in_stock_only(tmp_path, monkeypatch):
    """With a weekly-inventory dir, the hold-out is scored on what really sold
    over the in-stock weeks; the backtest carries a stockout count and the
    coverage dict the censored totals."""
    n = 14
    # BM/AAA: steady ~11/wk, but SUPPRESSED to 0 in an interior gap (weeks 6-7)
    # and in the last 2 weeks — all with a 0 on-hand reading.
    for w in range(n):
        ws = pd.Timestamp("2026-05-04") + pd.Timedelta(days=7 * w)
        we = ws + pd.Timedelta(days=6)
        tag = f"{ws.strftime('%d-%m-%Y')} to {we.strftime('%d-%m-%Y')}"
        dry = w in (6, 7) or w >= 12
        _week_file(tmp_path / f"BELMONT  {tag} Sales.xlsx", [
            ("AAA", "STEADY WIDGET", 0 if dry else 10 + w % 3),
            ("BBB", "OTHER BOLT", 6 + w % 2),
            ("CCC", "THIRD PART", 4),
        ])
    inv = tmp_path / "inv"
    inv.mkdir()
    for w in range(n):
        ws = pd.Timestamp("2026-05-04") + pd.Timedelta(days=7 * w)
        dry = w in (6, 7) or w >= 12
        _stock_file(inv / f"BM {ws.strftime('%d-%m-%Y')} Stock List.xlsx",
                    [("AAA", 0 if dry else 40), ("BBB", 30), ("CCC", 30)])
    monkeypatch.setattr(wf, "weekly_inventory_dir", lambda: inv)
    monkeypatch.setattr(wf.get_settings(), "weekly_unconstrain", True, raising=False)
    wf._PANEL_CACHE.clear()

    r = wf.build(directory=str(tmp_path), test_weeks=4)
    cov = r["coverage"]
    assert cov["unconstrain"] is True and cov["inventory_used"] is True
    assert cov["censored_weeks"] >= 2                     # interior gap weeks 6-7 lifted
    assert cov["censored_holdout"] >= 2                   # last 2 held-out weeks dry
    bt = r["backtest"]
    assert "stockout" in bt.columns
    aaa = bt[(bt.branch == "BM") & (bt.sku == "AAA")].iloc[0]
    assert aaa["stockout"] >= 2                           # 2 held-out weeks dry
    # AAA's held-out actual is the raw (0 in the dry weeks), not the lifted level
    assert aaa["actual"] <= 2 * (10 + 2)
    wf._PANEL_CACHE.clear()


def test_upload_weekly_route(tmp_path, monkeypatch, seeded):
    from fastapi.testclient import TestClient
    from wms.api.main import app
    from wms.web import routes as wr

    monkeypatch.setattr(wr.weekly_fc, "weekly_dir", lambda: tmp_path)
    monkeypatch.setattr(wr.monthly_sales, "history_dir", lambda: tmp_path / "no-monthly")
    wr.weekly_fc._CACHE.clear()

    c = TestClient(app)
    c.post("/login", data={"username": "controller", "password": "wms1234"},
           follow_redirects=False)

    def _bytes(rows):
        buf = io.BytesIO()
        df = pd.DataFrame([{"Item No": s, "Item": it, "Qty": q} for s, it, q in rows])
        with pd.ExcelWriter(buf) as xl:
            df.to_excel(xl, sheet_name="Item Statistics", index=False)
        return buf.getvalue()

    files = [
        ("files", ("BELMONT  02-08-2026 to 08-08-2026 Sales.xlsx",
                   _bytes([("AAA", "WIDGET", 3)]),
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
        ("files", ("VID 02-08-2026 to 08-08-2026 Sales.xlsx",
                   _bytes([("AAA", "WIDGET", 1)]),
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
        ("files", ("notes.txt", b"nope", "text/plain")),
    ]
    r = c.post("/allocation/upload-weekly", files=files, follow_redirects=False)
    assert r.status_code == 303

    saved = _real_weeks()
    assert saved == [("BM", pd.Timestamp("2026-08-02").date()),
                     ("GWA", pd.Timestamp("2026-08-02").date())]
    wr.weekly_fc._CACHE.clear()


def test_upload_weekly_route_branch_override(tmp_path, monkeypatch, seeded):
    """A picked branch attributes every file in the upload, even when the name
    has only a date and no recognisable branch token."""
    from fastapi.testclient import TestClient
    from wms.api.main import app
    from wms.web import routes as wr

    monkeypatch.setattr(wr.weekly_fc, "weekly_dir", lambda: tmp_path)
    monkeypatch.setattr(wr.monthly_sales, "history_dir", lambda: tmp_path / "no-monthly")
    wr.weekly_fc._CACHE.clear()

    c = TestClient(app)
    c.post("/login", data={"username": "controller", "password": "wms1234"},
           follow_redirects=False)

    def _xlsx(rows):
        buf = io.BytesIO()
        df = pd.DataFrame([{"Item No": s, "Item": it, "Qty": q} for s, it, q in rows])
        with pd.ExcelWriter(buf) as xl:
            df.to_excel(xl, sheet_name="Item Statistics", index=False)
        return buf.getvalue()

    xtype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    files = [
        ("files", ("weekly report 02-08-2026 to 08-08-2026.xlsx",
                   _xlsx([("AAA", "WIDGET", 3)]), xtype)),
        ("files", ("09-08-2026 to 15-08-2026.xlsx",
                   _xlsx([("AAA", "WIDGET", 4)]), xtype)),
        ("files", ("no dates here.xlsx", _xlsx([("AAA", "WIDGET", 1)]), xtype)),
    ]
    r = c.post("/allocation/upload-weekly", data={"branch_code": "gwa"},
               files=files, follow_redirects=False)
    assert r.status_code == 303

    saved = _real_weeks()
    # both dated files land on GWA; the one with no date in the name is skipped
    assert saved == [("GWA", pd.Timestamp("2026-08-02").date()),
                     ("GWA", pd.Timestamp("2026-08-09").date())]
    wr.weekly_fc._CACHE.clear()


def test_flow_summary_includes_revenue_and_margin(tmp_path, monkeypatch):
    """flow_summary's KPI-strip numbers include last week's revenue (+ WoW
    change) and the all-time gross margin, alongside the existing units
    figures - the Flow Analysis page's own revenue view, so a separate
    Revenue page/module would just duplicate what's already here."""
    start = pd.Timestamp("2026-05-04")
    for w in range(3):
        ws = start + pd.Timedelta(days=7 * w)
        we = ws + pd.Timedelta(days=6)
        tag = f"{ws.strftime('%d-%m-%Y')} to {we.strftime('%d-%m-%Y')}"
        _week_file(tmp_path / f"BELMONT  {tag} Sales.xlsx", [
            ("AAA", "STEADY WIDGET", 10, 30 + w, 100 + w * 10),   # qty, profit, turnover
        ])
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._PANEL_CACHE.clear()

    out = wf.flow_summary()
    assert out["has_data"] is True
    assert out["week_revenue"] == 120                 # last week: qty 10 x turnover 120
    assert out["revenue_wow_pct"] is not None
    assert out["margin_pct"] is not None and out["margin_pct"] > 0

    wf._PANEL_CACHE.clear()


def test_growth_overview_tracks_active_branches_and_products(tmp_path, monkeypatch):
    """Week-over-week sales growth, plus how many branches/products were
    actively selling each week - active-branch count should track a branch
    that only starts selling partway through the window."""
    start = pd.Timestamp("2026-05-04")
    for w in range(8):
        ws = start + pd.Timedelta(days=7 * w)
        we = ws + pd.Timedelta(days=6)
        tag = f"{ws.strftime('%d-%m-%Y')} to {we.strftime('%d-%m-%Y')}"
        _week_file(tmp_path / f"BELMONT  {tag} Sales.xlsx", [
            ("AAA", "STEADY WIDGET", 10 + w),
        ])
        if w >= 4:                                    # VID only starts selling halfway
            _week_file(tmp_path / f"VID {tag} Sales.xlsx", [
                ("BBB", "GWANDA ONLY", 3),
            ])
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._PANEL_CACHE.clear()

    out = wf.growth_overview(weeks=8)
    assert out["has_data"] is True
    assert out["n_weeks"] == 8
    assert out["active_branches"] == 2                # both selling by the last week
    assert out["active_products"] == 2
    assert out["units_period"] > 0
    branch_counts = [p["value"] for p in out["branch_growth"]["points"]]
    assert branch_counts[:4] == [1, 1, 1, 1]
    assert branch_counts[4:] == [2, 2, 2, 2]
    pcts = [p["pct"] for p in out["sales_growth"]["points"]]
    assert pcts[0] is None                            # no prior week to compare
    assert all(p is not None for p in pcts[1:])
    svg = out["sales_growth"]["svg"]
    assert svg["bars"] and len(svg["bars"]) == len(out["sales_growth"]["points"])
    # every bar carries its own real week label (not just a hover tooltip) and
    # a plotted value, so the progression across weeks reads without hovering
    assert svg["xticks"] and all(t["label"] for t in svg["xticks"])
    assert [b["value"] for b in svg["bars"]] == [p or 0 for p in pcts]   # None -> 0-height bar
    assert all("cx" in b and "value_y" in b for b in svg["bars"])

    wf._PANEL_CACHE.clear()


def test_growth_overview_bcode_scopes_only_the_sales_growth_chart(tmp_path, monkeypatch):
    """A bcode filters ONLY the sales_growth chart to one branch's own real
    units - the KPI-strip figures (active branches/products, units_period,
    overall_pct_growth) stay network-wide, since those are shared by the
    whole page, not the chart being filtered."""
    start = pd.Timestamp("2026-05-04")
    for w in range(8):
        ws = start + pd.Timedelta(days=7 * w)
        we = ws + pd.Timedelta(days=6)
        tag = f"{ws.strftime('%d-%m-%Y')} to {we.strftime('%d-%m-%Y')}"
        _week_file(tmp_path / f"BELMONT  {tag} Sales.xlsx", [
            ("AAA", "STEADY WIDGET", 10 + w),
        ])
        if w >= 4:
            _week_file(tmp_path / f"VID {tag} Sales.xlsx", [
                ("BBB", "GWANDA ONLY", 3),
            ])
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._PANEL_CACHE.clear()

    network = wf.growth_overview(weeks=8)
    bm_only = wf.growth_overview(weeks=8, bcode="BM")

    assert bm_only["sales_growth"]["bcode"] == "BM"
    assert bm_only["sales_growth"]["branch_label"] == "Belmont Shop"
    bm_values = [p["value"] for p in bm_only["sales_growth"]["points"]]
    assert bm_values == [10, 11, 12, 13, 14, 15, 16, 17]      # Belmont's own units only

    # the KPI-strip figures are untouched by bcode
    for key in ("active_branches", "active_products", "units_period", "overall_pct_growth"):
        assert bm_only[key] == network[key]

    # blank bcode is explicitly "All branches", network-wide, unchanged behaviour
    all_br = wf.growth_overview(weeks=8, bcode="")
    assert all_br["sales_growth"]["bcode"] == ""
    assert all_br["sales_growth"]["branch_label"] == "All branches"
    assert [p["value"] for p in all_br["sales_growth"]["points"]] == \
           [p["value"] for p in network["sales_growth"]["points"]]

    wf._PANEL_CACHE.clear()


def test_abc_classification_ranks_by_cumulative_revenue_share(tmp_path, monkeypatch):
    """Classic ABC analysis: products ranked by network-wide revenue, split
    into three tiers by CUMULATIVE share - Class A up to ~80%, Class B to
    ~95%, Class C the long tail past that (the strategic split a Demand-
    Driven Push-Pull allocation plan needs: fast/medium/slow-moving)."""
    _week_file(tmp_path / "BELMONT  04-05-2026 to 10-05-2026 Sales.xlsx", [
        ("SKUA", "TOP SELLER", 10, 0, 7500),
        ("SKUB", "MID SELLER", 5, 0, 1500),
        ("SKUC", "TAIL ITEM", 2, 0, 1000),
    ])
    monkeypatch.setattr(wf, "weekly_dir", lambda: tmp_path)
    wf._PANEL_CACHE.clear()

    out = wf.abc_classification()
    assert out["has_data"] is True
    by_sku = {r["sku"]: r for r in out["rows"]}
    assert by_sku["SKUA"]["class"] == "A"
    assert by_sku["SKUB"]["class"] == "B"
    assert by_sku["SKUC"]["class"] == "C"
    assert out["n_a"] == 1 and out["n_b"] == 1 and out["n_c"] == 1
    assert out["class_by_sku"]["SKUA"] == "A"
    assert [r["sku"] for r in out["rows"]] == ["SKUA", "SKUB", "SKUC"]  # revenue, descending
    assert by_sku["SKUA"]["pct_of_revenue"] == 75.0
    assert by_sku["SKUA"]["strategy"] == wf._ABC_STRATEGY["A"]["role"]

    wf._PANEL_CACHE.clear()


def test_low_stock_alerts_flags_only_fast_movers_below_cover(db):
    """Only high-priority (top-quartile network demand) products below the
    1.5-week cover threshold are flagged - a slow mover sitting at zero stock
    is a normal 'worst performer', not a reorder-point alert."""
    if not wf.has_data():
        pytest.skip("no weekly_sales files present")
    out = wf.low_stock_alerts(db)
    assert out["n_high_priority_skus"] >= 0
    for r in out["rows"]:
        assert r["cover_weeks"] < 1.5
        assert r["weekly_demand"] > 0
        assert r["on_hand"] >= 0
        assert r["branch"] and r["sku"]
    # sorted by weekly demand, highest first
    demands = [r["weekly_demand"] for r in out["rows"]]
    assert demands == sorted(demands, reverse=True)


def test_reorder_points_uses_safety_margin_above_average_sales(db):
    """Safety stock sits slightly above the product's own average weekly
    sales; the reorder point adds lead-time cover on top of that, and a
    branch-product's status matches whether on-hand has fallen to it."""
    if not wf.has_data():
        pytest.skip("no weekly_sales files present")
    out = wf.reorder_points(db)
    assert out["n_products"] >= 0
    for r in out["rows"]:
        assert r["avg_weekly_sales"] > 0
        assert r["safety_stock"] >= r["avg_weekly_sales"]        # "slightly higher"
        assert r["reorder_point"] >= r["safety_stock"]
        assert r["branch"] and r["sku"]
        if r["status"] == "reorder_now":
            assert r["on_hand"] <= r["reorder_point"]
        else:
            assert r["on_hand"] > r["reorder_point"]
    # branches at/below their reorder point sort first
    seen_ok = False
    for r in out["rows"]:
        if r["status"] == "ok":
            seen_ok = True
        elif seen_ok:
            pytest.fail("a 'reorder_now' row appeared after an 'ok' row")


def test_excess_stock_flags_deep_cover_and_dead_lines(db):
    """The mirror of low_stock_alerts: branch-products sitting on far more
    stock than their own sales pace justifies, or with on-hand but zero sales
    history at all (flagged 'dead' rather than given a cover figure)."""
    out = wf.excess_stock(db)
    assert out["total_excess"] >= 0
    for r in out["rows"]:
        assert r["on_hand"] > 0
        assert r["branch"] and r["sku"]
        if r["dead"]:
            assert r["cover_weeks"] is None
            assert r["weekly_demand"] == 0
        else:
            assert r["cover_weeks"] is not None and r["cover_weeks"] > 12
    # dead (zero-sales) lines sort before the merely-excess ones
    seen_non_dead = False
    for r in out["rows"]:
        if not r["dead"]:
            seen_non_dead = True
        elif seen_non_dead:
            pytest.fail("a dead-stock row appeared after a non-dead excess row")
