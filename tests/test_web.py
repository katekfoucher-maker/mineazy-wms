"""Browser app: auth, RBAC, ERP-style pages, document upload."""
import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def web(seeded):
    from wms.api.main import app
    return TestClient(app)


def _login(c, username, password="wms1234"):
    r = c.post("/login", data={"username": username, "password": password},
               follow_redirects=False)
    assert r.status_code == 303
    return c


def test_anonymous_is_redirected_to_login(web):
    r = web.get("/backorders", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_bad_credentials_rejected(web):
    r = web.post("/login", data={"username": "admin", "password": "nope"},
                 follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_home_redirects_to_flow_analysis(web):
    """Active Back Orders is temporarily off the nav, so home no longer
    lands there - it goes to Flow Analysis, which stays linked."""
    _login(web, "analyst")
    r = web.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/analysis"


def test_all_core_pages_render_with_erp_shell(web):
    _login(web, "controller")
    for path in ["/backorders", "/backorders?stage=OPEN", "/analysis",
                 "/dispatch/new", "/backorders/new", "/backorders/BO-000001",
                 "/delivery-notes/26503244",
                 "/analytics", "/analytics?tab=demand", "/analytics?tab=allocation",
                 "/reports"]:
        r = web.get(path)
        assert r.status_code == 200, path
        # ERP shell: white sidebar nav + account chip on the right
        assert 'class="account"' in r.text
        # Active Back Orders is temporarily off the nav (but a page like
        # /backorders can still legitimately say "Active Back Orders" in its
        # own heading, so check the exact nav-link text, not the page at large)
        assert "Active Back Orders</a>" not in r.text, path
        for label in ("Recon</a>", "Receiving Orders</a>", "Flow Analysis</a>",
                      "Allocation</a>", "Reports &amp; Exports</a>"):
            assert label in r.text, (path, label)


def test_flow_analysis_sections(web):
    _login(web, "analyst")
    r = web.get("/analysis")
    assert r.status_code == 200
    assert "<h1>Flow Analysis</h1>" in r.text or ">Flow Analysis<" in r.text
    assert "Worst performing products, overall" in r.text
    # revenue + gross margin live in the KPI strip here, not a separate page
    if "Revenue, last month" in r.text:
        assert "Gross margin" in r.text
    assert web.get("/revenue").status_code == 404       # standalone Revenue page removed
    # growth: KPIs + chart live here too, not a separate page
    if "Active branches" in r.text:
        assert "Active products" in r.text and "Growth" in r.text
        assert "Sales growth" in r.text and "growth-bar" in r.text
        assert 'name="growth_bcode"' in r.text            # branch selector on the chart
        assert 'value="BM" selected' in r.text             # Belmont is the default branch
        # the per-week sparkline breakouts were removed from the UI
        assert "Active products, weekly" not in r.text
        assert "Active branches, weekly" not in r.text
        # Sales growth now sits directly above the worst-performers card
        assert (r.text.index("Sales growth (month over month)")
                < r.text.index("Worst performing products, overall"))
    assert web.get("/growth").status_code == 404         # standalone Growth page removed
    # responsive layout guard: bare "1fr 1fr" grid columns don't shrink below
    # their content's min-content size and overflow the page at split-screen
    # widths (e.g. ~800-1100px); grid tracks must use minmax(0, 1fr) instead
    assert "grid-template-columns:1fr 1fr" not in r.text
    assert "grid-template-columns:minmax(0,1fr) minmax(0,1fr)" in r.text
    # responsive layout guard: below the 760px breakpoint the sidebar must
    # stay a plain full-width vertical stack. Making it flex-direction:row
    # (a prior version of this rule) puts the brand block and the nav list
    # side by side as two wrapping flex items, and since the brand block
    # gets stretched to the nav list's full height, its logo/label end up
    # vertically centered in the middle of the nav links instead of at the
    # top - a visibly broken, jumbled sidebar on any split-screen window.
    assert ".sidebar{width:100%;flex-direction:row" not in r.text
    assert "background:linear-gradient(90deg,var(--brand-deep),var(--brand))" not in r.text
    # removed sections
    for gone in ("Aging of open back orders", "Aging by current stage",
                 "Back orders vs sales, by branch", "Low sales: demand or supply?",
                 "Predicted vs actual sales, last week",
                 "Worst performing products, by branch"):
        assert gone not in r.text
    if "Sales by branch" in r.text:
        assert "Profit by branch" in r.text and 'id="bmix-filter"' in r.text
    # "Model comparison" now sits at the very bottom, after the worst-products card
    if "Model comparison, last week held out" in r.text:
        assert (r.text.index("Worst performing products, overall")
                < r.text.index("Model comparison, last week held out"))
    for gone in ("<h3>Product Performance</h3>",
                 "Stage funnel", "Cycle times", "Top products outstanding",
                 "Weekly trend", "the upstream signal",
                 "Fulfilment, cycle times, aging"):
        assert gone not in r.text


def test_worst_performers_product_links_to_sales_history(web):
    """Each product in 'Worst performing products' is a clickable control (not
    plain text) wired to pop up its sales-trend chart in a closable modal - so
    seeing a dead-stock item's own history is one click away instead of
    retyping its SKU into the trend search box by hand."""
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    _login(web, "analyst")
    r = web.get("/analysis")
    assert r.status_code == 200
    if "Nothing to flag here" in r.text:
        import pytest
        pytest.skip("no worst-performer rows in the sample data")
    assert 'class="wname-link"' in r.text
    assert "showProductHistory(event, '" in r.text
    assert "BRANCH_ID_BY_CODE" in r.text
    # the click handler reads the SKU it should query through the existing
    # sales-trend filter fields (restored afterwards, since this is a peek,
    # not a change to the page's own filter state)
    assert '#fa-filter input[name="sku"]' in r.text
    # the popup itself: starts hidden, and can be closed
    assert 'id="product-history-modal"' in r.text and "hidden" in r.text
    assert "closeProductHistory" in r.text


def test_flow_analysis_model_picker(web, tmp_path, monkeypatch):
    from wms.analytics import weekly_forecast as wfc
    monkeypatch.setattr(wfc, "_model_choice_path", lambda: tmp_path / "weekly_model.txt")
    wfc._CACHE.clear()
    _login(web, "controller")

    # the POST endpoint pins the model and persists the choice
    r = web.post("/analysis/model", data={"model": "snaive"},
                 follow_redirects=False)
    assert r.status_code == 303
    assert wfc.forced_model() == "snaive"
    r = web.post("/analysis/model", data={"model": ""},
                 follow_redirects=False)
    assert r.status_code == 303 and wfc.forced_model() == ""

    # when there is a hold-out comparison to show, the picker is on the page
    page = web.get("/analysis").text
    if "Model comparison, last week held out" in page:
        assert 'action="/analysis/model"' in page
        assert '>Auto (bias-aware pick)</option>' in page
        assert '<option value="snaive"' in page
    wfc.set_forced_model("")
    wfc._CACHE.clear()


def test_flow_analysis_retrain_route(web, monkeypatch):
    from wms.web import routes as wr
    calls = []
    monkeypatch.setattr(wr.subprocess, "Popen",
                        lambda *a, **k: calls.append((a, k)))
    _login(web, "controller")

    r = web.post("/analysis/retrain", data={}, follow_redirects=False)
    assert r.status_code == 303
    assert calls and "train_weekly" in " ".join(calls[0][0][0])   # spawned the trainer
    calls.clear()
    web.post("/analysis/retrain", data={"quick": "1"}, follow_redirects=False)
    assert "--quick" in calls[0][0][0]

    # a viewer without backorder.enter cannot trigger it
    _login(web, "analyst")
    calls.clear()
    r = web.post("/analysis/retrain", data={}, follow_redirects=False)
    assert r.status_code in (302, 303, 403) and not calls

    _login(web, "controller")
    page = web.get("/analysis").text
    if "Model comparison, last week held out" in page:
        assert 'action="/analysis/retrain"' in page
        assert "Saved models" in page


def test_allocation_tab_has_weekly_plan_no_upload(web):
    _login(web, "controller")
    r = web.get("/analytics?tab=allocation")
    assert r.status_code == 200
    assert "Split by predicted sales" in r.text           # the one card on this tab
    assert "Weekly orders" not in r.text                  # standalone card removed
    assert 'action="/analytics/order-request"' not in r.text
    assert 'action="/analytics/upload"' not in r.text     # inventory upload lives on Demand
    assert 'class="btn ghost dl-alloc"' not in r.text     # Excel button removed
    assert 'href="/download/allocation-plan?bcode' not in r.text
    assert 'id="fx-panel"' not in r.text                  # per-location plan table removed
    assert 'action="/analytics/split"' in r.text          # unified split form
    assert 'id="split-mode"' in r.text                    # One off / Weekly order
    # one selection box: pick a real Receiving Order to split, or upload a
    # quick one-off stock list instead
    assert 'id="ro-select"' in r.text
    assert 'name="stock_file"' in r.text
    assert 'id="split-weekly"' in r.text                  # per-branch order files


def test_split_upload_splits_each_line_by_predicted_sales(web):
    import io
    import re
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    st = wfc.cached_run()["state"]
    sku = st[st["weekly_demand"] > 0].iloc[0]["sku"]

    _login(web, "controller")
    body = f"SKU,Quantity\n{sku},600\n".encode()
    r = web.post("/analytics/split",
                 data={"split_mode": "oneoff"},
                 files={"stock_file": ("split_list.csv", io.BytesIO(body), "text/csv")})
    assert r.status_code == 200
    assert "Split result" in r.text and str(sku) in r.text
    block = r.text.split("Split result,")[1]
    # branches get some units; branch allocations + any warehouse hold-back
    # always add back up to the quantity entered (shown in the Total row)
    alloc_col = [int(x.replace(",", "")) for x in
                 re.findall(r'<td class="num fc-hi">([\d,]+)</td>', block)]
    assert alloc_col
    assert '<th class="num">600</th>' in block


def test_demand_tab_has_weekly_upload(web):
    _login(web, "controller")
    r = web.get("/analytics?tab=demand")
    assert r.status_code == 200
    assert 'action="/analytics/upload-weekly"' in r.text          # weekly sales
    # branch inventory now has its own page (see test_inventory_page_has_upload)
    assert 'name="kind" value="inventory"' not in r.text
    # the monthly-sales upload has no UI card (weekly upload is the front door),
    # but the route itself still works when posted directly - see
    # test_upload_monthly_sales_route_works_without_a_ui_card in test_weekly_simulate.py
    assert 'name="kind" value="sales"' not in r.text
    assert "Upload data" in r.text


def test_forecast_by_product_lists_every_branch(web, db):
    """The branch filter offers every real branch (not a hardcoded pair), and
    "All branches" actually shows rows from more than just the branch(es)
    with the most SKUs - branch_name is the primary sort key on the
    underlying data, so a naive row cap would let one big branch (e.g.
    Belmont Shop) crowd every other branch out of the preview entirely."""
    import re
    from wms.analytics import weekly_forecast as wfc
    from wms.models import Branch
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    _login(web, "controller")
    r = web.get("/analytics?tab=demand")
    assert r.status_code == 200

    # every branch in the DB is a selectable option, not just two hardcoded ones
    for b in db.query(Branch).all():
        assert f'value="{b.code}"' in r.text

    # the preview table itself spans more than one branch's Location column
    locations = set(re.findall(r"<tr>\s*<td>([^<]+)</td><td>", r.text))
    assert len(locations) > 1


def test_abc_classification_removed_from_ui_but_kept_in_backend(web):
    """The standalone "Equipment classification (ABC)" card (and its class /
    search filter form) is gone from the Allocation plan page, and so is the
    Split result's per-product Class badge (not useful there either) - but
    the underlying weekly_forecast.abc_classification() itself is still
    computed, since other code may still depend on the capability even
    though nothing in the UI renders it any more."""
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    _login(web, "controller")
    r = web.get("/analytics?tab=allocation")
    assert r.status_code == 200
    assert "Equipment classification (ABC)" not in r.text
    assert 'name="abc_class"' not in r.text and 'name="abc_q"' not in r.text

    out = wfc.abc_classification()
    assert out["has_data"] is True and out["rows"]


def test_allocation_split_tool_splits_by_predicted_sales(web):
    import re
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    st = wfc.cached_run()["state"]
    # a fast mover with a modest quantity: it all lands on branches, no hold-back
    sku = st.groupby("sku")["weekly_demand"].sum().idxmax()

    _login(web, "controller")
    r = web.post("/analytics/split",
                 data={"split_mode": "oneoff", "man_sku": str(sku), "man_qty": "40"})
    assert r.status_code == 200
    body = r.text.split("Split result,")[1]
    assert "Weekly sales" in body and ">Allocation<" in body
    # the removed narration is gone
    assert "sales over the last" not in body and "unallocated" not in body
    # the default "by product" view (the "by branch" view repeats the same
    # numbers in a differently-shaped table, so scope the sum to just this one)
    product_body = body.split('id="sort-by-branch"')[0]
    alloc_col = [int(x.replace(",", "")) for x in
                 re.findall(r'<td class="num fc-hi">([\d,]+)</td>', product_body)]
    assert alloc_col and sum(alloc_col) == 40
    assert '<th class="num">40</th>' in product_body        # Total row
    # the ABC class badge is gone from the split output - not useful there
    assert not re.search(r'abc-pill abc-[abc]"', body)
    assert "Class A" not in body and "abc-pill" not in body


def test_allocation_split_holds_back_slow_mover_stock(web):
    """A slow mover must not be dumped on one branch: cover a few weeks, seed a
    probe unit to branches with no recent sales, hold the rest at the warehouse."""
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    st = wfc.cached_run()["state"]
    agg = st.groupby("sku").agg(wd=("weekly_demand", "sum"),
                                ws=("weeks_sold", "max"))
    slow = agg[(agg.wd >= 1) & (agg.wd <= 3) & (agg.ws <= 4)]
    if slow.empty:
        import pytest
        pytest.skip("no slow mover in the sample data")
    sku = slow.index[0]

    _login(web, "controller")
    r = web.post("/analytics/split",
                 data={"split_mode": "oneoff", "man_sku": str(sku), "man_qty": "50"})
    assert r.status_code == 200
    body = r.text.split("Split result,")[1]
    assert "held at warehouse" in body.lower() or "Warehouse" in body
    assert '<th class="num">50</th>' in body        # branch + hold still totals 50
    # the per-product reasoning narration (why it's slow, why a branch got
    # probed/seeded, why stock was held back) is gone from the output - the
    # header's aggregate "held at warehouse" figure is enough
    assert "&#9878;" not in body
    assert "sized off the network's own lowest-selling branch" not in body
    assert "more than the branches can move" not in body


def test_weekly_order_generates_without_branch_files(web, db):
    """Weekly-order mode no longer requires an uploaded per-branch request
    file: with none attached, the order is generated straight from real
    weekly sales history + current stock - a real result, not an error."""
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    _login(web, "controller")
    r = web.post("/analytics/split", data={"split_mode": "weekly"})
    assert r.status_code == 200
    assert "Add at least one branch order file" not in r.text
    assert "system-generated" in r.text
    # every result line actually has a nested per-branch breakdown (real
    # branch-product need computed from sales history, not an empty stub)
    body = r.text.split("Split result,")[1]
    assert "<b>" in body and "Weekly sales" in body


def test_weekly_order_scopes_to_the_selected_branch(web, db):
    """Picking a branch for Weekly order must return that branch's own order
    document (one card, its SKUs, Rec. Qty based on its own sales history),
    not a network-wide product list spanning every branch."""
    from wms.analytics import weekly_forecast as wfc
    from wms.models import Branch
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    _login(web, "controller")
    r = web.post("/analytics/split",
                 data={"split_mode": "weekly", "split_branches": "BM"})
    assert r.status_code == 200
    other_branches = [b.name for b in db.query(Branch).all() if b.code != "BM"]
    body = r.text.split("Split result,")[1]
    assert "Belmont Shop" in body
    # no other branch's own order card appears on the page
    assert not any(name in body for name in other_branches)
    # nothing was actually requested (no file attached) - so the recommended
    # quantity is shown on its own, with no misleading "Requested" column
    # duplicating it (cut off before the trailing <script> blocks, which
    # contain an unrelated X-Requested-With fetch header)
    results = body.split("<script>")[0]
    assert "<th class=\"num\">Requested</th>" not in results
    # "On hand" justifies a Rec. Qty lower than Weekly sales
    assert "On hand" in results


def test_weekly_order_route_caps_to_uploaded_warehouse_stock(web, db):
    """The Weekly order route must actually use an uploaded warehouse quantity
    to cap the auto-generated recommendation across the selected branches -
    previously it silently ignored any warehouse data whenever no per-branch
    request file was attached, even if stock had been entered/uploaded."""
    from wms.web import routes as wr
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    uncapped = wr._run_auto_weekly_order(db, [])
    if not uncapped["rows"]:
        import pytest
        pytest.skip("no branch currently needs anything in the sample data")
    row = uncapped["rows"][0]
    sku, full_need = row["sku"], row["qty"]
    if full_need < 2:
        import pytest
        pytest.skip("need too small to meaningfully cap")
    half = full_need // 2

    _login(web, "controller")
    r = web.post("/analytics/split",
                 data={"split_mode": "weekly", "man_sku": str(sku), "man_qty": str(half)})
    assert r.status_code == 200
    assert "capped to warehouse stock" in r.text
    rbody = r.text.split("Split result,")[1]
    rresults = rbody.split("<script>")[0]
    assert "<th class=\"num\">Requested</th>" in rresults   # the cap makes the shortfall visible


def test_auto_weekly_order_respects_recency_and_cover(db):
    """Every branch-product the auto-generated weekly order includes must
    genuinely need it: real weekly demand, sold in roughly the last 4 weeks
    at that branch (not gone quiet), and on-hand short of a week's cover
    plus the ~3-day shipping delay - never a branch that's already stocked."""
    import numpy as np
    from wms.analytics import weekly_forecast as wfc
    from wms.services import stock as stock_svc
    from wms.web import routes as wr
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")

    batch = wr._run_auto_weekly_order(db, [])
    assert batch["weekly"] is True
    if not batch["rows"]:
        import pytest
        pytest.skip("no branch currently needs anything in the sample data")

    st = wfc.cached_run()["state"]
    rate_by = {(str(r.branch).upper(), str(r.sku).upper()): (float(r.weekly_demand or 0),
                                                              float(getattr(r, "recent_sales", 0) or 0))
               for r in st.itertuples()}
    inv = stock_svc.levels_df(db)
    on_hand = ({(str(r.branch_code).upper(), str(r.sku).upper()): int(r.on_hand or 0)
                for r in inv.itertuples()} if not inv.empty else {})
    code_by_name = {b["branch"]: b["code"] for b in batch["by_branch"]}

    for row in batch["rows"]:
        assert row["qty"] == sum(a["allocated"] for a in row["allocations"])
        for a in row["allocations"]:
            bc = code_by_name[a["branch"]]
            rate, recent = rate_by[(bc, row["sku"].upper())]
            assert rate > 0 and recent > 0            # actively & recently selling there
            target = int(np.ceil(rate * 10 / 7))
            oh = on_hand.get((bc, row["sku"].upper()), 0)
            assert oh < target                        # not already covered
            assert a["allocated"] == target - oh

    # rows are sorted by total network-wide need, highest first
    totals = [r["qty"] for r in batch["rows"]]
    assert totals == sorted(totals, reverse=True)


def test_auto_weekly_order_caps_to_warehouse_stock(db):
    """When a warehouse quantity is supplied, the auto-generated weekly order
    must never recommend more than what's actually on hand for a SKU, split
    across its branches weighted by their own weekly sales - the same "split
    based on branches and their sales" rule an uploaded request file already
    gets. A shortfall against real need is a real shortage, not stock sitting
    unshipped, so it must not feed the "held at warehouse" total."""
    from wms.web import routes as wr
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")

    uncapped = wr._run_auto_weekly_order(db, [])
    if not uncapped["rows"]:
        import pytest
        pytest.skip("no branch currently needs anything in the sample data")
    row = uncapped["rows"][0]           # the product with the largest network-wide need
    sku, full_need = row["sku"], row["qty"]
    if full_need < 2:
        import pytest
        pytest.skip("need too small to meaningfully cap")
    half = full_need // 2

    capped = wr._run_auto_weekly_order(db, [], warehouse_pairs=[(sku, half)])
    crow = next(r for r in capped["rows"] if r["sku"] == sku)
    total_allocated = sum(a["allocated"] for a in crow["allocations"])
    assert total_allocated <= half
    # the cap never gives a branch MORE than it would have gotten uncapped,
    # only less
    for a in crow["allocations"]:
        orig = next(x for x in row["allocations"] if x["branch"] == a["branch"])
        assert a["allocated"] <= orig["allocated"]
    assert crow["warehouse"] == 0 and capped["warehouse_total"] == 0
    # the per-product "note" (used by the combined Excel export's Reasoning
    # column) still records the shortfall, but no narration banner is shown
    # on screen for it any more
    assert crow["note"] and "short" in crow["note"]
    assert capped["warnings"] == []


def test_inventory_upload_feeds_the_weekly_plan(web, db, tmp_path, monkeypatch):
    import io
    import pandas as pd
    from wms.analytics import allocation, demand_forecast, inventory as inv_mod
    from wms.services import stock as stock_svc

    # isolate from the real data/inventory/ folder - this test previously wrote
    # a synthetic BM.xlsx over the real branch's live snapshot and then deleted
    # it in cleanup, permanently destroying real uploaded inventory data
    monkeypatch.setattr(inv_mod, "inventory_dir", lambda: tmp_path)
    _login(web, "controller")

    fc = demand_forecast.cached_run()["forecast"]
    bm = fc[fc.branch == "Belmont Shop"][["sku", "forecast_qty"]].head(6)
    xl = io.BytesIO()
    pd.DataFrame({"Item No": bm.sku, "Qty": [1] * len(bm)}).to_excel(xl, index=False)
    xl.seek(0)
    try:
        up = web.post("/analytics/upload",
                      data={"kind": "inventory", "branch_code": "BM"},
                      files={"file": ("BM.xlsx", xl,
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                      follow_redirects=True)
        # the upload writes straight into StockOnHand now (see inv_mod.parse_upload
        # in wms/web/routes.py) - it no longer saves the file to disk, so the DB
        # assertions below are the real check.
        assert up.status_code == 200

        # the general mechanism this test is about - the upload really did
        # write into StockOnHand - checked via the PLAIN lookup, since
        # Belmont's on-hand is deliberately excluded from allocation math
        # specifically (see stock.UNRELIABLE_FOR_ALLOCATION / test_allocation.py)
        real = stock_svc.levels_df(db)
        real_row = real[(real.branch_code == "BM") & (real.sku == bm.sku.iloc[0])]
        assert int(real_row["on_hand"].iloc[0]) == 1

        wk = allocation.weekly_allocation_plan(db, branch_code="belmont")
        assert not wk.empty
        row = wk[wk.sku == bm.sku.iloc[0]].iloc[0]
        assert row["on_hand"] == 0            # excluded from allocation, not the real 1
        # target covers the week + 3 transit days; nothing subtracted for Belmont
        assert row["target"] == -(-row["weekly_demand"] * 10 // 7)     # ceil
        assert row["to_transport"] == row["target"]
    finally:
        for f in inv_mod.inventory_dir().glob("BM.*"):
            f.unlink()
        from wms.models import Branch, StockOnHand
        bm_id = db.query(Branch).filter(Branch.code == "BM").scalar().id
        db.query(StockOnHand).filter(StockOnHand.branch_id == bm_id).delete()
        db.commit()


def test_inventory_page_shows_branch_summary_and_upload(web, db, tmp_path, monkeypatch):
    import io
    import pandas as pd
    from wms.analytics import inventory as inv_mod
    from wms.models import Branch, StockOnHand

    monkeypatch.setattr(inv_mod, "inventory_dir", lambda: tmp_path)
    _login(web, "controller")

    r = web.get("/inventory")
    assert r.status_code == 200
    assert 'action="/analytics/upload"' in r.text
    assert 'name="kind" value="inventory"' in r.text          # upload moved here
    assert "branch(es) reporting stock" not in r.text           # summary line removed
    assert '<th class="num">Product lines</th>' not in r.text and "Last updated" not in r.text  # per-branch table removed
    # dashboard: high-priority low-stock alert card
    assert "Low stock alerts" in r.text and "high priority" in r.text
    assert 'class="inv-kpi-value"' in r.text
    assert "Total stock" not in r.text                         # removed (duplicated the summary card's total)
    # excess/slow-moving stock: a computed planning view, shown in full on the
    # page (not just an API), with its own branch filter
    assert "Excess stock" in r.text and 'id="excess-stock"' in r.text
    assert 'name="excess_bcode"' in r.text
    # reorder points: removed from the UI (weekly_forecast.reorder_points()
    # itself stays - Suggested Orders / Allocation plan exports still use it)
    assert "Reorder now" not in r.text and 'id="reorder-points"' not in r.text
    assert 'name="reorder_bcode"' not in r.text
    # "Stock by product" now sits above "Low stock, high-priority products"
    assert r.text.index('id="stock-by-product"') < r.text.index('id="low-stock-list"')

    xl = io.BytesIO()
    pd.DataFrame({"Item No": ["ZZZ-TEST-SKU"], "Qty": [7]}).to_excel(xl, index=False)
    xl.seek(0)
    try:
        up = web.post("/analytics/upload",
                      data={"kind": "inventory", "branch_code": "BM"},
                      files={"file": ("BM.xlsx", xl,
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                      follow_redirects=True)
        assert up.status_code == 200
        assert up.request.url.path == "/inventory"        # the upload redirects here now

        page = web.get("/inventory")
        assert "ZZZ-TEST-SKU" in page.text

        only_bm = web.get("/inventory?bcode=BM&q=ZZZ-TEST")
        assert "ZZZ-TEST-SKU" in only_bm.text

        none_match = web.get("/inventory?q=NOT-A-REAL-SKU-XYZ")
        assert "Nothing matches" in none_match.text
    finally:
        bm_id = db.query(Branch).filter(Branch.code == "BM").scalar().id
        db.query(StockOnHand).filter(StockOnHand.branch_id == bm_id).delete()
        db.commit()


def test_products_page_add_search_and_update(web, db):
    """The Products module: add a new catalogue entry, find it by search, and
    posting the same SKU again updates it in place rather than duplicating
    it - the exact gap this closes is a SKU (like the real CTBN0104 case)
    that shows up in a branch-inventory upload with no name anywhere."""
    from wms.models import Product
    _login(web, "controller")
    try:
        r = web.get("/products")
        assert r.status_code == 200
        assert 'action="/products/new"' in r.text

        add = web.post("/products/new", data={
            "sku": "zztestprod1", "name": "Test Widget Mk1",
            "category": "Hand Tools", "uom": "EA", "unit_price": "",
        }, follow_redirects=True)
        assert add.status_code == 200
        assert add.request.url.path == "/products"

        found = web.get("/products?q=ZZTESTPROD1")
        assert "Test Widget Mk1" in found.text
        assert "Hand Tools" in found.text

        # posting the same SKU again updates, doesn't duplicate
        upd = web.post("/products/new", data={
            "sku": "ZZTESTPROD1", "name": "Test Widget Mk2",
            "category": "Hand Tools", "uom": "EA", "unit_price": "12.50",
        }, follow_redirects=True)
        assert "Updated product" in upd.text
        rows = db.query(Product).filter(Product.sku == "ZZTESTPROD1").all()
        assert len(rows) == 1
        assert rows[0].name == "Test Widget Mk2"
        assert float(rows[0].unit_price) == 12.50
    finally:
        db.query(Product).filter(Product.sku == "ZZTESTPROD1").delete()
        db.commit()


def test_products_new_requires_permission(web, db):
    """A viewer without products.manage (analyst) cannot add a product."""
    from wms.models import Product
    _login(web, "analyst")
    r = web.post("/products/new", data={"sku": "ZZNOPERM", "name": "Should Not Save"},
                follow_redirects=False)
    assert r.status_code in (302, 303, 403)
    assert db.query(Product).filter(Product.sku == "ZZNOPERM").first() is None


def test_account_and_role_shown_top_right(web):
    _login(web, "controller")
    r = web.get("/backorders")
    assert "Ivy Controller" in r.text and "Procurement Controller" in r.text


def test_active_back_orders_grid_matches_erp(web):
    _login(web, "analyst")
    r = web.get("/backorders")
    assert "Back Order Management" in r.text and "Active Back Orders" in r.text
    for col in ("Back order", "Branch", "Week", "Stage", "Lines",
                "Fulfillment", "Status", "Actions"):
        assert col in r.text
    assert "Order #" not in r.text            # tracked by weekly back-order no. now
    for label in ["All Stages", "Open", "Closed"]:
        assert label in r.text
    for gone in ['<option value="SUBMITTED"', '<option value="DISPATCHED"',
                 '<option value="CANCELLED"', "Warehouse Review", "Procurement Needed",
                 "Requisition", "PO Issued", "Goods Received", "Ready to Allocate"]:
        assert gone not in r.text


def test_inventory_pages_are_gone(web):
    _login(web, "controller")
    for path in ["/stock", "/movements", "/asns", "/counts", "/adjust"]:
        assert web.get(path).status_code == 404, path


def test_new_dispatch_form_fields(web):
    _login(web, "controller")
    html = web.get("/dispatch/new").text
    assert "New Recon" in html
    assert 'name="stock_movement_id"' in html          # replaces "Order ID"
    assert 'name="branch_id"' in html and 'name="doc_date"' in html
    assert 'name="requested_qty"' in html and 'name="sent_qty"' in html
    assert 'name="cycle"' in html and "Monthly" in html   # weekly/monthly cycle option
    assert ">Product<" in html and 'id="confirm"' in html  # Product column + confirm-to-submit
    assert "Confirm dispatch" in html
    assert "Order ID" not in html and "Order #" not in html
    assert 'action="/dispatch/new/upload"' in html     # the document upload
    # the legacy URL still resolves (redirects here)
    assert web.get("/backorders/new").status_code == 200


def test_dispatch_confirm_adds_stock_no_back_order(web, db):
    """Dispatch no longer computes/raises a back order at all (that
    calculation was retired) - it only records the delivery note and moves
    stock. The note's own requested-vs-sent-vs-shortfall is still shown on
    its detail page, just without ever creating a BackOrder row."""
    from wms.services import stock as stock_svc
    from wms.models import Branch, BackOrder
    gwa_id = db.query(Branch).filter(Branch.code == "GWA").first().id
    n_bo_before = db.query(BackOrder).count()
    _login(web, "controller")
    csv = ("Stock Movement,,,SM-UP-9\nTo Location,,Gwanda VID\nDate,,15/07/2026\n\n"
           "Item No,Req. Qty,Description,Sent Qty\n"
           "SFC1269,100,HELMET BLUE,40\n"
           "SFC1274,50,HELMET RED,\n"          # blank -> nothing dispatched
           "SFC1276,10,HELMET WHITE,10\n")
    r = web.post("/dispatch/new/upload",
                 files={"files": ("doc.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200
    assert 'value="SM-UP-9"' in r.text and 'value="2026-07-15"' in r.text
    assert 'value="SFC1274"' in r.text

    r = web.post("/dispatch/new", data={
        "branch_id": str(gwa_id), "stock_movement_id": "SM-UP-9", "doc_date": "2026-07-15",
        "cycle": "MONTHLY", "notes": "t",
        "sku": ["SFC1269", "SFC1274", "SFC1276"],
        "description": ["", "", ""],
        "requested_qty": ["100", "50", "10"],
        "sent_qty": ["40", "", "10"],
    }, follow_redirects=True)
    assert r.status_code == 200
    # landed on the delivery note itself, not a back order
    assert "SM-UP-9" in r.text
    assert "Requested" in r.text and "Sent" in r.text and "Backorder" in r.text
    assert db.query(BackOrder).count() == n_bo_before      # no back order raised

    # dispatched units are on the branch balance: 40 + 0 + 10
    lv = stock_svc.levels_df(db).set_index(["branch_code", "sku"])["on_hand"]
    assert lv.get(("GWA", "SFC1269")) == 40
    assert lv.get(("GWA", "SFC1276")) == 10

    # a second dispatch is its own independent delivery note
    r2 = web.post("/dispatch/new", data={
        "branch_id": str(gwa_id), "stock_movement_id": "SM-UP-10", "doc_date": "2026-07-17",
        "cycle": "WEEKLY", "notes": "",
        "sku": ["SFC1269"], "description": [""],
        "requested_qty": ["30"], "sent_qty": ["5"],   # short 25
    }, follow_redirects=True)
    assert r2.status_code == 200
    assert "SM-UP-10" in r2.text
    assert db.query(BackOrder).count() == n_bo_before      # still none
    lv2 = stock_svc.levels_df(db).set_index(["branch_code", "sku"])["on_hand"]
    assert lv2.get(("GWA", "SFC1269")) == 45              # 40 + 5

    # tidy the shared seeded DB
    from wms.models import DeliveryNote, StockOnHand
    gwa = db.query(Branch).filter(Branch.code == "GWA").first().id
    db.query(StockOnHand).filter(StockOnHand.branch_id == gwa).delete()
    for row in db.query(DeliveryNote).filter(DeliveryNote.branch_id == gwa).all():
        db.delete(row)
    db.commit()


def _make_stock_movement_pdf():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    rows = [
        "Mineazy Mining Solutions", "Stock Movement", "55501234",
        "To Location  GWA", "Date  15/07/2026",
        "Item No     Req. Qty   Description        Sent Qty   Rec. Qty",
        "SFC1269     100        HELMET BLUE        40",
        "SFC1274     50         HELMET RED",              # blank sent -> full backorder
        "SFC1276     10         HELMET WHITE       10",
    ]
    buf = io.BytesIO()
    with PdfPages(buf) as pp:
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95
        for r in rows:
            fig.text(0.08, y, r, family="monospace", fontsize=9)
            y -= 0.04
        pp.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


def test_upload_pdf_stock_movement_prefills_form(web):
    _login(web, "controller")
    r = web.post("/backorders/new/upload",
                 files={"files": ("Backorders.pdf", io.BytesIO(_make_stock_movement_pdf()),
                                 "application/pdf")})
    assert r.status_code == 200
    assert 'value="55501234"' in r.text            # stock movement id from PDF header
    assert 'value="2026-07-15"' in r.text          # date from PDF header
    assert 'value="SFC1274"' in r.text and 'value="SFC1276"' in r.text
    # GWA branch auto-selected
    assert 'selected' in r.text and "Gwanda VID" in r.text


def test_clerk_can_enter_controller_closes(web, db):
    # dispatch no longer raises a back order, so seed one directly (the
    # manual creation path - still very much part of the project) to exercise
    # the stage-advance permission check
    from wms.services import backorder_entry as bo_svc
    bo = bo_svc.create_back_order(db, branch_id=3, items=[{"sku": "SFC1276", "qty": 20}])

    _login(web, "clerk")
    # clerk cannot advance the stage
    r = web.post(f"/backorders/{bo.bo_no}/advance", data={"to_stage": "CLOSED"},
                 follow_redirects=False)
    assert r.status_code == 303

    _login(web, "controller")
    r = web.post(f"/backorders/{bo.bo_no}/advance", data={"to_stage": "CLOSED"},
                 follow_redirects=True)
    assert '<span class="pill CLOSED">Closed</span>' in r.text


def test_exports_download(web):
    _login(web, "analyst")
    for kind in ("backorders", "backorder-flow", "branch-sales"):
        r = web.get(f"/download/{kind}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def test_weekly_order_download_per_branch(web):
    _login(web, "analyst")
    r = web.get("/download/weekly-order?bcode=BM")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert "weekly_order_BM" in r.headers.get("content-disposition", "")


def test_allocation_plan_pdf_download(web):
    pytest.importorskip("fpdf")
    _login(web, "analyst")
    r = web.get("/download/allocation-plan-pdf?bcode=BM")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"
    assert "allocation_plan_BM" in r.headers.get("content-disposition", "")


def test_split_by_sales_branches_and_export(web):
    _login(web, "controller")
    # the unified split form carries a branch multi-select, and a one-off result
    # offers a single combined document (one Excel + one PDF) for the whole split
    page = web.post("/analytics/split",
                    data={"split_mode": "oneoff", "man_sku": "WIN001",
                          "man_qty": "1000", "split_branches": "BM"}).text
    assert 'type="checkbox" name="split_branches"' in page
    assert "Split result" in page

    x = web.get("/download/split-result?fmt=xlsx")
    assert x.status_code == 200
    assert x.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    p = web.get("/download/split-result?fmt=pdf")
    assert p.status_code == 200 and p.content[:5] == b"%PDF-"


def test_split_batch_export_is_one_document_for_many_products(web):
    _login(web, "controller")
    x = web.get("/download/split-batch?sku=WIN001&qty=600&sku=CTBMS1240&qty=250"
                "&alloc_branches=BM&alloc_branches=MP")
    assert x.status_code == 200
    assert x.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert int(x.headers["content-length"]) > 0
    p = web.get("/download/split-batch-pdf?sku=WIN001&qty=600&sku=CTBMS1240&qty=250")
    assert p.status_code == 200 and p.content[:5] == b"%PDF-"


def test_oneoff_split_sort_by_branch_and_per_branch_downloads(web):
    """A one-off split gets the same "sort by branch" toggle and per-branch /
    ZIP downloads as a weekly order, on top of the existing single combined
    document export - one file per branch, and one file for everything."""
    import re
    import zipfile
    import io
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    st = wfc.cached_run()["state"]
    sku = st.groupby("sku")["weekly_demand"].sum().idxmax()   # a real fast mover

    _login(web, "controller")
    r = web.post("/analytics/split",
                 data={"split_mode": "oneoff", "man_sku": str(sku), "man_qty": "5000"})
    assert r.status_code == 200
    assert 'id="sort-toggle"' in r.text
    assert 'id="sort-by-product"' in r.text and 'id="sort-by-branch"' in r.text
    assert 'href="/download/split-result/zip?doc=pdf"' in r.text
    assert 'href="/download/split-result/zip?doc=xlsx"' in r.text
    assert "Download this branch only" in r.text
    # the single combined document export still works alongside the new options
    assert 'href="/download/split-result?fmt=xlsx"' in r.text
    assert 'href="/download/split-result?fmt=pdf"' in r.text

    zp = web.get("/download/split-result/zip?doc=pdf")
    assert zp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zp.content)) as zf:
        names = zf.namelist()
        assert len(names) > 1
        assert all(n.endswith(".pdf") for n in names)
        assert all(zf.read(n)[:5] == b"%PDF-" for n in names)

    bcode = re.search(r'/download/split-result/branch/([A-Z0-9]+)\?fmt=pdf"', r.text).group(1)
    bp = web.get(f"/download/split-result/branch/{bcode}?fmt=pdf")
    assert bp.status_code == 200 and bp.content[:5] == b"%PDF-"


def test_unified_split_one_off_and_result_export(web):
    _login(web, "controller")
    r = web.post("/analytics/split",
                 data={"split_mode": "oneoff",
                       "man_sku": "WIN001", "man_qty": "40"})
    assert r.status_code == 200
    assert "Split result" in r.text and "one off" in r.text
    x = web.get("/download/split-result?fmt=xlsx")
    assert x.status_code == 200
    assert x.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    p = web.get("/download/split-result?fmt=pdf")
    assert p.status_code == 200 and p.content[:5] == b"%PDF-"


def test_unified_split_weekly_order_mode(web):
    import io
    _login(web, "controller")
    order = io.BytesIO(b"Item No,Description,Requested\nWIN001,WINPOW,20\n")
    r = web.post("/analytics/split",
                 data={"split_mode": "weekly",
                       "man_sku": "WIN001", "man_qty": "100",
                       "wk_branch": "BM"},
                 files={"wk_file": ("bm_order.csv", order, "text/csv")})
    assert r.status_code == 200
    assert "Split result" in r.text and "weekly order" in r.text
    assert "Requested" in r.text and "WIN001" in r.text
    # 100 in stock, branch asked for 20 -> 20 to the branch, 80 held at warehouse
    assert "held at warehouse" in r.text
    # the PDF export is a filled Stock-Movement dispatch note (Rec. Qty column)
    p = web.get("/download/split-result?fmt=pdf")
    assert p.status_code == 200 and p.content[:5] == b"%PDF-"


def test_weekly_order_zip_and_per_branch_downloads(web):
    """A weekly order covering more than one branch can be downloaded as one
    ZIP (one PDF or Excel per branch), or each branch on its own."""
    import io
    import zipfile
    _login(web, "controller")
    bm_order = io.BytesIO(b"Item No,Description,Requested\nWIN001,WINPOW,20\n")
    mp_order = io.BytesIO(b"Item No,Description,Requested\nWIN001,WINPOW,10\n")
    r = web.post("/analytics/split",
                 data={"split_mode": "weekly",
                       "man_sku": "WIN001", "man_qty": "100",
                       "wk_branch": ["BM", "MP"]},
                 files=[("wk_file", ("bm_order.csv", bm_order, "text/csv")),
                       ("wk_file", ("mp_order.csv", mp_order, "text/csv"))])
    assert r.status_code == 200
    assert "Split result" in r.text
    assert 'href="/download/split-result/zip?doc=pdf"' in r.text
    assert 'href="/download/split-result/zip?doc=xlsx"' in r.text

    zp = web.get("/download/split-result/zip?doc=pdf")
    assert zp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zp.content)) as zf:
        names = zf.namelist()
        assert len(names) == 2                          # one PDF per branch
        assert all(n.endswith(".pdf") for n in names)
        assert all(zf.read(n)[:5] == b"%PDF-" for n in names)

    zx = web.get("/download/split-result/zip?doc=xlsx")
    assert zx.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zx.content)) as zf:
        names = zf.namelist()
        assert len(names) == 2 and all(n.endswith(".xlsx") for n in names)

    bp = web.get("/download/split-result/branch/BM?fmt=pdf")
    assert bp.status_code == 200 and bp.content[:5] == b"%PDF-"
    bx = web.get("/download/split-result/branch/BM?fmt=xlsx")
    assert bx.status_code == 200
    assert bx.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # a branch not in the last split just bounces back to the tool, no crash
    bad = web.get("/download/split-result/branch/ZZ?fmt=pdf", follow_redirects=False)
    assert bad.status_code in (302, 303)


def test_weekly_order_works_without_any_inventory_data(web, monkeypatch):
    """A weekly order must be creatable from sales predictions and the
    branches' own requested quantities alone - no manual stock line, no
    uploaded stock file, no Receiving Orders on record. Previously the
    route demanded a stock figure up front and refused to run without one."""
    import io
    from wms.web import routes as wr

    monkeypatch.setattr(wr, "_dc_stock_pairs", lambda db: [])
    _login(web, "controller")
    order = io.BytesIO(b"Item No,Description,Requested\nWIN001,WINPOW,20\n")
    r = web.post("/analytics/split",
                 data={"split_mode": "weekly", "wk_branch": "BM"},
                 files={"wk_file": ("bm_order.csv", order, "text/csv")})
    assert r.status_code == 200
    assert "Split result" in r.text and "weekly order" in r.text
    assert "Requested" in r.text and "WIN001" in r.text
    # nothing was held back - there was no warehouse cap to hold anything against
    assert "held at warehouse" not in r.text


def test_receiving_page_loads_and_requires_permission(web):
    _login(web, "controller")
    r = web.get("/receiving/new")
    assert r.status_code == 200
    assert "Receiving Orders" in r.text and "Receiving details" in r.text
    assert 'action="/receiving/new/upload"' in r.text
    assert 'action="/receiving/new"' in r.text

    _login(web, "analyst")
    r = web.get("/receiving/new", follow_redirects=False)
    assert r.status_code in (302, 303)


def test_receiving_upload_matches_derived_sku_to_existing_catalogue_product(web, db):
    """A description-only invoice (no Item No column) should reuse an existing
    product's real SKU when its description matches, instead of prefilling
    the form with a SKU manufactured from the invoice text."""
    from wms.models import Product
    sku = "ZZEXIST-MTR1"
    p = Product(sku=sku, name="1.1KW-6 polo", uom="EA")
    db.add(p)
    db.commit()
    try:
        _login(web, "controller")
        csv = "Description,QTY (set)\n1.   Model : 1.1KW-6 polo,10sets\n"
        r = web.post("/receiving/new/upload",
                     files={"files": ("invoice.csv", io.BytesIO(csv.encode()), "text/csv")})
        assert r.status_code == 200
        assert f'value="{sku}"' in r.text
        assert "1.1KW-6POLO" not in r.text      # the manufactured SKU must not leak through
    finally:
        db.delete(p)
        db.commit()


def test_receiving_order_route_creates_updates_stock_and_reverses(web, db):
    """The full web flow: submit a receiving order, see it added to the
    warehouse's stock and listed in Receiving records, then reverse it and
    see both the record and the stock disappear again."""
    from wms.models import Branch, Product, ReceivingOrder, StockOnHand
    dc = db.query(Branch).filter(Branch.code == "DC").first()
    sku = "ZZRECVWEB1"
    try:
        _login(web, "controller")
        r = web.post("/receiving/new", data={
            "branch_id": str(dc.id), "ro_no": "RO-WEB-TEST-1",
            "doc_date": "2026-01-10", "supplier": "Acme",
            "sku": sku, "description": "Web test widget", "received_qty": "12",
        }, follow_redirects=True)
        assert r.status_code == 200
        assert "RO-WEB-TEST-1" in r.text
        soh = (db.query(StockOnHand)
              .filter(StockOnHand.branch_id == dc.id, StockOnHand.sku == sku).first())
        assert soh is not None and soh.qty_on_hand == 12

        r = web.post("/receiving/RO-WEB-TEST-1/delete", follow_redirects=True)
        assert r.status_code == 200
        # the flash message itself still names it ("Reversed receiving order
        # RO-WEB-TEST-1: ...") - check it's gone from the records table instead
        records = r.text.split("Receiving records")[1]
        assert "RO-WEB-TEST-1" not in records
        db.refresh(soh)
        assert soh.qty_on_hand == 0
    finally:
        row = db.query(ReceivingOrder).filter(
            ReceivingOrder.ro_no == "RO-WEB-TEST-1").first()
        if row:
            db.delete(row)
        db.query(StockOnHand).filter(StockOnHand.branch_id == dc.id,
                                     StockOnHand.sku == sku).delete()
        db.query(Product).filter(Product.sku == sku).delete()
        db.commit()


def test_recon_page_loads_and_requires_permission(web):
    _login(web, "controller")
    r = web.get("/recon/new")
    assert r.status_code == 200
    assert "Recon" in r.text and "Dispatch details" in r.text
    assert 'action="/recon/new/upload"' in r.text
    assert 'action="/recon/new"' in r.text
    # the destination can't be the warehouse itself
    assert '>DC —' not in r.text

    _login(web, "analyst")
    r = web.get("/recon/new", follow_redirects=False)
    assert r.status_code in (302, 303)


def test_recon_route_moves_stock_and_reverses(web, db):
    """The full web flow: receive stock, dispatch some of it to a branch, see
    the warehouse balance drop and the branch's rise, then reverse it."""
    from wms.models import Branch, DispatchOrder, Product, ReceivingOrder, StockOnHand
    dc = db.query(Branch).filter(Branch.code == "DC").first()
    bm = db.query(Branch).filter(Branch.code == "BM").first()
    sku = "ZZRECONWEB1"
    try:
        _login(web, "controller")
        web.post("/receiving/new", data={
            "branch_id": str(dc.id), "ro_no": "RO-RECONWEB-1",
            "doc_date": "2026-01-10", "sku": sku, "description": "Recon web test widget",
            "received_qty": "100",
        })
        r = web.post("/recon/new", data={
            "branch_id": str(bm.id), "do_no": "DO-WEB-TEST-1",
            "doc_date": "2026-01-11", "sku": sku, "description": "Recon web test widget",
            "dispatched_qty": "40",
        }, follow_redirects=True)
        assert r.status_code == 200
        assert "DO-WEB-TEST-1" in r.text

        dc_soh = db.query(StockOnHand).filter(StockOnHand.branch_id == dc.id,
                                              StockOnHand.sku == sku).first()
        bm_soh = db.query(StockOnHand).filter(StockOnHand.branch_id == bm.id,
                                              StockOnHand.sku == sku).first()
        assert dc_soh.qty_on_hand == 60 and bm_soh.qty_on_hand == 40

        r = web.post("/recon/DO-WEB-TEST-1/delete", follow_redirects=True)
        assert r.status_code == 200
        records = r.text.split("Recon records")[1]
        assert "DO-WEB-TEST-1" not in records
        db.refresh(dc_soh)
        db.refresh(bm_soh)
        assert dc_soh.qty_on_hand == 100 and bm_soh.qty_on_hand == 0
    finally:
        for m, no in ((DispatchOrder, "DO-WEB-TEST-1"), (ReceivingOrder, "RO-RECONWEB-1")):
            row = db.query(m).filter(getattr(m, "do_no" if m is DispatchOrder else "ro_no") == no).first()
            if row:
                db.delete(row)
        db.query(StockOnHand).filter(StockOnHand.sku == sku).delete()
        db.query(Product).filter(Product.sku == sku).delete()
        db.commit()


def test_receiving_order_feeds_the_split_tool(web, db):
    """Stock added via a Receiving Order must actually be what the Split
    tool falls back to when nothing is manually entered - closing the loop
    from "Receiving Orders" back into "Sales & Forecasting"."""
    from wms.models import Branch, Product, ReceivingOrder, StockOnHand
    from wms.analytics import weekly_forecast as wfc
    if not wfc.has_data():
        import pytest
        pytest.skip("no weekly_sales files present")
    dc = db.query(Branch).filter(Branch.code == "DC").first()
    st = wfc.cached_run()["state"]
    sku = str(st.groupby("sku")["weekly_demand"].sum().idxmax())  # a real fast mover
    try:
        _login(web, "controller")
        web.post("/receiving/new", data={
            "branch_id": str(dc.id), "ro_no": "RO-WEB-TEST-2",
            "doc_date": "2026-01-10", "sku": sku, "received_qty": "77",
        })
        r = web.post("/analytics/split", data={"split_mode": "oneoff"})
        assert r.status_code == 200
        body = r.text.split("Split result,")[1]
        assert sku.upper() in body.upper()
        assert '<th class="num">77</th>' in body        # the received qty, in full
    finally:
        row = db.query(ReceivingOrder).filter(
            ReceivingOrder.ro_no == "RO-WEB-TEST-2").first()
        if row:
            db.delete(row)
        db.query(StockOnHand).filter(StockOnHand.branch_id == dc.id,
                                     StockOnHand.sku == sku).delete()
        db.commit()
