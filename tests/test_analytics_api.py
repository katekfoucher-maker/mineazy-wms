"""Statistics, sales analysis, forecasting, allocation - API surface."""


def test_overall_and_branch_stats(client):
    o = client.get("/api/analytics/overall").json()
    assert o["products"] == 90
    assert o["sales_qty_90d"] > 0
    assert "open_back_orders" in o and "bottleneck_stage" in o

    b = client.get("/api/analytics/branch/1").json()
    assert "avg_daily_demand" in b and "sales_qty_90d" in b


def test_branch_sales_summary(client):
    rows = client.get("/api/analytics/branch-sales").json()
    assert rows and all("sales_qty" in r and "trend_pct" in r for r in rows)


def test_abc_by_sales_value(client):
    rows = client.get("/api/analytics/abc").json()
    assert rows and {r["class"] for r in rows} <= {"A", "B", "C"}


def test_suggested_orders_anticipate_demand(client):
    rows = client.get("/api/analytics/suggested-orders?branch_id=1").json()
    assert rows and all(r["suggested_order_qty"] > 0 for r in rows)
    # order-up-to-target: suggested == ceil(target_level)
    assert all(r["suggested_order_qty"] >= r["target_level"] - 1 for r in rows)


def test_allocation_respects_available_qty(client):
    fc = client.get("/api/analytics/suggested-orders?branch_id=1").json()
    pid = fc[0]["product_id"]
    r = client.post(f"/api/analytics/allocate?product_id={pid}&available_qty=100").json()
    assert r["available_qty"] == 100
    assert r["allocated_total"] <= 100
    assert sum(a["allocated_qty"] for a in r["allocations"]) == r["allocated_total"]


def test_audit_trail_has_user_tracking(client):
    r = client.get("/api/reports/audit?limit=20").json()
    assert r and any(row["user_id"] for row in r)


def test_exports_stream_files(client):
    for path, ct in [("/api/reports/backorder-flow.xlsx", "spreadsheetml"),
                     ("/api/reports/branch-sales.xlsx", "spreadsheetml"),
                     ("/api/reports/sales.csv", "text/csv"),
                     ("/api/reports/back-orders.csv", "text/csv")]:
        rr = client.get(path)
        assert rr.status_code == 200 and ct in rr.headers["content-type"]
