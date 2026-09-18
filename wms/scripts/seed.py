"""Seed the WMS with demo data.

    python -m wms.scripts.seed

Loads: users, a 90-item catalogue (from Stock-Movement document 26503244) with
categories & prices, the company branches (official HansaWorld codes), ~90 days of per-branch sales
history, and the real delivery document 26503244 (DC -> Belmont Shop -> sales + one
back order for the shortfall). No fabricated back-order history.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from wms.db import Base, engine, SessionLocal
from wms.enums import SalesSource
from wms.models import User, Branch, Product, SalesRecord, BackOrder
from wms.security import hash_password
from wms.services import backorders as dn_service
from wms.scripts.sample_data import DOC_26503244, categorise, indicative_price

RNG = random.Random(26503244)

DEMO_PASSWORD = "wms1234"
USERS = [
    ("admin", "System Administrator", "admin"),
    ("controller", "Ivy Controller", "controller"),
    ("clerk", "Colin Clerk", "clerk"),
    ("branch", "Bella Branch", "branch"),
    ("analyst", "Ada Analyst", "analyst"),
]


def run() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        pw = hash_password(DEMO_PASSWORD)
        users = {u: User(username=u, full_name=fn, role=r, password_hash=pw)
                 for u, fn, r in USERS}
        db.add_all(users.values())
        db.flush()
        admin = users["admin"]

        products: dict[str, Product] = {}
        for sku, _req, _sent, desc in DOC_26503244["lines"]:
            p = Product(sku=sku, name=desc, category=categorise(desc), uom="EA",
                        unit_price=indicative_price(desc))
            db.add(p)
            products[sku] = p
        db.flush()

        # Official HansaWorld branch/location codes
        branch_defs = [
            ("BM", "Belmont Shop"),
            ("BTA", "Botswana"),
            ("DC", "DISTRIBUTION CENTER"),
            ("ES", "Esigodini"),
            ("ES2", "Esigodini 2"),
            ("FL", "Filabusi Mswela"),
            ("FLM", "Filabusi Mthwakazi"),
            ("FMS", "Filabusi Main Shop"),
            ("FWH", "Filabusi Warehouse"),
            ("GW", "Gweru"),
            ("GWA", "Gwanda VID"),
            ("GWL", "Gweru Luton Rd"),
            ("GWT", "Gwanda Thobelani"),
            ("JS", "Junkshop"),
            ("MP", "Maphisa"),
            ("TG", "Tongogara"),
            ("ZMA", "Zambia"),
        ]
        branches = [Branch(code=c, name=n) for c, n in branch_defs]
        db.add_all(branches)
        db.flush()
        by_code = {b.code: b for b in branches}
        belmont = by_code["BM"]
        users["branch"].branch_id = belmont.id

        # ~90 days of sales history for every branch
        today = date.today()
        for branch in branches:
            n_sku = 45 if branch is belmont else RNG.randint(18, 28)
            scale = 1.0 if branch is belmont else RNG.uniform(0.3, 0.8)  # smaller branches
            for sku in RNG.sample(list(products), n_sku):
                p = products[sku]
                mean_daily = RNG.choice([0.5, 1, 2, 3, 5, 8, 12, 20]) * scale
                for d in range(90, 0, -1):
                    qty = max(0, int(RNG.gauss(mean_daily, mean_daily * 0.6)))
                    if qty:
                        db.add(SalesRecord(branch_id=branch.id, product_id=p.id,
                                           sale_date=today - timedelta(days=d), qty=qty,
                                           unit_price=p.unit_price,
                                           source=SalesSource.POS.value))
        db.commit()

        # delivery document 26503244  (sent -> sales, shortfall -> back order)
        doc = DOC_26503244
        dn = dn_service.enter_delivery_note(
            db, branch_id=belmont.id,
            lines=[{"sku": s, "requested_qty": r, "sent_qty": (x or 0)}
                   for s, r, x, _ in doc["lines"]],
            from_location_label=doc["from_location"], doc_no=doc["doc_no"],
            doc_date=doc["doc_date"], comment=doc["comment"], user_id=admin.id,
        )

        print("Seed complete.")
        print(f"  users            : {len(USERS)}")
        print(f"  products         : {len(products)}")
        print(f"  branches         : {len(branches)}")
        print(f"  sales records    : {db.query(SalesRecord).count()}")
        print(f"  delivery note    : {dn.dn_no} ({len(doc['lines'])} lines)")
        print(f"  back orders      : {db.query(BackOrder).count()} "
              f"(open {db.query(BackOrder).filter(BackOrder.status == 'OPEN').count()})")
        print()
        print("Web app :  python -m uvicorn wms.api.main:app --port 8000   ->  http://127.0.0.1:8000")
        print("Console :  python -m wms.console            (add --demo for a headless tour)")
        print()
        print(f"Logins (password: {DEMO_PASSWORD}):")
        for u, fn, r in USERS:
            print(f"  {u:<12} {r}")
    finally:
        db.close()


if __name__ == "__main__":
    run()
