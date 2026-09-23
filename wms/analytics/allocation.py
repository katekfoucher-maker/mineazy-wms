"""Sales-weighted allocation of a constrained quantity to branches.

You supply how much of a product is available; the system splits it across the
branches that have demand for it:

  1. rank branches by soonest need (highest avg daily demand first),
  2. pass 1 - bring each branch up to its reorder point in rank order,
  3. pass 2 - split any remainder proportional to trailing demand share,
  4. round down to whole units.

There is no stock-on-hand tracking, so "need" is measured against the reorder
point (avg*lead_time + safety), not against a current balance.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from wms.models import Branch, Product

_WEEKS_PER_MONTH = 365 / 7 / 12          # ~4.345


def _real_forecast_table(db: Session) -> pd.DataFrame:
    """Real per-branch-product demand + reorder-point statistics, replacing
    the old ``forecast.forecast_table()`` (built on the SalesRecord table's
    fabricated seed history - see seed.py). Sourced from
    :func:`wms.analytics.weekly_forecast.reorder_points`, joined to the DB
    Branch/Product tables so branch_id/product_id-keyed callers (the JSON
    API) keep working. Only SKUs that also exist in the (small, legacy)
    Product catalog can appear here - branch_id/product_id are DB-row
    concepts the ~10k-SKU weekly panel doesn't otherwise carry."""
    from wms.analytics import weekly_forecast as wfc
    rows = wfc.reorder_points(db, limit=100000)["rows"]
    if not rows:
        return pd.DataFrame()
    branch_by_code = {b.code.upper(): b.id for b in db.query(Branch).all()}
    prod_by_sku = {p.sku.upper(): p for p in db.query(Product).all()}
    out = []
    for r in rows:
        p = prod_by_sku.get(str(r["sku"]).upper())
        bid = branch_by_code.get(str(r["branch_code"]).upper())
        if not p or not bid:
            continue
        out.append({
            "branch_id": bid, "branch": r["branch"], "sku": r["sku"],
            "product_id": p.id, "description": p.name,
            "avg_daily_demand": round(r["avg_weekly_sales"] / 7, 3),
            "reorder_point": float(r["reorder_point"]),
            "target_level": float(r["reorder_point"]),
            "suggested_order_qty": max(0, r["reorder_point"] - r["on_hand"]),
        })
    return pd.DataFrame(out)


def weekly_allocation_plan(db: Session, *, branch_code: str = "", q: str = ""
                           ) -> pd.DataFrame:
    """What to send each branch for the coming week, per product.

    weekly demand  = per-SKU weekly forecast   (the weekly model when weekly
                     sales files have been uploaded, else next-month sales
                     forecast / 4.345 rounded up)
    target         = ceil(weekly demand x (7 + dispatch_transit_days) / 7) -
                     enough to cover the review week and the days in transit
    to transport   = max(0, target - stock on hand)

    Stock on hand is the live per-branch balance in ``stock_on_hand`` (an
    inventory upload replaces a branch's balance; a confirmed dispatch adds its
    *sent* quantities), falling back to an uploaded spreadsheet snapshot only
    when that table is empty. Branches/products with nothing to move are
    dropped. Filter by ``branch_code`` and by a product ``q`` (SKU or name).
    """
    from wms.analytics import demand_forecast as dfc
    from wms.analytics import weekly_forecast as wfc
    from wms.analytics import inventory as inv_mod
    from wms.services import stock as stock_svc
    from wms.config import get_settings

    s = get_settings()
    transit = getattr(s, "dispatch_transit_days", 3)
    cover_days = getattr(s, "review_period_days", 7) + transit      # 7 + 3 = 10
    cols = ["branch", "sku", "product", "weekly_demand", "target", "on_hand",
            "to_transport"]

    inv = stock_svc.levels_df_for_allocation(db)      # DB balance is authoritative
    if inv.empty:
        inv = inv_mod.load_inventory()               # fall back to an uploaded snapshot
    on_hand = (inv.set_index(["branch_code", "sku"])["on_hand"]
               if not inv.empty else pd.Series(dtype=float))

    if wfc.has_data():
        st = wfc.cached_run()["state"]
        # a recent sales spike can outpace the model before it catches up
        # (e.g. a branch was out of stock for a while and demand is only now
        # showing up in the raw numbers) - never target BELOW what's actually
        # been selling these last few weeks, even if the model's own forecast
        # hasn't caught up to it yet
        weekly_demand = np.maximum(st["weekly_demand"].astype(int),
                                   st["recent_sales"].astype(int))
        r = pd.DataFrame({
            "branch": st["branch_name"],
            "branch_code": st["branch"],
            "sku": st["sku"],
            "product": st["item"],
            "weekly_demand": weekly_demand,
        })
    else:
        fc = dfc.cached_run()["forecast"]
        if fc.empty:
            return pd.DataFrame(columns=cols)
        code_by_name = {b.name: b.code for b in db.query(Branch).all()}
        r = fc[["branch", "sku", "item", "forecast_qty"]].copy()
        r["branch_code"] = r["branch"].map(code_by_name)
        r["weekly_demand"] = np.ceil(r["forecast_qty"] / _WEEKS_PER_MONTH).astype(int)
        r = r.rename(columns={"item": "product"})

    # top branches up to cover the review week PLUS the dispatch transit days, so
    # stock stays on the shelf while the transfer is on the road
    r["target"] = np.ceil(r["weekly_demand"] * cover_days / 7).astype(int)
    r["on_hand"] = [int(on_hand.get((bc, sk), 0)) for bc, sk in zip(r.branch_code, r.sku)]
    r["to_transport"] = (r["target"] - r["on_hand"]).clip(lower=0).astype(int)

    if branch_code:
        bc = branch_code.strip().lower()
        r = r[r["branch_code"].str.lower().eq(bc)
              | r["branch"].str.lower().str.startswith(bc)]
    if q:
        s = q.strip().lower()
        r = r[r["sku"].str.lower().str.contains(s, regex=False)
              | r["product"].str.lower().str.contains(s, regex=False)]

    r = r[r["to_transport"] > 0]
    return (r[cols].sort_values(["branch", "to_transport"], ascending=[True, False])
            .reset_index(drop=True))


# --- how aggressively a branch may be stocked from a one-off split -------------
# A split tops a branch up to roughly one sales week plus the dispatch transit,
# and no further - it is not a bulk restock. Anything a branch cannot sell in
# that window is held at the warehouse (it makes no sense to send 800 to a
# branch that sells 300).
_COVER_WEEKS      = 1.5    # normal mover: ~1 week of sales + a few days in transit
_SLOW_COVER_WEEKS = 2      # slow mover: 2 weeks, so a ~1/wk item still ships 2 not 0
_SLOW_NET_WEEKLY  = 3.0    # whole-network forecast at/below this (units/wk) = slow mover
_SLOW_WEEKS_SOLD  = 4      # ... or the busiest branch has sold in <= this many weeks

# a much stricter bar than "slow" above - informational only (how barely a
# product moves network-wide), no longer changes how a probe is sized - see
# the probe step below, which sizes off branch size + product performance
# elsewhere instead of this threshold.
_VERY_SLOW_NET_WEEKLY = 1.0
_VERY_SLOW_WEEKS_SOLD = 2

# probing untested branches (see allocate_by_forecast): a probe below this
# many units is a fragment, not a real test, so it caps how many branches
# share what's left of a split; _PROBE_RESERVE_FRACTION additionally keeps
# that fraction of the remaining quantity held back at the warehouse
# whenever more than one branch is being probed at once.
_MIN_PROBE_QTY = 3
_PROBE_RESERVE_FRACTION = 0.25


def _split_capped(total: int, weights: dict, caps: dict) -> dict:
    """Hand out ``total`` whole units across the keys of ``caps`` roughly in
    proportion to ``weights``, never giving a key more than ``caps[key]``.
    Leftover from a capped-out key spills to the others; the biggest weight
    breaks ties on the final rounding units."""
    alloc = {c: 0 for c in caps}
    keys = [c for c in caps if caps[c] > 0]
    total = min(int(total), sum(caps[c] for c in keys))
    while total > 0 and keys:
        wsum = sum(max(0.0, weights.get(c, 0.0)) for c in keys)
        if wsum <= 0:
            wts = {c: 1.0 for c in keys}
            wsum = float(len(keys))
        else:
            wts = {c: max(0.0, weights.get(c, 0.0)) for c in keys}
        handed = 0
        for c in keys:
            give = min(int(total * wts[c] / wsum), caps[c] - alloc[c])
            if give > 0:
                alloc[c] += give
                handed += give
        if handed == 0:                       # rounding stalled - give 1 to the top
            for c in sorted(keys, key=lambda k: wts[k], reverse=True):
                if total - handed <= 0:
                    break
                if caps[c] - alloc[c] > 0:
                    alloc[c] += 1
                    handed += 1
        total -= handed
        keys = [c for c in keys if caps[c] - alloc[c] > 0]
    return alloc


def allocate_by_forecast(db: Session, *, sku, qty: int,
                         branch_codes=None) -> dict:
    """Split ``qty`` units of one product across branches **without letting any
    branch hoard stock it cannot move**.

      1. cover  - top each selling branch up to about one sales week plus the
                  dispatch transit (less stock already on hand), best sellers
                  first. That is the ceiling: a split is not a bulk restock, so
                  a branch selling 300/wk is never sent 800.
      2. seed a branch with no sales history for this product (and none on
                  hand), out of whatever is left after cover, so it gets a
                  chance to show whether it sells there too. How much depends
                  on how well the product actually moves:
                    - a genuinely barely-moving product (see
                      ``_VERY_SLOW_NET_WEEKLY``/``_VERY_SLOW_WEEKS_SOLD``) only
                      gets a token 1-2 unit probe - not worth committing more.
                    - anything else, ordinary "slow" movers included, gets a
                      real seed instead: roughly half to three quarters of
                      what the network's own lowest-selling (but actively
                      selling) branch already received, so a fast mover with
                      stock left over after covering its existing branches
                      isn't wasted on a single test unit at a new one.
      3. hold   - everything still spare stays at the warehouse rather than
                  being pushed onto a branch that cannot move it.

    ``branch_codes`` (optional) restricts the split to those branches (by code).
    ``None``/empty = every operating branch. History (the weekly model's blended
    rate and weeks-sold count) drives every cap, so a product that sells once a
    quarter is treated very differently from a fast mover.
    """
    from wms.analytics import weekly_forecast as wfc
    from wms.services import stock as stock_svc

    ref = str(sku).strip()
    # accept "NAME (SKU)" as pasted from the pick list
    m = re.search(r"\(([^()]+)\)\s*$", ref)
    if m:
        ref = m.group(1).strip()
    p = (db.query(Product).filter(Product.id == int(ref)).first()
         if ref.isdigit()
         else db.query(Product).filter(Product.sku == ref).first())
    sku_code = p.sku if p else ref
    name = p.name if p else ref

    qty = max(0, int(qty or 0))
    st = wfc.cached_run()["state"]
    rows_fc = st[st["sku"].str.lower() == sku_code.lower()] if not st.empty else st
    # not a SKU? the user may have typed a product name - resolve it against the
    # weekly forecast (exact name first, then a unique substring match)
    if (rows_fc is None or rows_fc.empty) and not st.empty and ref:
        exact = st[st["item"].str.lower() == ref.lower()]
        part = (exact if not exact.empty
                else st[st["item"].str.contains(re.escape(ref), case=False, na=False)])
        if not part.empty and part["sku"].str.upper().nunique() == 1:
            sku_code = str(part["sku"].iloc[0])
            rows_fc = st[st["sku"].str.lower() == sku_code.lower()]
    # weekly-only SKUs are not always in the Product table - take the descriptive
    # name from the weekly sales files
    if (not name or name == sku_code or name == ref) and \
            rows_fc is not None and not rows_fc.empty:
        _nm = str(rows_fc["item"].iloc[0] or "").strip()
        if _nm and _nm.lower() != sku_code.lower():
            name = _nm

    picked = sorted({str(c).strip().upper() for c in (branch_codes or []) if str(c).strip()})

    # branch universe: the picked branches, else every operating branch
    if not st.empty:
        uni = st[["branch", "branch_name"]].drop_duplicates()
        universe = [(str(c).upper(), n) for c, n in
                    zip(uni["branch"], uni["branch_name"])]
    else:
        universe = []
    if picked:
        universe = [(c, n) for c, n in universe if c in picked]

    # a product with no forecast history anywhere (or none at the chosen
    # branches) isn't "no demand" any more than an untested branch is - it's
    # simply unproven. Rather than dumping the whole quantity to warehouse
    # hold, fall through with an empty per_sku: every branch in scope ends up
    # "untested" below and gets a real, branch-size-scaled probe instead.
    per_sku = rows_fc if rows_fc is not None else st.iloc[0:0]
    if picked:
        per_sku = per_sku[per_sku["branch"].str.upper().isin(picked)]

    uni_codes = {c for c, _n in universe}
    rate: dict = {}
    wsold: dict = {}
    for r in per_sku.itertuples():
        c = str(r.branch).upper()
        if uni_codes and c not in uni_codes:
            continue
        rate[c] = max(float(getattr(r, "weekly_demand", 0) or 0),
                      float(getattr(r, "recent_sales", 0) or 0))   # recent_sales is a wk rate
        wsold[c] = int(getattr(r, "weeks_sold", 0) or 0)

    # branches selling this SKU that are missing from the universe list (defensive)
    for c in rate:
        if c not in uni_codes:
            universe.append((c, c))
            uni_codes.add(c)

    inv = stock_svc.levels_df_for_allocation(db)
    on_hand: dict = {}
    if not inv.empty:
        sub = inv[inv["sku"].str.lower() == sku_code.lower()]
        for r in sub.itertuples():
            on_hand[str(r.branch_code).upper()] = int(getattr(r, "on_hand", 0) or 0)

    branches = sorted(universe, key=lambda t: rate.get(t[0], 0.0), reverse=True)
    net_weekly = float(sum(rate.values()))
    slow = (net_weekly <= _SLOW_NET_WEEKLY) or (
        bool(wsold) and max(wsold.values()) <= _SLOW_WEEKS_SOLD)
    very_slow = (net_weekly <= _VERY_SLOW_NET_WEEKLY) or (
        bool(wsold) and max(wsold.values()) <= _VERY_SLOW_WEEKS_SOLD)
    cover = _SLOW_COVER_WEEKS if slow else _COVER_WEEKS

    # each branch's own overall size (its total weekly demand across every
    # product it sells) - the probe step below uses this so an untested
    # branch's test quantity reflects how much it sells in general, not just
    # a flat token amount regardless of whether it's Belmont or a corner shop
    branch_size: dict = {}
    if not st.empty:
        for bc, tot in st.groupby(st["branch"].str.upper())["weekly_demand"].sum().items():
            branch_size[bc] = float(tot)

    alloc = {c: 0 for c, _n in branches}
    remaining = qty

    def _fill(cap_weeks: float) -> None:
        """Top selling branches up towards ``cap_weeks`` weeks of their own
        demand (less stock on hand and what they already got), splitting what's
        available in proportion to each branch's weekly rate."""
        nonlocal remaining
        if remaining <= 0:
            return
        room = {}
        for c, _n in branches:
            rt = rate.get(c, 0.0)
            if rt <= 0:
                continue
            free = int(np.ceil(rt * cap_weeks)) - on_hand.get(c, 0) - alloc[c]
            if free > 0:
                room[c] = free
        if not room:
            return
        got = _split_capped(remaining, {c: rate[c] for c in room}, room)
        for c, a in got.items():
            alloc[c] += a
            remaining -= a

    _fill(cover)

    probes: dict = {}
    if remaining > 0:
        untested = [c for c, _n in branches
                   if rate.get(c, 0.0) <= 0 and on_hand.get(c, 0) == 0
                   and alloc[c] == 0]
        if untested:
            # an untested branch (never sold this product, none on hand) still
            # deserves a real test quantity, not a token unit - sized off how
            # well this product already performs at the branches that DO
            # carry it (yield per unit of branch size), scaled by the
            # untested branch's OWN size (its total weekly demand across
            # every product it sells). A flagship branch like Belmont gets a
            # meaningfully bigger probe than a small branch for a product
            # neither has ever stocked, because its general sales exposure
            # says it's more likely to move this one too. When nothing sells
            # the product anywhere yet, fall back to splitting what's left by
            # branch size alone.
            covered = [c for c, a in alloc.items() if a > 0]
            per_size = [alloc[c] / branch_size[c] for c in covered
                       if branch_size.get(c, 0.0) > 0]
            yield_per_size = (sum(per_size) / len(per_size)) if per_size else 0.0

            # thin stock spread across every untested branch turns into
            # fragments that tell the network nothing - test a few branches
            # properly (biggest first) and hold the rest, instead of probing
            # everywhere at once. And when there IS enough to probe several
            # branches, don't spend the whole remaining quantity on it either
            # - a split still isn't a bulk restock, so some stays held back.
            ranked = sorted(untested, key=lambda c: -branch_size.get(c, 0.0))
            probe_budget = (remaining if len(ranked) <= 1
                            else int(remaining * (1 - _PROBE_RESERVE_FRACTION)))
            max_branches = max(1, probe_budget // _MIN_PROBE_QTY)
            ranked = ranked[:max_branches]

            untested_size_total = sum(branch_size.get(c, 0.0) for c in ranked)
            for c in ranked:
                if remaining <= 0 or probe_budget <= 0:
                    break
                size = branch_size.get(c, 0.0)
                if yield_per_size > 0:
                    give = int(round(yield_per_size * size))
                elif untested_size_total > 0:
                    give = int(round(probe_budget * size / untested_size_total))
                else:
                    give = 1
                give = max(1, min(give, remaining, probe_budget))
                alloc[c] += give
                probes[c] = give
                remaining -= give
                probe_budget -= give

    # everything left over stays at the warehouse - a split does not bulk-restock
    warehouse = max(0, remaining)

    # every branch in scope is reported, even at 0 units - a branch that got
    # nothing (already covered from its own on-hand stock, or no demand here)
    # must say so rather than silently vanish from the table, which read as a
    # missing/buggy branch once the split started covering many branches at once
    allocs = []
    for c, n in branches:
        a = alloc[c]
        rt = rate.get(c, 0.0)
        oh = on_hand.get(c, 0)
        if a > 0:
            kind = "probe" if c in probes else "cover"
        elif rt > 0:
            kind = "held"           # ran out of quantity to allocate here
        else:
            kind = "untested"       # never probed - not the same as "no demand"
        allocs.append({
            "branch": n, "predicted": int(round(rt)), "allocated": int(a),
            "kind": kind,
            "cover_weeks": (round((a + oh) / rt, 1) if rt > 0 else None),
        })
    allocs.sort(key=lambda d: (d["allocated"], d["predicted"]), reverse=True)

    return {
        "product": sku_code, "description": name, "qty": qty,
        "allocated_total": int(sum(alloc.values())),
        "warehouse": int(warehouse),
        "allocations": allocs, "branches": picked,
        "slow": bool(slow), "very_slow": bool(very_slow),
        "cover_weeks": cover, "note": None,
    }


def allocate_product(
    db: Session, *, product_id: int, available_qty: int,
    branch_ids: Optional[list[int]] = None,
    policy: str = "fill_to_rop_then_fair_share",
    _fc: Optional[pd.DataFrame] = None,
) -> dict:
    product_id = int(product_id)
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        return {"error": "product not found"}

    fc = _real_forecast_table(db) if _fc is None else _fc
    if fc.empty:
        return {"product": product.sku, "available_qty": available_qty,
                "allocations": [], "note": "no demand history"}
    fc = fc[fc.product_id == product_id].copy()
    if branch_ids:
        fc = fc[fc.branch_id.isin(branch_ids)]
    if fc.empty:
        return {"product": product.sku, "available_qty": available_qty,
                "allocations": [], "note": "no demand for this product"}

    fc = fc.sort_values("avg_daily_demand", ascending=False)
    remaining = int(available_qty)
    alloc = {int(r.branch_id): 0 for r in fc.itertuples()}

    for r in fc.itertuples():                       # pass 1: up to reorder point
        if remaining <= 0:
            break
        give = min(int(np.ceil(r.reorder_point)), remaining)
        alloc[int(r.branch_id)] += give
        remaining -= give

    if remaining > 0:                               # pass 2: demand-weighted
        share = fc.set_index("branch_id")["avg_daily_demand"]
        tot = share.sum()
        if tot:
            for bid, frac in (share / tot).items():
                alloc[int(bid)] += int(np.floor(remaining * frac))

    tot_demand = float(fc.avg_daily_demand.sum()) or 1.0
    out = [{
        "branch_id": int(r.branch_id), "branch": r.branch,
        "avg_daily_demand": float(r.avg_daily_demand),
        "reorder_point": float(r.reorder_point),
        "target_level": float(r.target_level),
        "allocated_qty": int(alloc[int(r.branch_id)]),
        "demand_share_pct": round(float(r.avg_daily_demand) / tot_demand * 100, 1),
        "days_covered": round(alloc[int(r.branch_id)] / r.avg_daily_demand, 1)
            if r.avg_daily_demand else None,
    } for r in fc.itertuples()]
    return {
        "product": product.sku, "description": product.name,
        "available_qty": int(available_qty),
        "allocated_total": int(sum(alloc.values())),
        "unallocated": int(available_qty - sum(alloc.values())),
        "policy": policy,
        "allocations": sorted(out, key=lambda x: -x["allocated_qty"]),
    }


def allocation_plan(db: Session) -> pd.DataFrame:
    """Suggested orders rolled up per product x branch (the demand picture with no
    supply constraint applied - use ``allocate_by_forecast`` for a constrained
    split). Real reorder-point statistics from actual weekly sales history and
    current on-hand stock (see weekly_forecast.reorder_points) - not the
    SalesRecord table's fabricated seed history."""
    from wms.analytics import weekly_forecast as wfc
    rows = wfc.reorder_points(db, limit=100000)["rows"]
    if not rows:
        return pd.DataFrame(columns=["product", "branch", "suggested_order_qty",
                                     "avg_daily_demand", "reorder_point",
                                     "on_hand", "total_need"])
    fc = pd.DataFrame(rows)
    fc["avg_daily_demand"] = (fc["avg_weekly_sales"] / 7).round(3)
    fc["suggested_order_qty"] = (fc["reorder_point"] - fc["on_hand"]).clip(lower=0)
    need = fc.groupby("sku")["suggested_order_qty"].transform("sum")
    out = fc.assign(total_need=need)[
        ["sku", "branch", "avg_daily_demand", "reorder_point", "on_hand",
         "suggested_order_qty", "total_need"]
    ].rename(columns={"sku": "product"})
    return out.sort_values(["total_need", "suggested_order_qty"], ascending=False)
