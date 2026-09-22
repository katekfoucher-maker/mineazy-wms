"""Per-branch stock-on-hand balance (``stock_on_hand`` table).

  * a **dispatch confirmation** adds the dispatched quantity onto the branch's
    balance (:func:`add_stock`),
  * an **inventory upload** replaces a branch's balance with the snapshot in the
    file (:func:`set_branch_stock`).

Rows are keyed by ``(branch_id, sku)`` where ``sku`` is free text so it lines up
with the forecast / uploaded-spreadsheet SKUs without needing a catalogue entry.
The Allocation plan reads :func:`levels_df`.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.config import get_settings
from wms.models import Branch, StockOnHand


def _row(db: Session, branch_id: int, sku: str) -> StockOnHand:
    row = (db.query(StockOnHand)
           .filter(StockOnHand.branch_id == branch_id, StockOnHand.sku == sku)
           .first())
    if row is None:
        row = StockOnHand(branch_id=branch_id, sku=sku, qty_on_hand=0)
        db.add(row)
        db.flush()
    return row


def add_stock(db: Session, *, branch_id: int, items: list[dict],
              user_id: Optional[int] = None, commit: bool = True) -> int:
    """Increment the branch balance. ``items``: [{sku, qty}].
    Returns the number of units added; a balance never drops below zero."""
    added = 0
    for raw in items:
        qty = int(raw.get("qty") or raw.get("qty_dispatched") or 0)
        sku = str(raw.get("sku") or "").strip()
        if not sku or qty == 0:
            continue
        row = _row(db, branch_id, sku)
        row.qty_on_hand = max(0, int(row.qty_on_hand or 0) + qty)
        added += max(0, qty)
    if added:
        write_audit(db, entity_type="StockOnHand", entity_id=branch_id,
                    action="ADD", detail={"branch_id": branch_id, "units": added},
                    user_id=user_id)
    if commit:
        db.commit()
    return added


def set_branch_stock(db: Session, *, branch_id: int, items: list[dict],
                     user_id: Optional[int] = None, commit: bool = True) -> int:
    """Replace a branch's whole balance with ``items`` ([{sku, qty|on_hand}])."""
    db.query(StockOnHand).filter(StockOnHand.branch_id == branch_id).delete()
    db.flush()
    n = 0
    for raw in items:
        sku = str(raw.get("sku") or "").strip()
        if not sku:
            continue
        qty = max(0, int(raw.get("qty") or raw.get("on_hand") or 0))
        db.add(StockOnHand(branch_id=branch_id, sku=sku, qty_on_hand=qty))
        n += 1
    write_audit(db, entity_type="StockOnHand", entity_id=branch_id, action="SET",
                detail={"branch_id": branch_id, "lines": n}, user_id=user_id)
    if commit:
        db.commit()
    return n


def levels_df(db: Session) -> pd.DataFrame:
    """One row per (branch_code, sku): columns branch_code, sku, on_hand."""
    import time
    query = (db.query(Branch.code, StockOnHand.sku, StockOnHand.qty_on_hand)
             .join(Branch, Branch.id == StockOnHand.branch_id))
    rows = None
    for attempt in range(1, 4):
        try:
            rows = query.all()
            break
        except Exception:                                 # noqa: BLE001
            db.rollback()                # a Session is unusable after an error until rolled back
            if attempt == 3:
                raise
            time.sleep(1.5 * attempt)
    if not rows:
        return pd.DataFrame(columns=["branch_code", "sku", "on_hand"])
    df = pd.DataFrame(rows, columns=["branch_code", "sku", "on_hand"])
    df["on_hand"] = pd.to_numeric(df["on_hand"], errors="coerce").fillna(0).clip(lower=0)
    return df


# Branches whose reported stock-on-hand isn't reliable enough to use for
# allocation math (flagged by the business, not derived from anything in the
# data itself). Read-only views (the Inventory page, coverage() below) still
# show the real figure via plain levels_df() - only allocation call sites use
# levels_df_for_allocation().
UNRELIABLE_FOR_ALLOCATION = {"BM"}         # Belmont Shop


def levels_df_for_allocation(db: Session) -> pd.DataFrame:
    """Like :func:`levels_df`, but zeroes on-hand wherever it isn't reliable
    enough to use for allocation math. Allocation treats a zeroed branch as
    if nothing were on its shelf - the conservative reading ("send the full
    target") rather than letting a number nobody trusts suppress what gets
    sent there. The real stock_on_hand rows are untouched - read-only views
    (the Inventory page, coverage() below) still show the real figure via
    plain levels_df().

    ``allocation_use_inventory=False`` (see wms.config) zeroes EVERY branch -
    a temporary network-wide switch for while stock-on-hand data generally
    isn't trusted, so allocation runs on sales/demand alone. It's checked
    ahead of the narrower ``UNRELIABLE_FOR_ALLOCATION`` list (Belmont) so
    that list stays in place, ready to matter again the moment inventory
    data is switched back on."""
    df = levels_df(db)
    if df.empty:
        return df
    if not getattr(get_settings(), "allocation_use_inventory", True):
        df = df.copy()
        df["on_hand"] = 0
        return df
    if not UNRELIABLE_FOR_ALLOCATION:
        return df
    df = df.copy()
    df.loc[df["branch_code"].str.upper().isin(UNRELIABLE_FOR_ALLOCATION), "on_hand"] = 0
    return df


def coverage(db: Session) -> dict:
    df = levels_df(db)
    if df.empty:
        return {"branches": 0, "rows": 0, "branch_codes": ""}
    return {
        "branches": int(df["branch_code"].nunique()),
        "rows": int(len(df)),
        "branch_codes": ", ".join(sorted(df["branch_code"].unique())),
    }
