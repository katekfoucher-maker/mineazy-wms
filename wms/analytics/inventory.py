"""Current stock-on-hand per branch, uploaded as one spreadsheet per branch.

Files live in ``data/inventory/`` named ``<BRANCH_CODE>.xlsx`` (BM, GWA, MP ...).
Each is a snapshot - a new upload replaces the branch's previous file. The sheet
just needs a product-code column and a quantity column; extra columns are ignored.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

from wms.config import get_settings

_SKU_KEYS = ("item no", "item_no", "itemno", "sku", "code", "product", "part no", "part_no")
_QTY_KEYS = ("qty", "quantity", "on hand", "on_hand", "onhand", "in stock",
             "stock", "balance", "closing", "closing balance")


def inventory_dir() -> Path:
    p = Path(getattr(get_settings(), "inventory_dir", "./data/inventory"))
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p


def _pick(cols, keys):
    low = {str(c).strip().lower(): c for c in cols}
    for k in keys:
        if k in low:
            return low[k]
    for k in keys:                       # loose contains match
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None


def load_inventory(directory=None) -> pd.DataFrame:
    """One row per (branch_code, sku): columns branch_code, sku, on_hand."""
    d = Path(directory) if directory else inventory_dir()
    files = sorted(glob.glob(str(d / "*.xls*"))) + sorted(glob.glob(str(d / "*.csv")))
    files = [f for f in files
             if not os.path.basename(f).startswith(("~$", "_"))]   # _warehouse.csv etc.
    frames = []
    for f in files:
        code = os.path.splitext(os.path.basename(f))[0].strip().upper()
        try:
            raw = (pd.read_csv(f, dtype=str) if f.lower().endswith(".csv")
                   else pd.read_excel(f, dtype=str))
        except Exception:
            continue
        if raw.empty:
            continue
        sku_c = _pick(raw.columns, _SKU_KEYS)
        qty_c = _pick(raw.columns, _QTY_KEYS)
        if sku_c is None or qty_c is None:
            continue
        out = pd.DataFrame({
            "branch_code": code,
            "sku": raw[sku_c].astype(str).str.strip(),
            "on_hand": pd.to_numeric(raw[qty_c], errors="coerce"),
        })
        out = out[(out["sku"].str.len() > 0) & out["sku"].str.lower().ne("nan")]
        out["on_hand"] = out["on_hand"].fillna(0).clip(lower=0)
        frames.append(out.groupby(["branch_code", "sku"], as_index=False)["on_hand"].sum())

    if not frames:
        return pd.DataFrame(columns=["branch_code", "sku", "on_hand"])
    return (pd.concat(frames, ignore_index=True)
              .groupby(["branch_code", "sku"], as_index=False)["on_hand"].sum())


def coverage(inv: pd.DataFrame | None = None) -> dict:
    inv = load_inventory() if inv is None else inv
    if inv.empty:
        return {"branches": 0, "rows": 0, "branch_codes": ""}
    return {
        "branches": int(inv["branch_code"].nunique()),
        "rows": int(len(inv)),
        "branch_codes": ", ".join(sorted(inv["branch_code"].unique())),
    }


# ----------------------------------------------------------------------
# central warehouse / DC stock - one snapshot, used to split scarce stock
# across branches when weekly order requests come in
# ----------------------------------------------------------------------
def warehouse_file() -> Path:
    """The normalised warehouse-stock snapshot: a 2-column CSV (sku, on_hand)."""
    return inventory_dir() / "_warehouse.csv"


def save_warehouse_inventory(lines) -> int:
    """Replace the warehouse snapshot from ``[{"sku", "qty"}, ...]`` (or objects
    with those keys). Returns the number of product lines stored."""
    rows = {}
    for ln in lines or []:
        get = ln.get if isinstance(ln, dict) else lambda k, _o=ln: getattr(_o, k, None)
        sku = str(get("sku") or "").strip()
        if not sku or sku.lower() == "nan":
            continue
        try:
            qty = int(round(float(get("qty") if get("qty") is not None
                                  else get("on_hand") or 0)))
        except (TypeError, ValueError):
            qty = 0
        rows[sku.upper()] = rows.get(sku.upper(), 0) + max(0, qty)
    d = inventory_dir()
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(sorted(rows.items()), columns=["sku", "on_hand"])
    df.to_csv(warehouse_file(), index=False)
    return len(df)


def load_warehouse_inventory() -> pd.DataFrame:
    """One row per sku: columns ``sku`` (upper-case), ``on_hand``. Empty frame
    when no warehouse snapshot has been uploaded."""
    f = warehouse_file()
    if not f.exists():
        return pd.DataFrame(columns=["sku", "on_hand"])
    try:
        df = pd.read_csv(f, dtype={"sku": str})
    except Exception:                                    # noqa: BLE001
        return pd.DataFrame(columns=["sku", "on_hand"])
    df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    df["on_hand"] = pd.to_numeric(df["on_hand"], errors="coerce").fillna(0).clip(lower=0)
    return df[df["sku"].str.len() > 0].reset_index(drop=True)


def warehouse_coverage() -> dict:
    wh = load_warehouse_inventory()
    if wh.empty:
        return {"rows": 0, "units": 0}
    return {"rows": int(len(wh)), "units": int(wh["on_hand"].sum())}
