"""Export a tidy training dataset for the "anticipate branch orders" model.

    python -m wms.scripts.export_training_data                 # -> output/training_data_*.csv
    python -m wms.scripts.export_training_data --format parquet --weeks 78

One row per (branch, product, ISO-week). Target = next week's demand.
Features are computed strictly from the past (no leakage): demand lags,
rolling stats, calendar, ABC/price band, intermittency, recent-backorder signals.
See the README "Steps to train the model" section.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from wms.config import get_settings
from wms.db import SessionLocal, init_db
from wms.analytics import loaders

settings = get_settings()


def build(db, weeks: int = 78) -> pd.DataFrame:
    cutoff = date.today() - timedelta(weeks=weeks)

    # ---- weekly demand per (branch, product) ----
    dem = loaders.demand_df(db, days=weeks * 7 + 14)
    if dem.empty:
        return pd.DataFrame()
    dem = dem[dem.date >= pd.Timestamp(cutoff)]
    dem["week"] = dem["date"].dt.to_period("W").dt.start_time
    wk = (dem.groupby(["branch_id", "branch", "product_id", "sku", "week"])["qty"]
             .sum().reset_index())

    # dense (branch, product) x week grid so gaps are real zeros
    weeks_idx = pd.date_range(wk.week.min(), wk.week.max(), freq="W-MON")
    keys = wk[["branch_id", "branch", "product_id", "sku"]].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"week": weeks_idx}), how="cross")
    df = grid.merge(wk, on=["branch_id", "branch", "product_id", "sku", "week"], how="left")
    df["qty"] = df["qty"].fillna(0.0)
    df = df.sort_values(["branch_id", "product_id", "week"])

    def col(fn):
        return df.groupby(["branch_id", "product_id"])["qty"].transform(fn)

    # ---- target: next week's demand ----
    df["target_next_week_qty"] = col(lambda s: s.shift(-1))

    # ---- lag + rolling features (past only: everything shifted >= 1) ----
    for lag in (1, 2, 3, 4, 8):
        df[f"lag_{lag}"] = col(lambda s, l=lag: s.shift(l))
    for win in (4, 8, 13):
        df[f"roll_mean_{win}"] = col(lambda s, w=win: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"roll_std_{win}"] = col(lambda s, w=win: s.shift(1).rolling(w, min_periods=2).std())
    df["trend_4"] = df["roll_mean_4"] - col(
        lambda s: s.shift(5).rolling(4, min_periods=1).mean())
    df["weeks_since_last_sale"] = col(
        lambda s: s.eq(0).groupby((s != 0).cumsum()).cumcount())
    df["zero_share_13"] = col(
        lambda s: s.shift(1).rolling(13, min_periods=4).apply(lambda w: float((w == 0).mean())))

    # ---- calendar ----
    iso = df["week"].dt.isocalendar()
    df["week_of_year"] = iso.week.astype(int)
    df["month"] = df["week"].dt.month
    df["sin_woy"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_woy"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["is_month_end_week"] = (df["week"].dt.day >= 24).astype(int)

    # ---- product attributes ----
    prods = {p.id: p for p in _all_products(db)}
    df["category"] = df["product_id"].map(lambda i: getattr(prods.get(i), "category", None))
    df["unit_price"] = df["product_id"].map(
        lambda i: float(getattr(prods.get(i), "unit_price", 0) or 0))
    df["price_band"] = pd.qcut(df["unit_price"].rank(method="first"), 5,
                               labels=[1, 2, 3, 4, 5]).astype(int)

    # ---- recent backorder signal per (branch, product) ----
    items = loaders.back_order_items_df(db)
    bo = loaders.back_orders_df(db)
    if not items.empty and not bo.empty:
        m = items.merge(bo[["bo_no", "submitted_at"]], on="bo_no", how="left")
        m["week"] = pd.to_datetime(m["submitted_at"]).dt.to_period("W").dt.start_time
        bo_wk = (m.groupby(["branch_id", "sku", "week"])
                   .agg(bo_qty=("qty_ordered", "sum"), bo_count=("bo_no", "nunique"))
                   .reset_index())
        df = df.merge(bo_wk, on=["branch_id", "sku", "week"], how="left")
    if "bo_qty" not in df:
        df["bo_qty"] = 0.0
    if "bo_count" not in df:
        df["bo_count"] = 0.0
    df[["bo_qty", "bo_count"]] = df[["bo_qty", "bo_count"]].fillna(0.0)
    df["bo_qty_roll_4"] = df.groupby(["branch_id", "product_id"])["bo_qty"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).sum())

    df = df.rename(columns={"qty": "demand_qty"})
    df = df.dropna(subset=["target_next_week_qty"])       # last week has no target
    return df.reset_index(drop=True)


def _all_products(db):
    from wms.models import Product
    return db.query(Product).all()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export model training data")
    ap.add_argument("--weeks", type=int, default=78)
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    init_db()
    db = SessionLocal()
    try:
        df = build(db, weeks=a.weeks)
    finally:
        db.close()
    if df.empty:
        raise SystemExit("No demand history to export - seed or import sales first.")

    stamp = date.today().strftime("%Y%m%d")
    path = a.out or str(settings.out / f"training_data_{stamp}.{a.format}")
    if a.format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"rows        : {len(df):,}")
    print(f"pairs       : {df.groupby(['branch_id','product_id']).ngroups:,}")
    print(f"weeks       : {df['week'].nunique()}  ({df['week'].min().date()} .. {df['week'].max().date()})")
    print(f"columns     : {len(df.columns)}")
    print(f"target mean : {df['target_next_week_qty'].mean():.2f}")
    print(f"written     : {path}")


if __name__ == "__main__":
    main()
