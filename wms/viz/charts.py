"""Matplotlib + Seaborn chart renderers. Each returns the saved PNG path (or None)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import seaborn as sns                    # noqa: E402

from wms.config import get_settings      # noqa: E402
from wms.analytics import loaders, statistics              # noqa: E402
from wms.analytics import backorders as bo_an             # noqa: E402
from wms.analytics import backorder_flow as bof           # noqa: E402

settings = get_settings()
PALETTE = ["#1F3864", "#2E75B6", "#8FAADC", "#C55A11", "#E8A33D", "#548235", "#7F7F7F"]
_THEMED = False


def _theme():
    global _THEMED
    if _THEMED:
        return
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({"figure.figsize": (10, 5.5), "figure.dpi": 110,
                         "axes.titleweight": "bold", "savefig.bbox": "tight"})
    _THEMED = True


def _save(fig, name: str, out_dir: Optional[Path]) -> Path:
    out = Path(out_dir or settings.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def backorder_fill_rate_by_branch(db, out_dir=None):
    _theme()
    df = bo_an.by_branch(loaders.dn_lines_df(db))
    if df.empty:
        return None
    fig, ax = plt.subplots()
    sns.barplot(data=df, y="branch", x="fill_rate_qty", color=PALETTE[1], ax=ax)
    ax.set(title="Delivery-note fill rate by branch", xlabel="sent / requested",
           ylabel="", xlim=(0, 1))
    return _save(fig, "backorder_fill_rate_by_branch", out_dir)


def backorder_top_items(db, top=15, out_dir=None):
    _theme()
    df = bo_an.by_item(loaders.dn_lines_df(db), top=top)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=df, y="sku", x="backorder_qty", color=PALETTE[3], ax=ax)
    ax.set(title=f"Top {top} items by backorder quantity", xlabel="backorder qty", ylabel="")
    return _save(fig, "backorder_top_items", out_dir)


def backorder_trend(db, out_dir=None):
    _theme()
    df = bo_an.trend(loaders.dn_lines_df(db), freq="W")
    if df.empty:
        return None
    fig, ax1 = plt.subplots()
    ax1.bar(df["period"], df["backorder_qty"], width=4, color=PALETTE[2])
    ax1.set_ylabel("backorder qty")
    ax2 = ax1.twinx()
    ax2.plot(df["period"], df["fill_rate_qty"], color=PALETTE[3], marker="o")
    ax2.set(ylabel="fill rate", ylim=(0, 1))
    ax1.set_title("Backorder quantity & fill rate over time")
    fig.autofmt_xdate()
    return _save(fig, "backorder_trend", out_dir)


def backorder_ageing(db, out_dir=None):
    _theme()
    df = bo_an.ageing(loaders.dn_lines_df(db))
    if df.empty or df["backorder_qty"].sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="bucket", y="backorder_qty", color=PALETTE[0], ax=ax)
    ax.set(title="Backorder ageing (days open)", xlabel="days", ylabel="backorder qty")
    return _save(fig, "backorder_ageing", out_dir)


def branch_item_heatmap(db, out_dir=None, top_items=20):
    _theme()
    dl = loaders.dn_lines_df(db)
    if dl.empty:
        return None
    keep = (dl.groupby("sku")["backorder_qty"].sum()
              .sort_values(ascending=False).head(top_items).index)
    piv = (dl[dl.sku.isin(keep)]
           .pivot_table(index="sku", columns="branch", values="backorder_qty",
                        aggfunc="sum", fill_value=0))
    if piv.empty:
        return None
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * piv.shape[1], 0.45 * piv.shape[0] + 2))
    sns.heatmap(piv, cmap="OrRd", annot=True, fmt=".0f", linewidths=.5, ax=ax)
    ax.set(title="Backorder qty - branch x item", xlabel="", ylabel="")
    return _save(fig, "branch_item_heatmap", out_dir)


def sales_by_branch(db, out_dir=None):
    _theme()
    df = statistics.branch_sales_summary(db)
    if df.empty:
        return None
    fig, ax = plt.subplots()
    sns.barplot(data=df, y="branch", x="sales_qty", color=PALETTE[5], ax=ax)
    ax.set(title="Branch sales (last 90 days)", xlabel="units sold", ylabel="")
    return _save(fig, "sales_by_branch", out_dir)


def abc_pareto(db, out_dir=None):
    _theme()
    df = statistics.abc_classification(db)
    if df.empty:
        return None
    df = df.head(30)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(df["sku"], df["value"], color=PALETTE[1])
    ax1.set_ylabel("demand value")
    ax2 = ax1.twinx()
    ax2.plot(df["sku"], df["cum_share"] * 100, color=PALETTE[3], marker="o")
    ax2.axhline(80, color="grey", ls="--", lw=1)
    ax2.set_ylabel("cumulative %")
    ax1.set_title("ABC Pareto (top 30 by value)")
    fig.autofmt_xdate(rotation=75)
    return _save(fig, "abc_pareto", out_dir)


def demand_forecast_chart(db, branch_id, product_id, out_dir=None):
    _theme()
    dem = loaders.demand_df(db, days=120)
    dem = dem[(dem.branch_id == branch_id) & (dem.product_id == product_id)]
    if dem.empty:
        return None
    s = dem.groupby("date")["qty"].sum().sort_index()
    fig, ax = plt.subplots()
    ax.bar(s.index, s.values, width=1.0, color=PALETTE[2])
    ax.plot(s.index, s.rolling(7, min_periods=1).mean(), color=PALETTE[3])
    ax.set_title("Demand history & 7-day average")
    fig.autofmt_xdate()
    return _save(fig, "demand_forecast", out_dir)


# ======================================================================
# BACKORDER PROCESSING FLOW
# ======================================================================
def bo_stage_funnel(db, out_dir=None):
    _theme()
    df = bof.stage_funnel(loaders.back_orders_df(db))
    df = df[df.stage != "CANCELLED"]
    if df.empty or df.back_orders.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, y="label", x="back_orders", color=PALETTE[1], ax=ax)
    for i, v in enumerate(df.back_orders):
        ax.text(v, i, f" {int(v)}", va="center")
    ax.set(title="Back orders by stage (funnel)", xlabel="open back orders", ylabel="")
    return _save(fig, "bo_stage_funnel", out_dir)


def bo_days_in_stage(db, out_dir=None):
    _theme()
    df = bof.cycle_times(loaders.back_order_events_df(db))
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, y="label", x="mean_days", color=PALETTE[3], ax=ax)
    ax.set(title="Mean days spent in each stage", xlabel="days", ylabel="")
    return _save(fig, "bo_days_in_stage", out_dir)


def bo_fulfilment_by_branch(db, out_dir=None):
    _theme()
    bo = loaders.back_orders_df(db)
    it = loaders.back_order_items_df(db)
    df = bof.by_branch(bo, it)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, y="branch", x="fill_rate_qty", color=PALETTE[5], ax=ax)
    ax.set(title="Back-order fill rate by branch", xlabel="fulfilled / ordered",
           ylabel="", xlim=(0, 1))
    return _save(fig, "bo_fulfilment_by_branch", out_dir)


def bo_aging_heatmap(db, out_dir=None):
    _theme()
    piv = bof.aging_by_stage(loaders.back_orders_df(db))
    if piv is None or piv.empty:
        return None
    piv = piv.set_index("stage")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(piv) + 2))
    sns.heatmap(piv, cmap="OrRd", annot=True, fmt=".0f", linewidths=.5, ax=ax,
                cbar_kws={"label": "open back orders"})
    ax.set(title="Open back orders - stage x age (days)", xlabel="age bucket", ylabel="")
    return _save(fig, "bo_aging_heatmap", out_dir)


def bo_branch_backorder_vs_sales(db, out_dir=None):
    _theme()
    df = bof.branch_backorders_vs_sales(db)
    if df.empty:
        return None
    x = range(len(df))
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    w = 0.4
    ax1.bar([i - w / 2 for i in x], df.sales_qty, width=w, color=PALETTE[5], label="sales qty")
    ax1.bar([i + w / 2 for i in x], df.backordered_qty, width=w, color=PALETTE[3],
            label="backordered qty")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df.branch, rotation=30, ha="right")
    ax1.set_ylabel("quantity")
    ax2 = ax1.twinx()
    ax2.plot(list(x), df.demand_met_pct, color=PALETTE[0], marker="o", label="demand met %")
    ax2.set(ylabel="demand met", ylim=(0, 1))
    ax1.set_title("Branch: sales vs backordered demand")
    ax1.legend(loc="upper left")
    return _save(fig, "bo_branch_backorder_vs_sales", out_dir)


def bo_leadtime_hist(db, out_dir=None):
    _theme()
    bo = loaders.back_orders_df(db)
    lead = bo[bo.status == "CLOSED"]["lead_time_days"].dropna() if not bo.empty else []
    if len(lead) < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(lead, bins=min(20, len(lead)), color=PALETTE[1], ax=ax, kde=True)
    ax.axvline(float(lead.median()), color=PALETTE[3], ls="--",
               label=f"median {lead.median():.0f}d")
    ax.set(title="Back-order lead time (submitted → closed)", xlabel="days")
    ax.legend()
    return _save(fig, "bo_leadtime_hist", out_dir)


def bo_weekly_flow(db, out_dir=None):
    _theme()
    df = bof.trend(loaders.back_orders_df(db), loaders.back_order_events_df(db))
    if df.empty:
        return None
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(df.period, df.raised, width=4, color=PALETTE[3], label="raised")
    ax1.bar(df.period, -df.closed, width=4, color=PALETTE[5], label="closed")
    ax1.plot(df.period, df.outstanding, color=PALETTE[0], marker="o", label="outstanding")
    ax1.axhline(0, color="grey", lw=.8)
    ax1.set_title("Back orders raised / closed / outstanding per week")
    ax1.legend()
    fig.autofmt_xdate()
    return _save(fig, "bo_weekly_flow", out_dir)


def backorder_flow_pack(db, out_dir=None) -> list[Path]:
    fns = [bo_stage_funnel, bo_days_in_stage, bo_fulfilment_by_branch, bo_aging_heatmap,
           bo_branch_backorder_vs_sales, bo_leadtime_hist, bo_weekly_flow]
    return _run_pack(fns, db, out_dir)


def dashboard_pack(db, out_dir=None) -> list[Path]:
    fns = [sales_by_branch, abc_pareto,
           backorder_fill_rate_by_branch, backorder_top_items, backorder_trend,
           backorder_ageing, branch_item_heatmap]
    return _run_pack(fns, db, out_dir)


def _run_pack(fns, db, out_dir) -> list[Path]:
    paths = []
    for fn in fns:
        try:
            p = fn(db, out_dir=out_dir)
            if p:
                paths.append(p)
        except Exception as exc:
            print(f"[viz] {fn.__name__}: {exc}")
    return paths
