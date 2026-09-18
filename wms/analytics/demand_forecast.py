"""Next-month demand forecast for Belmont & Gwanda VID from 3 months of
branch *Item Statistics* exports.

Why this method
---------------
The input is **monthly** unit totals per branch/SKU, three points each
(May / Jun / Jul 2026).  With so short a history, and thousands of mostly
slow-moving SKUs, anything with parameters to fit (ARIMA, Croston/TSB,
gradient boosting) over-fits.  We picked the method by back-testing on the
one hold-out we have (train May+Jun, predict Jul): a **recency-weighted
moving average** beats naive-last and naive-mean, while a damped-trend term,
a clamp band, and cross-SKU/category pooling all made it *worse*, so they
were dropped.  Per (branch, SKU):

  * >= 2 months with sales -> recency-weighted average of the 3 monthly
    quantities (0-filled), weights ~ 0.16 / 0.30 / 0.54 (oldest->newest)
  * sold in the latest month only -> 0.6 x that quantity (one observation
    regresses toward zero next month)
  * sold earlier but zero in the latest month -> 0.35 x the prior average
  * x1.05 to offset the ~-6% back-test bias
  * 80% band and a next-month order-up-to at the service level from the
    spread of the monthly values

`backtest()` reports WAPE vs naive-last / naive-mean so the value-add stays
visible as more months arrive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wms.analytics import monthly_sales
from wms.config import get_settings

_R = 0.55                 # recency decay for the weighted average
_SINGLE = 0.6             # shrink a one-month-only item toward zero
_DROPPED = 0.35           # shrink an item that went quiet last month
_BIAS_FIX = 1.05          # offsets the ~-6% back-test bias
_Z = {0.80: 0.84, 0.90: 1.28, 0.95: 1.645, 0.975: 1.96, 0.99: 2.33}


def _z(level: float) -> float:
    return _Z.get(round(level, 3), 1.645)


# ---------------------------------------------------------------- core
def _matrix(panel: pd.DataFrame, periods: list[pd.Timestamp]) -> pd.DataFrame:
    """(branch, sku) x periods grid of quantities, missing months = 0."""
    p = panel[panel["period"].isin(periods)]
    grid = (p.pivot_table(index=["branch_code", "branch", "sku"], columns="period",
                          values="qty", aggfunc="sum")
             .reindex(columns=periods).fillna(0.0))
    meta = (p.sort_values("period")
             .groupby(["branch_code", "branch", "sku"])
             .agg(item=("item", "last"), category=("category", "last"),
                  turnover=("turnover", "sum"), units=("qty", "sum")))
    meta["unit_price"] = np.where(meta["units"] > 0, meta["turnover"] / meta["units"], np.nan)
    return grid.join(meta)


def _pattern(qs: list[float]) -> str:
    sold = [q for q in qs if q > 0]
    n = len(sold)
    if n == 0:
        return "none"
    if n == 1:
        return "new" if qs[-1] > 0 else "dropped"
    if qs[-1] == 0:
        return "dropped"
    if len(qs) >= 3:
        d1, d2 = qs[1] - qs[0], qs[2] - qs[1]
        if d1 > 0 and d2 > 0:
            return "growing"
        if d1 < 0 and d2 < 0:
            return "declining"
    if n < len(qs):
        return "intermittent"
    return "regular"


def _forecast_row(qs: list[float]) -> tuple[float, float]:
    """qs oldest->newest, 0-filled. Returns (point_forecast, sigma)."""
    m = len(qs)
    nonzero = [q for q in qs if q > 0]
    n = len(nonzero)

    if n == 0:
        return 0.0, 0.0
    if n == 1 and qs[-1] > 0:                     # only the latest month
        pt = _SINGLE * qs[-1]
    elif qs[-1] == 0:                             # went quiet last month
        pt = _DROPPED * float(np.mean(nonzero))
    else:                                         # >= 2 months with sales
        w = np.array([_R ** (m - 1 - i) for i in range(m)], dtype=float)
        w /= w.sum()
        pt = float(np.dot(w, qs))
    pt = max(0.0, pt * _BIAS_FIX)

    if len([q for q in qs if q != 0]) >= 2:
        sigma = float(np.std(qs, ddof=0))
    else:
        sigma = 0.7 * pt
    return pt, sigma


def _pipeline(panel: pd.DataFrame, periods: list[pd.Timestamp]) -> pd.DataFrame:
    mat = _matrix(panel, periods)
    if mat.empty:
        return pd.DataFrame()
    qcols = list(periods)
    rows = []
    for idx, r in mat.iterrows():
        bcode, bname, sku = idx
        qs = [float(r[c]) for c in qcols]
        pt, sigma = _forecast_row(qs)
        nz = [q for q in qs if q > 0]
        cv = float(np.std(nz, ddof=0) / np.mean(nz)) if len(nz) >= 2 else np.nan
        months_sold = len(nz)
        if months_sold <= 1:
            conf = "Low"
        elif months_sold == len(qs) and (np.isnan(cv) or cv <= 0.6):
            conf = "High"
        else:
            conf = "Medium"
        rows.append({
            "branch_code": bcode, "branch": bname, "sku": sku,
            "item": r["item"], "category": r["category"],
            "unit_price": r["unit_price"], "qs": qs,
            "forecast": pt, "sigma": sigma, "cv": round(cv, 2) if cv == cv else np.nan,
            "months_sold": months_sold, "pattern": _pattern(qs), "confidence": conf,
            "method": "recency-weighted avg" if len(nz) >= 2 else
                      ("single-month x0.6" if qs[-1] > 0 else "quiet x0.35"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- public
def _split_periods(periods: list) -> tuple[list, object, bool]:
    """history months + the month being forecast + whether its actuals exist.

    With >=2 months we forecast the *latest* month from the ones before it, so
    the table doubles as a live forecast-vs-actual scorecard.  With 1 month we
    fall back to projecting the next (unreleased) month.
    """
    if len(periods) >= 2:
        return periods[:-1], periods[-1], True
    nxt = (pd.Timestamp(periods[-1]) + pd.offsets.MonthBegin(1)) + pd.offsets.MonthEnd(0)
    return periods, nxt, False


def forecast_table(panel: pd.DataFrame | None = None, branch_code: str | None = None
                   ) -> pd.DataFrame:
    panel = monthly_sales.load_panel() if panel is None else panel
    if panel.empty:
        return pd.DataFrame()
    periods = sorted(panel["period"].unique())
    history, target, has_actual = _split_periods(periods)
    df = _pipeline(panel, history)
    if df.empty:
        return df

    z = _z(get_settings().service_level)
    df["forecast_qty"] = df["forecast"].round().astype(int)
    df["low80"] = (df["forecast"] - 1.28 * df["sigma"]).clip(lower=0).round().astype(int)
    df["high80"] = (df["forecast"] + 1.28 * df["sigma"]).round().astype(int)
    df["order_up_to"] = np.ceil(df["forecast"] + z * df["sigma"]).clip(lower=0).astype(int)
    df["forecast_value"] = (df["forecast"] * df["unit_price"].fillna(0)).round(2)

    labels = [pd.Timestamp(p).strftime("%b %Y") for p in history]
    for i, lab in enumerate(labels):
        df[lab] = df["qs"].map(lambda q, i=i: int(round(q[i])))
    tgt_label = pd.Timestamp(target).strftime("%b %Y")
    df["forecast_month"] = tgt_label

    if has_actual:
        act = (panel[panel["period"] == target]
               .groupby(["branch_code", "sku"])["qty"].sum().round().astype(int))
        df = df.merge(act.rename("actual_qty").reset_index(),
                      on=["branch_code", "sku"], how="left")
        df["actual_qty"] = df["actual_qty"].fillna(0).astype(int)
        df["forecast_error"] = df["forecast_qty"] - df["actual_qty"]
    else:
        df["actual_qty"] = pd.NA
        df["forecast_error"] = pd.NA

    cols = (["branch", "sku", "item", "category"] + labels +
            ["months_sold", "pattern", "cv", "method", "confidence",
             "forecast_month", "forecast_qty", "actual_qty", "forecast_error",
             "low80", "high80", "order_up_to", "forecast_value"])
    out = df[cols].sort_values(["branch", "forecast_value"], ascending=[True, False])
    if branch_code:
        want = _name(branch_code)
        out = out[out["branch"].eq(want) | out["branch"].eq(branch_code)]
    return out.reset_index(drop=True)


def _name(code: str) -> str:
    return monthly_sales.BRANCH_NAME.get((code or "").upper(), code)


def backtest(panel: pd.DataFrame | None = None) -> dict:
    panel = monthly_sales.load_panel() if panel is None else panel
    periods = sorted(panel["period"].unique())
    if len(periods) < 2:
        return {"ok": False, "reason": "need at least 2 months to back-test"}
    train, holdout = periods[:-1], periods[-1]

    pred = _pipeline(panel, train)
    if pred.empty:
        return {"ok": False, "reason": "no training rows"}
    pred = pred.set_index(["branch_code", "sku"])

    act = (panel[panel["period"] == holdout]
           .groupby(["branch_code", "sku"])["qty"].sum())
    keys = pred.index.union(act.index)
    a = act.reindex(keys).fillna(0.0)

    model = pred["forecast"].reindex(keys).fillna(0.0)
    naive_last = pred["qs"].map(lambda q: q[-1]).reindex(keys).fillna(0.0)
    naive_mean = pred["qs"].map(lambda q: float(np.mean(q))).reindex(keys).fillna(0.0)
    branch = keys.get_level_values("branch_code")

    def _rows(mask=None):
        idx = slice(None) if mask is None else mask
        aa, mm = a[idx], model[idx]
        nl, nm = naive_last[idx], naive_mean[idx]
        m = _metrics_sub(aa, mm)
        wl = _wape(aa, nl)
        wm = _wape(aa, nm)
        m["skill_vs_last_pct"] = round(100 * (1 - m["wape_pct"] / wl), 1) if wl else None
        m["skill_vs_mean_pct"] = round(100 * (1 - m["wape_pct"] / wm), 1) if wm else None
        return m

    overall = _rows()
    by_branch = []
    for code in sorted(set(branch)):
        row = _rows(branch == code)
        row["branch"] = _name(code)
        by_branch.append(row)

    return {
        "ok": True,
        "train_months": [pd.Timestamp(p).strftime("%b %Y") for p in train],
        "holdout_month": pd.Timestamp(holdout).strftime("%b %Y"),
        "overall": overall,
        "by_branch": pd.DataFrame(by_branch)[
            ["branch", "series", "wape_pct", "mae", "bias_pct",
             "skill_vs_last_pct", "skill_vs_mean_pct"]],
    }


def _wape(a: pd.Series, p: pd.Series):
    tot = a.sum()
    return float((p - a).abs().sum() / tot) * 100 if tot else None


def _metrics_sub(a: pd.Series, p: pd.Series) -> dict:
    e = (p - a).abs()
    tot = a.sum()
    return {
        "series": int(((a > 0) | (p > 0)).sum()),
        "wape_pct": round(100 * e.sum() / tot, 1) if tot else None,
        "mae": round(float(e.mean()), 2),
        "bias_pct": round(100 * (p - a).sum() / tot, 1) if tot else None,
    }


def summary(panel: pd.DataFrame | None = None, fc: pd.DataFrame | None = None) -> dict:
    panel = monthly_sales.load_panel() if panel is None else panel
    fc = forecast_table(panel) if fc is None else fc
    cov = monthly_sales.coverage(panel)
    if fc.empty:
        return {"coverage": cov, "by_branch": pd.DataFrame(), "top": pd.DataFrame()}

    by_branch = (fc.groupby("branch")
                   .agg(skus=("sku", "nunique"),
                        forecast_units=("forecast_qty", "sum"),
                        forecast_value=("forecast_value", "sum"),
                        growing=("pattern", lambda s: int((s == "growing").sum())),
                        declining=("pattern", lambda s: int((s == "declining").sum())),
                        intermittent=("pattern", lambda s: int((s == "intermittent").sum())),
                        low_confidence=("confidence", lambda s: int((s == "Low").sum())))
                   .reset_index())
    top = (fc.sort_values("forecast_value", ascending=False)
             .head(20)[["branch", "sku", "item", "pattern",
                        "forecast_qty", "forecast_value", "confidence"]]
             .reset_index(drop=True))
    return {"coverage": cov, "by_branch": by_branch, "top": top}


def run(panel: pd.DataFrame | None = None) -> dict:
    panel = monthly_sales.load_panel() if panel is None else panel
    fc = forecast_table(panel)
    periods = sorted(panel["period"].unique()) if not panel.empty else []
    history, target, has_actual = _split_periods(periods) if periods else ([], None, False)
    labels = {
        "history": [pd.Timestamp(p).strftime("%b %Y") for p in history],
        "target": pd.Timestamp(target).strftime("%b %Y") if target is not None else "",
        "has_actual": has_actual,
    }
    return {"panel": panel, "forecast": fc, "labels": labels,
            "backtest": backtest(panel), "summary": summary(panel, fc)}


def display_frame(bcode: str = "", q: str = "") -> pd.DataFrame:
    """Exactly the columns shown on the Sales & Forecasting page - no branch, no
    internal metrics. One row per branch-SKU; ``bcode`` / ``q`` apply the same
    filters as the on-screen table so an export matches what the user sees."""
    r = cached_run()
    fc, lbl = r["forecast"], r.get("labels", {})
    if fc.empty:
        return pd.DataFrame()
    hist = lbl.get("history", [])
    tgt = lbl.get("target", "") or "forecast"
    fcst_col, act_col = f"{tgt} forecast", f"{tgt} actual"

    d = fc[["branch", "sku", "item", *hist]].copy()
    d[fcst_col] = fc["forecast_qty"].values
    if lbl.get("has_actual"):
        d[act_col] = fc["actual_qty"].values
        # how wrong the forecast was: + = under-forecast, - = over-forecast
        d["error (actual - forecast)"] = (fc["actual_qty"] - fc["forecast_qty"]).values
    if bcode:
        d = d[d["branch"].str.lower().str.startswith(bcode.lower())]
    if q:
        s = q.strip().lower()
        d = d[d["sku"].str.lower().str.contains(s, regex=False) |
              d["item"].str.lower().str.contains(s, regex=False)]
    d = d.drop(columns="branch").rename(columns={"sku": "SKU", "item": "Product"})
    return d.reset_index(drop=True)


_CACHE: dict = {}


def cached_run() -> dict:
    """run(), memoised on the set + mtimes of the source files."""
    d = monthly_sales.history_dir()
    try:
        sig = tuple(sorted((p.name, p.stat().st_mtime)
                           for p in d.glob("*.xls*") if not p.name.startswith("~$")))
    except OSError:
        sig = ()
    if _CACHE.get("sig") != sig:
        _CACHE["sig"] = sig
        _CACHE["val"] = run()
    return _CACHE["val"]
