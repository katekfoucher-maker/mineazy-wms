"""Borrow a little demand from sibling products.

A "family" is products with the same name but different specs (an electric
cable in 1.5mm / 2.5mm / 4mm, a pulley in different sizes...). At one branch a
product can look like it barely sells only because it was out of stock in some
months, while its siblings sell steadily. This nudges such a product's weekly
demand part of the way up toward what its siblings sell there.

It is deliberately conservative:
  * needs at least two siblings at the same branch that sell reliably and
    recently, and the product itself must have sold recently (a dead line is
    not brought back to life);
  * the pull fades as the product's own sales history gets denser - a product
    that sells most months, even at a low level, is trusted as it is - and is
    halved when its last sale was a while ago;
  * it only ever lifts, never lowers;
  * the result never exceeds the most the product has ever sold in a period at
    that branch.
"""
from __future__ import annotations

import re

import numpy as np

# tokens that describe the spec (size, unit, colour), not what the product is
_SPEC_WORDS = {
    "X", "MM", "CM", "M", "/M", "MTR", "MTRS", "MTS", "INCH", "IN", "KG", "KW",
    "LT", "LTR", "LTRS", "L", "T", "PCS", "CORE", "C", "CL", "SZ", "SIZE",
    "BLACK", "WHITE", "RED", "BLUE", "GREEN", "YELLOW", "GREY", "GRAY",
    "ORANGE", "BROWN", "SMALL", "MEDIUM", "LARGE",
}

MAX_BORROW = 0.6          # at most this share of the gap to the siblings is closed
CAP_MULT = 1.0            # never above what the product has ever sold in one period there
MIN_SIBLINGS = 2          # reliable siblings needed at the same branch
SIB_MIN_DENSITY = 0.6     # a sibling must have sold in at least this share of its periods
MIN_OWN_PERIODS = 3       # need this many periods since first sale to judge density
RECENT_PERIODS = 6        # "sold recently" = a sale in the last this-many periods (the
                          # forecast's own "dead" rule is 6 periods of nothing)
FRESH_PERIODS = 3         # ...and a sale inside this many is "fresh" (full pull);
STALE_FACTOR = 0.5        # older than that, the pull is only this share as strong


def family_stem(name) -> str | None:
    """Product name with the specs stripped, or None when too little of a name
    is left to mean a family (a one-word stem like "BEARING" is too generic)."""
    s = re.sub(r"\([^)]*\)", " ", str(name or "").upper())
    keep = []
    for tok in re.split(r"[\s,;:]+", s):
        tok = tok.strip(" .-_\"'")
        if not tok or tok in _SPEC_WORDS or any(ch.isdigit() for ch in tok):
            continue
        keep.append(tok)
    return " ".join(keep) if len(keep) >= 2 else None


def borrow_from_siblings(branch_of, items, weekly_demand, MAT_raw,
                         per_period_weeks: float = 1.0, oos=None):
    """-> ``(new_weekly_demand, borrowed)``, both int arrays like the input.

    ``weekly_demand`` is the model's weekly rate per series; ``MAT_raw`` the
    (series x period) matrix of what actually sold; ``per_period_weeks`` how
    many weeks one period spans (4 for monthly data, 1 for weekly); ``oos`` an
    optional bool matrix of periods known to be out of stock (they don't count
    against a product's sales density)."""
    wd = np.asarray(weekly_demand, float)
    S = len(wd)
    new = np.round(wd).astype(int)
    borrowed = np.zeros(S, int)
    MAT = np.asarray(MAT_raw, float)
    if S == 0 or MAT.ndim != 2 or MAT.shape[0] != S or MAT.shape[1] == 0:
        return new, borrowed
    W = MAT.shape[1]
    sold = MAT > 0
    nnz = sold.sum(1)
    first = np.where(nnz > 0, sold.argmax(1), W)
    live = sold[:, -min(RECENT_PERIODS, W):].any(1)
    fresh = sold[:, -min(FRESH_PERIODS, W):].any(1)
    peak_wk = MAT.max(1) / max(per_period_weeks, 1e-9)

    density = np.zeros(S)
    span = W - first
    for i in range(S):
        if nnz[i] == 0 or span[i] < MIN_OWN_PERIODS:
            continue
        seen = span[i]
        if oos is not None and getattr(oos, "shape", None) == MAT.shape:
            seen = int((~oos[i, first[i]:]).sum())
        density[i] = min(1.0, nnz[i] / max(seen, nnz[i], 1))

    groups: dict = {}
    for i in range(S):
        stem = family_stem(items[i])
        if stem:
            groups.setdefault((str(branch_of[i]), stem), []).append(i)

    for members in groups.values():
        if len(members) < MIN_SIBLINGS + 1:
            continue
        for i in members:
            if not live[i] or span[i] < MIN_OWN_PERIODS or density[i] >= 1.0:
                continue
            sibs = [j for j in members
                    if j != i and live[j] and wd[j] > 0
                    and density[j] >= SIB_MIN_DENSITY]
            if len(sibs) < MIN_SIBLINGS:
                continue
            sib_level = float(np.median(wd[sibs]))
            sib_density = float(np.median(density[sibs]))
            if sib_level <= wd[i] or sib_density <= 0:
                continue
            gap = min(1.0, max(0.0, 1.0 - density[i] / sib_density))
            if gap <= 0:
                continue
            pull = MAX_BORROW * gap * (1.0 if fresh[i] else STALE_FACTOR)
            target = wd[i] + pull * (sib_level - wd[i])
            target = min(target, max(wd[i], CAP_MULT * peak_wk[i]))
            out = int(np.ceil(target))
            if out > new[i]:
                borrowed[i] = out - new[i]
                new[i] = out
    return new, borrowed
