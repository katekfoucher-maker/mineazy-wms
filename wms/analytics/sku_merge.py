"""Join a product's sales history when it was re-coded.

Sometimes the same product is issued a new SKU code: the old code stops selling
in the very month the new one starts, and the two codes carry the identical
name. Each code then looks like a fragment - the new one is a beginner with 1-5
months of history, the old one looks dead - so the forecast for the live code is
far too low. This finds those hand-overs and folds the old code's history into
the code that replaced it, so the forecast sees one continuous series.

A pair (old -> new) at a branch is a re-code when ALL hold:
  * identical product name (ignoring case, spacing and punctuation - NOT
    sizes/specs), long enough to mean something. Looser matching (similar
    names, matching timing alone) was tried on the full history and produces
    mostly false joins: different sizes or variants of one family that happen
    to start as another stops, so only this exact rule is used;
  * the old code stops selling no later than the month the new code starts
    (they never both sell for more than a single overlapping month);
  * the old code was established (sold in at least ``MIN_OLD_MONTHS`` months);
  * unit prices are alike (within a factor of ``PRICE_RATIO``) when both known.

The old code's history is added at the SHARE of its volume the new code has
actually picked up (the new code's own monthly level since it began, over the
old code's level in its last 6 selling months, held between ``SCALE_LO`` and 1):
a new code that sells as much as the old one gets all of it, one that has taken
over only part of the volume gets part. Back-tested on the last 3 months this
halves the over-forecast of a straight sum.

A pair seen at ``GLOBAL_MIN_BRANCHES`` or more branches is a company-wide
re-code: it is then also applied at the other branches that carry both codes
(where the old code may have too little history to qualify on its own).
"""
from __future__ import annotations

import collections
import re

import numpy as np

MIN_NAME_LEN = 5
MIN_OLD_MONTHS = 4          # old code must have sold in at least this many periods
PRICE_RATIO = 1.67          # unit prices must agree within this factor
MAX_OVERLAP = 1             # periods where both codes sold
GLOBAL_MIN_BRANCHES = 3     # a pair seen at this many branches is a company-wide re-code
SCALE_LO = 0.25             # the old history always counts for at least this share
OLD_LEVEL_MONTHS = 6        # the old code's level = its last this-many selling periods


def normalise_name(name) -> str:
    """Case, spacing and punctuation-insensitive form of a product name
    ("V BELT B 2591" == "V BELT B2591"). Specs stay in the name, so different
    sizes are never confused for one another."""
    return re.sub(r"[^A-Z0-9]+", "", str(name or "").upper())


def _unit_price(REV, MAT, i):
    if REV is None:
        return None
    q = float(MAT[i][MAT[i] > 0].sum())
    if q <= 0:
        return None
    r = float(REV[i][MAT[i] > 0].sum())
    return r / q if r > 0 else None


def detect(keys, item_of, MAT, REV=None) -> list:
    """-> list of ``{"branch", "old", "new", "name", "basis"}`` re-codes."""
    MAT = np.asarray(MAT, float)
    S, W = MAT.shape
    if S == 0 or W == 0:
        return []
    sold = MAT > 0
    nnz = sold.sum(1)
    first = np.where(nnz > 0, sold.argmax(1), W)
    last = np.where(nnz > 0, W - 1 - sold[:, ::-1].argmax(1), -1)

    groups = collections.defaultdict(list)
    for i, k in enumerate(keys):
        nm = normalise_name(item_of.get(k, ""))
        if len(nm) >= MIN_NAME_LEN and nnz[i] > 0:
            groups[(k[0], nm)].append(i)

    def handover(a, c):
        # old a ends no later than new c begins, with at most one shared period
        return (a != c and last[a] <= first[c] and last[c] > last[a]
                and int((sold[a] & sold[c]).sum()) <= MAX_OVERLAP)

    def price_ok(a, c):
        pa, pc = _unit_price(REV, MAT, a), _unit_price(REV, MAT, c)
        if not pa or not pc:
            return True
        r = pc / pa
        return 1 / PRICE_RATIO <= r <= PRICE_RATIO

    found = {}                 # (branch, old_sku) -> (new_sku, name, idx_old, idx_new)
    pair_branches = collections.defaultdict(set)
    loose = []                 # handover pairs that fail only the "established" test
    for (br, nm), idx in groups.items():
        if len(idx) < 2:
            continue
        for a in idx:
            cands = [c for c in idx if handover(a, c) and price_ok(a, c)]
            if not cands:
                continue
            c = min(cands, key=lambda j: (first[j], -nnz[j]))       # earliest successor
            if nnz[a] >= MIN_OLD_MONTHS:
                found[(br, keys[a][1])] = (keys[c][1], nm)
                pair_branches[(keys[a][1], keys[c][1])].add(br)
            else:
                loose.append((br, keys[a][1], keys[c][1], nm))

    global_pairs = {p for p, b in pair_branches.items() if len(b) >= GLOBAL_MIN_BRANCHES}
    out = [{"branch": br, "old": old, "new": new, "name": nm, "basis": "branch"}
           for (br, old), (new, nm) in found.items()]
    for br, old, new, nm in loose:                 # company-wide pair, thin local history
        if (old, new) in global_pairs and (br, old) not in found:
            out.append({"branch": br, "old": old, "new": new, "name": nm, "basis": "company-wide"})
    return _resolve_chains(out)


def _resolve_chains(merges: list) -> list:
    """old -> new -> newer  becomes  old -> newer, per branch (cycles dropped)."""
    to = {(m["branch"], m["old"]): m["new"] for m in merges}
    res = []
    for m in merges:
        tgt, seen = m["new"], {m["old"]}
        while (m["branch"], tgt) in to and tgt not in seen:
            seen.add(tgt)
            tgt = to[(m["branch"], tgt)]
        if tgt in seen:
            continue                                # cycle: leave it alone
        res.append({**m, "new": tgt})
    return res


def transfer_share(old_row, new_row) -> float:
    """Share (``SCALE_LO``..1) of the old code's volume the new code has taken
    over: its own average since it began / the old code's recent selling level."""
    old_row, new_row = np.asarray(old_row, float), np.asarray(new_row, float)
    sold = np.nonzero(new_row > 0)[0]
    old_act = old_row[old_row > 0][-OLD_LEVEL_MONTHS:]
    if not sold.size or not old_act.size or old_act.mean() <= 0:
        return 1.0
    own = float(new_row[sold[0]:].mean())
    return float(np.clip(own / float(old_act.mean()), SCALE_LO, 1.0))


def merge_panel(pan: dict, merges: list | None = None, scale: bool = True) -> dict:
    """A copy of ``pan`` with each old code's rows added into its successor
    (at :func:`transfer_share` of their size unless ``scale=False``) and the old
    series removed. ``pan['merges']`` records what was joined, with the share
    used. The input panel is never modified (it is shared/cached)."""
    keys = list(pan["keys"])
    if merges is None:
        merges = detect(keys, pan["item_of"], pan["MAT"], pan.get("REV"))
    index = {k: i for i, k in enumerate(keys)}
    pairs = []
    for m in merges:
        a, c = index.get((m["branch"], m["old"])), index.get((m["branch"], m["new"]))
        if a is not None and c is not None and a != c:
            pairs.append((a, c, m))
    if not pairs:
        return {**pan, "merges": []}

    arrays = {}
    for name in ("MAT", "PROFIT", "REV"):
        A = pan.get(name)
        if A is not None and getattr(A, "shape", None) is not None and A.shape[0] == len(keys):
            arrays[name] = np.array(A, copy=True)
    drop = set()
    orig = np.asarray(pan["MAT"], float)          # shares come from the untouched history
    for a, c, m in pairs:
        share = transfer_share(orig[a], orig[c]) if scale else 1.0
        m["share"] = round(share, 3)
        for A in arrays.values():
            A[c] = A[c] + share * A[a]
        drop.add(a)
    keep = [i for i in range(len(keys)) if i not in drop]
    out = dict(pan)
    for name, A in arrays.items():
        out[name] = A[keep]
    out["keys"] = [keys[i] for i in keep]
    out["item_of"] = {k: v for k, v in pan["item_of"].items() if k not in {keys[i] for i in drop}}
    out["merges"] = [m for _a, _c, m in pairs]
    return out


def aliases(merges: list) -> dict:
    """``{OLD_SKU: NEW_SKU}`` (upper-cased), taking the most common successor
    when a code was replaced differently at different branches."""
    votes = collections.defaultdict(collections.Counter)
    for m in merges:
        votes[str(m["old"]).upper()][str(m["new"]).upper()] += 1
    return {o: c.most_common(1)[0][0] for o, c in votes.items()}


def merge_report(raw_pan: dict, merges: list):
    """``(products, by_branch)`` DataFrames describing the joins, from the panel
    as recorded (``raw_pan``, before any joining) and the merge list. ``products``
    has one row per old -> new code (all branches); ``by_branch`` one row per
    branch join with the months each code sold."""
    import pandas as pd
    keys, weeks, MAT = raw_pan["keys"], raw_pan["weeks"], np.asarray(raw_pan["MAT"], float)
    index = {k: i for i, k in enumerate(keys)}

    def span(row):
        nz = np.nonzero(row > 0)[0]
        if not len(nz):
            return ""
        f = pd.Timestamp(weeks[nz[0]]).strftime("%b %Y")
        l = pd.Timestamp(weeks[nz[-1]]).strftime("%b %Y")
        return f"{f} - {l}"

    rows = []
    for m in merges:
        a, c = index.get((m["branch"], m["old"])), index.get((m["branch"], m["new"]))
        if a is None or c is None:
            continue
        rows.append({
            "Branch": m["branch"], "Product": raw_pan["item_of"].get((m["branch"], m["new"]), ""),
            "Old code": m["old"], "New code": m["new"],
            "Old code sold": span(MAT[a]), "New code sold": span(MAT[c]),
            "Old code units": int(MAT[a].sum()), "New code units": int(MAT[c].sum()),
            "Share of old history used in forecast": m.get("share"), "Basis": m.get("basis", "")})
    cols = ["Branch", "Product", "Old code", "New code", "Old code sold", "New code sold",
            "Old code units", "New code units", "Share of old history used in forecast", "Basis"]
    detail = pd.DataFrame(rows, columns=cols)
    if detail.empty:
        return pd.DataFrame(columns=["Product", "Old code", "New code", "Branches",
                                     "Old code units (all branches)", "New code units (all branches)",
                                     "Avg share used in forecast"]), detail
    prod = (detail.groupby(["Old code", "New code"], as_index=False)
                  .agg(Product=("Product", "first"), Branches=("Branch", "nunique"),
                       old_units=("Old code units", "sum"), new_units=("New code units", "sum"),
                       share=("Share of old history used in forecast", "mean")))
    prod["share"] = prod["share"].round(2)
    prod = prod.rename(columns={"old_units": "Old code units (all branches)",
                                "new_units": "New code units (all branches)",
                                "share": "Avg share used in forecast"})
    prod = prod.sort_values("Old code units (all branches)", ascending=False)[
        ["Product", "Old code", "New code", "Branches", "Old code units (all branches)",
         "New code units (all branches)", "Avg share used in forecast"]]
    detail = detail.sort_values(["Old code units", "Branch"], ascending=[False, True])
    return prod.reset_index(drop=True), detail.reset_index(drop=True)

