"""Product-catalogue resolution.

A back order / delivery note may reference a SKU that isn't in the catalogue yet
(new line item typed in, or a fresh code on an uploaded stock-movement doc).
Rather than rejecting it, we register the product on the fly so the document can
be saved; it can be enriched later.
"""
from __future__ import annotations

import difflib
import re
from typing import Optional

from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.errors import WMSError
from wms.models import Product


def resolve_product(db: Session, ref, *, name: Optional[str] = None,
                    unit_price: Optional[float] = None,
                    user_id: Optional[int] = None) -> Product:
    """Return the Product for ``ref`` (int id or SKU string).

    A numeric id that doesn't exist is an error.  An unknown SKU is **created**
    (name from the caller if given, else the SKU itself) and audited.
    """
    if isinstance(ref, int):
        p = db.query(Product).filter(Product.id == ref).first()
        if not p:
            raise WMSError(f"Unknown product id {ref}.")
        return p

    sku = str(ref or "").strip()
    if not sku:
        raise WMSError("A line needs a product code.")

    p = db.query(Product).filter(Product.sku == sku).first()
    if p:
        return p

    nm = (name or "").strip() or sku
    p = Product(sku=sku, name=nm[:200], category=None, uom="EA",
                unit_price=unit_price if unit_price is not None else 0)
    db.add(p)
    db.flush()
    write_audit(db, entity_type="Product", entity_id=p.id, action="AUTO_CREATE",
                detail={"sku": sku, "name": nm}, user_id=user_id)
    return p


def _normalize_desc(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def match_product_by_description(description: str, products: list[Product],
                                  *, threshold: float = 0.72) -> Optional[Product]:
    """Best-effort match of a free-text description (e.g. an OCR'd invoice line
    with no product code) against the existing catalogue, so a document import
    reuses a known product's real SKU instead of manufacturing a new one from
    the description text.

    Returns ``None`` when nothing is a confident enough match - the caller
    should fall back to its own SKU (and a new product gets created for it,
    same as any other unrecognised code)."""
    desc_n = _normalize_desc(description)
    if not desc_n:
        return None
    best, best_score = None, 0.0
    for p in products:
        name_n = _normalize_desc(p.name)
        if not name_n:
            continue
        score = difflib.SequenceMatcher(None, desc_n, name_n).ratio()
        if desc_n == name_n:
            score = 1.0
        elif desc_n in name_n or name_n in desc_n:
            score = max(score, 0.9)
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= threshold else None
