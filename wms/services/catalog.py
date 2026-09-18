"""Reference-data helpers: products, branches, users."""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from wms.audit import write_audit
from wms.errors import WMSError
from wms.models import Branch, Product, User


def upsert_product(db: Session, *, sku: str, name: str, category: Optional[str] = None,
                   uom: str = "EA", unit_price: Optional[float] = None,
                   user_id: Optional[int] = None) -> Product:
    p = db.query(Product).filter(Product.sku == sku).first()
    if p:
        p.name, p.category, p.uom = name, category, uom
        if unit_price is not None:
            p.unit_price = unit_price
    else:
        p = Product(sku=sku, name=name, category=category, uom=uom, unit_price=unit_price)
        db.add(p)
    write_audit(db, entity_type="Product", action="UPSERT", detail={"sku": sku}, user_id=user_id)
    db.commit()
    db.refresh(p)
    return p


def create_branch(db: Session, *, code: str, name: str, user_id: Optional[int] = None) -> Branch:
    if db.query(Branch).filter(Branch.code == code).first():
        raise WMSError(f"Branch '{code}' already exists.")
    b = Branch(code=code, name=name)
    db.add(b)
    write_audit(db, entity_type="Branch", action="CREATE", detail={"code": code}, user_id=user_id)
    db.commit()
    db.refresh(b)
    return b


def get_or_create_user(db: Session, *, username: str, full_name: Optional[str] = None,
                       role: str = "clerk") -> User:
    u = db.query(User).filter(User.username == username).first()
    if not u:
        u = User(username=username, full_name=full_name or username, role=role)
        db.add(u)
        db.commit()
        db.refresh(u)
    return u
