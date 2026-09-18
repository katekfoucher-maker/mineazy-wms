"""Statistics, backorder analysis, forecasting and allocation."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from wms.api.deps import current_user, db_session
from wms.models import User
from wms.analytics import loaders, statistics, allocation
from wms.analytics import backorders as bo_an

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _recs(df):
    return df.to_dict("records") if df is not None and not df.empty else []


@router.get("/overall")
def overall(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return statistics.overall_kpis(db)


@router.get("/branch/{branch_id}")
def branch(branch_id: int, db: Session = Depends(db_session), _: User = Depends(current_user)):
    return statistics.branch_kpis(db, branch_id)


@router.get("/branch-comparison")
def branch_comparison(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return _recs(statistics.branch_comparison(db))


@router.get("/abc")
def abc(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return _recs(statistics.abc_classification(db))


@router.get("/sales-trend")
def sales_trend(branch_id: int | None = None, db: Session = Depends(db_session),
                _: User = Depends(current_user)):
    return _recs(statistics.sales_trend(db, branch_id=branch_id))


@router.get("/backorders")
def backorders(branch_id: int | None = None, db: Session = Depends(db_session),
               _: User = Depends(current_user)):
    dl = loaders.dn_lines_df(db, branch_id=branch_id)
    return {
        "summary": bo_an.overall_summary(dl),
        "by_fill_status": _recs(bo_an.by_fill_status(dl)),
        "by_branch": _recs(bo_an.by_branch(dl)),
        "by_item": _recs(bo_an.by_item(dl, top=25)),
        "by_category": _recs(bo_an.by_category(dl)),
        "ageing": _recs(bo_an.ageing(dl)),
        "trend": _recs(bo_an.trend(dl)),
        "recurring": _recs(bo_an.recurring(dl, min_occurrences=1)),
    }


@router.get("/forecast")
def forecast_table(branch_id: int | None = None, db: Session = Depends(db_session),
                   _: User = Depends(current_user)):
    df = allocation._real_forecast_table(db)
    if branch_id and not df.empty:
        df = df[df.branch_id == branch_id]
    return _recs(df)


@router.get("/suggested-orders")
def suggested_orders(branch_id: int | None = None, db: Session = Depends(db_session),
                     _: User = Depends(current_user)):
    df = allocation._real_forecast_table(db)
    if branch_id and not df.empty:
        df = df[df.branch_id == branch_id]
    if not df.empty:
        df = df[df.suggested_order_qty > 0]
    return _recs(df)


@router.get("/allocation-plan")
def allocation_plan(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return _recs(allocation.allocation_plan(db))


@router.post("/allocate")
def allocate(product_id: int = Query(...), available_qty: int = Query(..., ge=0),
             db: Session = Depends(db_session), _: User = Depends(current_user)):
    return allocation.allocate_product(db, product_id=product_id, available_qty=available_qty)


@router.get("/branch-sales")
def branch_sales(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return _recs(statistics.branch_sales_summary(db))
