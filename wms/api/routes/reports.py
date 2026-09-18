"""Audit trail + Excel / CSV exports."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from wms.api.deps import current_user, db_session
from wms.analytics import loaders, monthly_sales
from wms.models import AuditLog, User
from wms.exports import excel, csv_export

router = APIRouter(prefix="/api/reports", tags=["reports"])

_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _xlsx(path):
    return FileResponse(path, media_type=_XLSX, filename=path.name)


def _csv(path):
    return FileResponse(path, media_type="text/csv", filename=path.name)


@router.get("/audit")
def audit(entity_type: str | None = None, entity_id: str | None = None,
          user_id: int | None = None, action: str | None = None,
          limit: int = 200, db: Session = Depends(db_session), _: User = Depends(current_user)):
    q = db.query(AuditLog)
    if entity_type:
        q = q.filter(AuditLog.entity_type == entity_type)
    if entity_id:
        q = q.filter(AuditLog.entity_id == str(entity_id))
    if user_id:
        q = q.filter(AuditLog.user_id == user_id)
    if action:
        q = q.filter(AuditLog.action == action)
    return [{"id": r.id, "ts": r.created_at, "entity_type": r.entity_type,
             "entity_id": r.entity_id, "action": r.action, "detail": r.detail,
             "reason": r.reason, "user_id": r.user_id}
            for r in q.order_by(AuditLog.id.desc()).limit(min(limit, 2000))]


# ---- Excel ----
@router.get("/backorders.xlsx")
def backorders_xlsx(branch_id: int | None = None, db: Session = Depends(db_session),
                    _: User = Depends(current_user)):
    return _xlsx(excel.backorder_workbook(db, branch_id=branch_id))


@router.get("/backorder-flow.xlsx")
def backorder_flow_xlsx(branch_id: int | None = None, db: Session = Depends(db_session),
                        _: User = Depends(current_user)):
    return _xlsx(excel.backorder_flow_workbook(db, branch_id=branch_id))


@router.get("/branch-sales.xlsx")
def branch_sales_xlsx(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return _xlsx(excel.sales_workbook(db))


@router.get("/suggested-orders.xlsx")
def suggested_orders_xlsx(branch_id: int | None = None, db: Session = Depends(db_session),
                          _: User = Depends(current_user)):
    return _xlsx(excel.suggested_orders_workbook(db, branch_id=branch_id))


@router.get("/allocation-plan.xlsx")
def allocation_plan_xlsx(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return _xlsx(excel.allocation_workbook(db))


@router.get("/branch-stats.xlsx")
def branch_stats_xlsx(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return _xlsx(excel.branch_stats_workbook(db))


# ---- CSV ----
@router.get("/back-orders.csv")
def back_orders_csv(db: Session = Depends(db_session), _: User = Depends(current_user)):
    return _csv(csv_export.dataframe_to_csv(loaders.back_orders_df(db), "back_orders"))


@router.get("/sales.csv")
def sales_csv(days: int = 180, db: Session = Depends(db_session),
              _: User = Depends(current_user)):
    """Real monthly Hansa exports (data/sales_history), not the SalesRecord
    table - its sales history is almost entirely fabricated demo/seed data."""
    return _csv(csv_export.dataframe_to_csv(monthly_sales.cached_panel(), "sales"))
