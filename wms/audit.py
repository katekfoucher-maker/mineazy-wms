"""Audit-log writer for non-quantity actions.

Quantity history is the ``stock_movements`` table; this records approvals,
cancellations, threshold changes, imports, etc. Both carry ``user_id``.
"""
from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from wms.models import AuditLog


def write_audit(
    db: Session,
    *,
    entity_type: str,
    action: str,
    entity_id: Optional[object] = None,
    detail: Optional[dict] = None,
    reason: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """Append an audit row. Does NOT commit - the caller owns the transaction."""
    db.add(AuditLog(
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        action=action,
        detail=json.dumps(detail, default=str) if detail is not None else None,
        reason=reason,
        user_id=user_id,
    ))
