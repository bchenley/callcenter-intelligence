# callcenter-intelligence
# src/security/audit.py

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Engine, select

from src.database.connection import session_scope
from src.database.models import AuditLogEntry

# Append-only by construction: this class exposes log() and readers, and nothing
# that updates or deletes. An audit trail that can be edited is not an audit trail.


class AuditLogger:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def log(
        self,
        call_id: str,
        action: str,
        # FAQ: the automated pipeline records user="app"; a named identifier is used
        # only in supervised workflows such as supervisor review.
        user: str = "app",
        details: dict[str, Any] | None = None,
    ) -> int:
        """Returns the new row id. details is serialized to JSON text so the schema
        does not have to grow a column every time a node records something new."""
        with session_scope(self.engine) as session:
            entry = AuditLogEntry(
                call_id=call_id,
                action=action,
                user=user,
                details=json.dumps(details, sort_keys=True) if details is not None else None,
            )
            session.add(entry)
            session.flush()
            return entry.id

    def entries_for_call(self, call_id: str) -> list[AuditLogEntry]:
        with session_scope(self.engine) as session:
            stmt = (
                select(AuditLogEntry)
                .where(AuditLogEntry.call_id == call_id)
                .order_by(AuditLogEntry.id)
            )
            return list(session.scalars(stmt))

    def recent(self, limit: int = 20) -> list[AuditLogEntry]:
        """Most recent first - the observability tab shows the latest 20 events."""
        with session_scope(self.engine) as session:
            stmt = select(AuditLogEntry).order_by(AuditLogEntry.id.desc()).limit(limit)
            return list(session.scalars(stmt))

    def count(self) -> int:
        with session_scope(self.engine) as session:
            return len(list(session.scalars(select(AuditLogEntry.id))))
