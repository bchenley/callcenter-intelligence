# callcenter-intelligence
# tests/security/test_audit.py

from __future__ import annotations

import json

import pytest
from sqlalchemy import inspect

from src.database.models import AuditLogEntry
from src.security.audit import AuditLogger


def test_append_only_second_write_keeps_first(audit_logger: AuditLogger) -> None:
    audit_logger.log("call-1", "intake")
    audit_logger.log("call-1", "transcribe")
    assert [e.action for e in audit_logger.entries_for_call("call-1")] == ["intake", "transcribe"]


def test_writes_to_the_database_not_a_file(audit_logger: AuditLogger, engine) -> None:
    audit_logger.log("call-1", "intake")
    assert "audit_log" in inspect(engine).get_table_names()
    assert audit_logger.count() == 1


def test_no_mutation_api_exposed() -> None:
    """Append-only is enforced by the class surface, not by convention."""
    for forbidden in ("update", "delete", "edit", "remove", "purge"):
        assert not any(forbidden in name for name in dir(AuditLogger) if not name.startswith("_"))


def test_entry_fields(audit_logger: AuditLogger) -> None:
    audit_logger.log("call-1", "pii_redaction", user="pipeline", details={"types": ["SSN"]})
    entry = audit_logger.entries_for_call("call-1")[0]
    assert entry.call_id == "call-1"
    assert entry.action == "pii_redaction"
    assert entry.user == "pipeline"
    assert json.loads(entry.details) == {"types": ["SSN"]}
    assert entry.timestamp is not None


def test_details_optional(audit_logger: AuditLogger) -> None:
    audit_logger.log("call-1", "intake")
    assert audit_logger.entries_for_call("call-1")[0].details is None


def test_user_defaults_to_app(audit_logger: AuditLogger) -> None:
    """FAQ: 'the user (always "app" in the automated pipeline, or a specific user
    identifier in supervised workflows)'."""
    audit_logger.log("call-1", "intake")
    assert audit_logger.entries_for_call("call-1")[0].user == "app"


def test_entries_scoped_by_call_id(audit_logger: AuditLogger) -> None:
    audit_logger.log("call-1", "intake")
    audit_logger.log("call-1", "transcribe")
    audit_logger.log("call-2", "intake")
    assert len(audit_logger.entries_for_call("call-1")) == 2
    assert len(audit_logger.entries_for_call("call-2")) == 1


def test_recent_is_newest_first_and_limited(audit_logger: AuditLogger) -> None:
    for i in range(25):
        audit_logger.log(f"call-{i}", "intake")
    recent = audit_logger.recent(limit=20)
    assert len(recent) == 20
    assert recent[0].call_id == "call-24"


def test_recent_default_limit_is_twenty(audit_logger: AuditLogger) -> None:
    for i in range(30):
        audit_logger.log(f"call-{i}", "intake")
    assert len(audit_logger.recent()) == 20


def test_log_returns_row_id(audit_logger: AuditLogger) -> None:
    first = audit_logger.log("call-1", "intake")
    second = audit_logger.log("call-1", "transcribe")
    assert isinstance(first, int) and second > first


def test_empty_log_reads_empty(audit_logger: AuditLogger) -> None:
    assert audit_logger.entries_for_call("nobody") == []
    assert audit_logger.count() == 0


@pytest.mark.parametrize(
    "action", ["intake", "transcribe", "injection_block", "pii_redaction", "report"]
)
def test_actions_round_trip(audit_logger: AuditLogger, action: str) -> None:
    audit_logger.log("call-1", action)
    assert audit_logger.entries_for_call("call-1")[0].action == action


def test_call_id_is_indexed_but_not_unique(engine) -> None:
    indexes = inspect(engine).get_indexes("audit_log")
    call_id_idx = [i for i in indexes if i["column_names"] == ["call_id"]]
    assert call_id_idx, "audit_log.call_id must be indexed"
    assert bool(call_id_idx[0]["unique"]) is False, "one call writes many audit rows"


def test_model_has_required_columns(engine) -> None:
    cols = {c["name"] for c in inspect(engine).get_columns(AuditLogEntry.__tablename__)}
    assert {"id", "call_id", "action", "user", "timestamp", "details"} <= cols
