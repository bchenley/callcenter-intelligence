# callcenter-intelligence
# tests/unit/test_observability.py

from __future__ import annotations

import json

import pytest

from src.database.connection import session_scope
from src.database.models import CallRecord
from src.graph.state import CallStatus
from src.security.audit import AuditLogger
from src.services.observability import (
    AUDIT_COLUMNS,
    RECENT_AUDIT_LIMIT,
    collect_metrics,
    format_langsmith_status,
    format_metrics,
    get_observability_dashboard,
)


def _record(engine, call_id: str, status: str, score: float | None = None, flags: int = 0) -> None:
    qa = None
    if score is not None:
        qa = json.dumps(
            {
                "overall_score": score,
                "compliance_flags": [
                    {"description": f"f{i}", "severity": "high"} for i in range(flags)
                ],
            }
        )
    with session_scope(engine) as s:
        s.add(CallRecord(call_id=call_id, status=status, qa_scores_json=qa))


def test_empty_database_is_all_zero(engine) -> None:
    m = collect_metrics(engine)
    assert (m.total_calls, m.completed, m.failed, m.flagged) == (0, 0, 0, 0)
    assert m.success_rate == 0.0
    assert m.average_qa_score is None
    assert m.recent_audit == []


def test_counts_by_status(engine) -> None:
    _record(engine, "c1", CallStatus.COMPLETED.value, 4.0)
    _record(engine, "c2", CallStatus.COMPLETED.value, 3.0)
    _record(engine, "c3", CallStatus.FAILED.value)
    _record(engine, "c4", CallStatus.FLAGGED_FOR_REVIEW.value, 2.0)
    m = collect_metrics(engine)
    assert (m.total_calls, m.completed, m.failed, m.flagged) == (4, 2, 1, 1)


def test_success_rate_denominator_is_every_call(engine) -> None:
    """A crashed call is a pipeline failure. Excluding it would let the rate climb
    while the system got worse."""
    _record(engine, "c1", CallStatus.COMPLETED.value, 4.0)
    _record(engine, "c2", CallStatus.FAILED.value)
    assert collect_metrics(engine).success_rate == 50.0


def test_average_qa_ignores_unscored_calls(engine) -> None:
    _record(engine, "c1", CallStatus.COMPLETED.value, 5.0)
    _record(engine, "c2", CallStatus.COMPLETED.value, 3.0)
    _record(engine, "c3", CallStatus.FAILED.value)
    assert collect_metrics(engine).average_qa_score == 4.0


def test_compliance_flags_counted_across_calls(engine) -> None:
    _record(engine, "c1", CallStatus.COMPLETED.value, 4.0, flags=2)
    _record(engine, "c2", CallStatus.COMPLETED.value, 4.0, flags=1)
    assert collect_metrics(engine).compliance_flags == 3


def test_corrupt_qa_json_does_not_crash_the_dashboard(engine) -> None:
    """The dashboard must render even when one row is unparseable."""
    with session_scope(engine) as s:
        s.add(
            CallRecord(call_id="bad", status=CallStatus.COMPLETED.value, qa_scores_json="{not json")
        )
    _record(engine, "good", CallStatus.COMPLETED.value, 4.0)
    m = collect_metrics(engine)
    assert m.total_calls == 2
    assert m.average_qa_score == 4.0


def test_recent_audit_capped_at_twenty_newest_first(engine) -> None:
    audit = AuditLogger(engine)
    for i in range(25):
        audit.log(f"call-{i:02d}", "intake_passed")
    rows = collect_metrics(engine).recent_audit
    assert len(rows) == RECENT_AUDIT_LIMIT == 20
    assert rows[0][1] == "call-24"


def test_audit_row_shape_matches_the_columns(engine) -> None:
    AuditLogger(engine).log("c1", "pii_redacted", details={"types": ["SSN"], "pii_found": True})
    row = collect_metrics(engine).recent_audit[0]
    assert len(row) == len(AUDIT_COLUMNS) == 4
    assert row[1] == "c1" and row[2] == "pii_redacted"
    assert "pii_found=True" in row[3] and "SSN" in row[3]


def test_details_summary_is_single_line(engine) -> None:
    AuditLogger(engine).log("c1", "analyzed", details={"overall_score": 4.0})
    assert "\n" not in collect_metrics(engine).recent_audit[0][3]


def test_audit_event_total_counts_all_not_just_recent(engine) -> None:
    audit = AuditLogger(engine)
    for i in range(30):
        audit.log("c1", "intake_passed")
    m = collect_metrics(engine)
    assert m.audit_events == 30 and len(m.recent_audit) == 20


def test_format_metrics_is_ascii_and_complete(engine) -> None:
    _record(engine, "c1", CallStatus.COMPLETED.value, 4.0, flags=1)
    text = format_metrics(collect_metrics(engine))
    text.encode("ascii")
    for label in (
        "Total calls",
        "Completed",
        "Failed",
        "Flagged",
        "Success rate",
        "Average QA score",
        "Compliance flags",
        "Audit events",
    ):
        assert label in text


def test_average_renders_na_when_nothing_scored(engine) -> None:
    assert "n/a" in format_metrics(collect_metrics(engine))


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_langsmith_status_reflects_env(monkeypatch, enabled: str) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", enabled)
    monkeypatch.setenv("LANGCHAIN_PROJECT", "callcenter")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "k")
    text = format_langsmith_status()
    text.encode("ascii")
    assert ("enabled" in text) is (enabled == "true")


def test_dashboard_returns_three_parts(engine) -> None:
    _record(engine, "c1", CallStatus.COMPLETED.value, 4.0)
    AuditLogger(engine).log("c1", "completed")
    metrics_md, langsmith_md, rows = get_observability_dashboard(engine)
    assert metrics_md.startswith("## Pipeline health")
    assert langsmith_md.startswith("## LangSmith")
    assert isinstance(rows, list) and len(rows[0]) == 4
