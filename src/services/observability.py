# callcenter-intelligence
# src/services/observability.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from sqlalchemy import Engine, select

from src.database.connection import session_scope
from src.database.models import AuditLogEntry, CallRecord
from src.graph.state import CallStatus

RECENT_AUDIT_LIMIT = 20
AUDIT_COLUMNS: tuple[str, ...] = ("Timestamp", "Call ID", "Action", "Details")


@dataclass(frozen=True)
class PipelineMetrics:
    total_calls: int = 0
    completed: int = 0
    failed: int = 0
    flagged: int = 0
    success_rate: float = 0.0
    average_qa_score: float | None = None
    compliance_flags: int = 0
    audit_events: int = 0
    recent_audit: list[list[str]] = field(default_factory=list)


def _details_summary(raw: str | None) -> str:
    """One line. The dashboard is a table; a pretty-printed JSON blob in a cell
    makes every row tall and the table unreadable."""
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return raw[:120]
    if not isinstance(parsed, dict):
        return str(parsed)[:120]
    return ", ".join(f"{k}={v}" for k, v in sorted(parsed.items()) if v not in (None, [], {}))[:120]


def collect_metrics(engine: Engine) -> PipelineMetrics:
    """Every query in one session_scope. Six separate sessions for six numbers on
    one screen is six round trips and six chances to read a half-written state."""
    with session_scope(engine) as session:
        records = list(session.scalars(select(CallRecord)))
        audit = list(session.scalars(select(AuditLogEntry).order_by(AuditLogEntry.id.desc())))

    total = len(records)
    completed = sum(1 for r in records if r.status == CallStatus.COMPLETED.value)
    failed = sum(1 for r in records if r.status == CallStatus.FAILED.value)
    flagged = sum(1 for r in records if r.status == CallStatus.FLAGGED_FOR_REVIEW.value)

    scores: list[float] = []
    flags = 0
    for record in records:
        if not record.qa_scores_json:
            continue
        try:
            payload = json.loads(record.qa_scores_json)
        except ValueError:
            continue
        if isinstance(payload.get("overall_score"), (int, float)):
            scores.append(float(payload["overall_score"]))
        flags += len(payload.get("compliance_flags") or [])

    return PipelineMetrics(
        total_calls=total,
        completed=completed,
        failed=failed,
        flagged=flagged,
        # Denominator is every call, not every scored call: a crash is a failure of
        # the pipeline and hiding it would make the rate meaningless.
        success_rate=round(completed / total * 100, 1) if total else 0.0,
        average_qa_score=round(sum(scores) / len(scores), 2) if scores else None,
        compliance_flags=flags,
        audit_events=len(audit),
        recent_audit=[
            [
                e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else "",
                e.call_id,
                e.action,
                _details_summary(e.details),
            ]
            for e in audit[:RECENT_AUDIT_LIMIT]
        ],
    )


def format_metrics(metrics: PipelineMetrics) -> str:
    average = (
        "n/a" if metrics.average_qa_score is None else f"{metrics.average_qa_score:.2f} / 5.00"
    )
    return "\n".join(
        [
            "## Pipeline health",
            "",
            f"- Total calls processed: {metrics.total_calls}",
            f"- Completed: {metrics.completed}",
            f"- Failed: {metrics.failed}",
            f"- Flagged for review: {metrics.flagged}",
            f"- Success rate: {metrics.success_rate:.1f}%",
            f"- Average QA score: {average}",
            f"- Compliance flags raised: {metrics.compliance_flags}",
            f"- Audit events recorded: {metrics.audit_events}",
        ]
    )


def format_langsmith_status() -> str:
    enabled = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in {"1", "true", "yes"}
    project = os.getenv("LANGCHAIN_PROJECT", "").strip() or "not set"
    if not enabled:
        return (
            "## LangSmith\n\nTracing disabled. Set `LANGCHAIN_TRACING_V2=true` and "
            "`LANGCHAIN_API_KEY` to record per-node latency and token counts."
        )
    key = "present" if os.getenv("LANGCHAIN_API_KEY", "").strip() else "MISSING"
    return f"## LangSmith\n\nTracing enabled. Project: {project}. API key: {key}."


def get_observability_dashboard(engine: Engine) -> tuple[str, str, list[list[str]]]:
    """(metrics markdown, langsmith markdown, audit rows) - the three things the
    Observability tab renders."""
    metrics = collect_metrics(engine)
    return format_metrics(metrics), format_langsmith_status(), metrics.recent_audit
