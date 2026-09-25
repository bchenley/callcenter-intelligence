# callcenter-intelligence
# src/agents/report.py

from __future__ import annotations

import io
from datetime import UTC, datetime
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sqlalchemy import Engine

from src.database.connection import session_scope
from src.database.models import CallRecord
from src.graph.state import (
    CallReport,
    CallStatus,
    IntakeResult,
    QAScoreResult,
    SummaryResult,
    TranscriptionResult,
)
from src.utils.formatters import DIMENSION_LABELS, NO_FLAGS_TEXT, SEVERITY_MARKERS


def compile_report(
    intake: IntakeResult,
    transcription: TranscriptionResult,
    summary: SummaryResult,
    qa_scores: QAScoreResult,
    *,
    status: CallStatus = CallStatus.COMPLETED,
    error: str | None = None,
    trace_id: str | None = None,
    processing_seconds: float | None = None,
) -> CallReport:
    call_id = intake.call_id
    for name, nested in (
        ("transcription", transcription),
        ("summary", summary),
        ("qa_scores", qa_scores),
    ):
        # Empty string is the Pydantic default on SummaryResult/QAScoreResult, not
        # a competing identity. A non-empty disagreement would persist the wrong call.
        if nested.call_id and nested.call_id != call_id:
            raise ValueError(
                f"{name}.call_id {nested.call_id!r} disagrees with intake.call_id {call_id!r}"
            )
    # model_copy: overwriting an empty nested call_id must not mutate PipelineState.
    return CallReport(
        call_id=call_id,
        status=status,
        filename=intake.filename,
        transcript=transcription.model_copy(update={"call_id": call_id}),
        summary=summary.model_copy(update={"call_id": call_id}),
        qa_scores=qa_scores.model_copy(update={"call_id": call_id}),
        processed_at=datetime.now(UTC),
        processing_seconds=processing_seconds,
        error=error,
        trace_id=trace_id,
    )


def generate_report_json(report: CallReport) -> str:
    # exclude_none would drop keys that consumers expect to be present when a field is unset.
    return report.model_dump_json(indent=2)


def persist_report(report: CallReport, engine: Engine) -> None:
    with session_scope(engine) as session:
        record = CallRecord(
            call_id=report.call_id,
            status=report.status.value,
            audio_filename=report.filename,
            transcript_text=report.transcript.full_text if report.transcript else None,
            summary_json=report.summary.model_dump_json() if report.summary else None,
            qa_scores_json=report.qa_scores.model_dump_json() if report.qa_scores else None,
            # Same serializer as the downloadable JSON so the two cannot drift.
            report_json=generate_report_json(report),
            trace_id=report.trace_id,
        )
        if report.processed_at is not None:
            # Passing None would override the column default with NULL.
            record.processed_at = report.processed_at
        session.add(record)


def _para(text: str, style: object) -> Paragraph:
    # Paragraph parses XML; a justification containing "<" would truncate the cell.
    return Paragraph(escape(text), style)


def generate_report_pdf(report: CallReport) -> bytes:
    if report.summary is None:
        raise ValueError("generate_report_pdf requires report.summary")
    if report.qa_scores is None:
        raise ValueError("generate_report_pdf requires report.qa_scores")
    summary, qa = report.summary, report.qa_scores
    styles = getSampleStyleSheet()
    story = [
        _para("Call Report", styles["Title"]),
        _para(f"Call ID: {report.call_id}", styles["Normal"]),
        _para(f"Status: {report.status.value}", styles["Normal"]),
    ]
    if report.filename:
        story.append(_para(f"File: {report.filename}", styles["Normal"]))
    if report.processing_seconds is not None:
        story.append(_para(f"Processing time: {report.processing_seconds:.2f}s", styles["Normal"]))

    story += [
        Spacer(1, 12),
        _para("Summary", styles["Heading2"]),
        _para(f"Purpose: {summary.call_purpose}", styles["Normal"]),
        _para(f"Resolution: {summary.resolution_status.value}", styles["Normal"]),
        _para(f"Sentiment: {summary.sentiment_trajectory}", styles["Normal"]),
    ]
    if summary.key_discussion_points:
        story.append(_para("Key discussion points:", styles["Normal"]))
        story.extend(_para(f"- {p}", styles["Normal"]) for p in summary.key_discussion_points)
    if summary.action_items:
        story.append(_para("Action items:", styles["Normal"]))
        for item in summary.action_items:
            deadline = f" (due {item.deadline})" if item.deadline else ""
            story.append(
                _para(f"- {item.description} - owner: {item.owner}{deadline}", styles["Normal"])
            )
    if summary.entities:
        story.append(_para("Entities:", styles["Normal"]))
        story.extend(_para(f"- {e.name} ({e.type})", styles["Normal"]) for e in summary.entities)

    story += [
        Spacer(1, 12),
        _para("QA Scores", styles["Heading2"]),
        _para(f"Overall: {qa.overall_score:.2f} / 5.00", styles["Normal"]),
    ]
    for field, label in DIMENSION_LABELS.items():
        dim = getattr(qa, field)
        story.append(_para(f"{label}: {dim.score}/5", styles["Normal"]))
        story.append(_para(dim.justification, styles["Normal"]))

    story += [Spacer(1, 12), _para("Compliance Flags", styles["Heading2"])]
    if not qa.compliance_flags:
        story.append(_para(NO_FLAGS_TEXT, styles["Normal"]))
    else:
        for flag in qa.compliance_flags:
            marker = SEVERITY_MARKERS[flag.severity]
            ref = f" at {flag.timestamp_reference}" if flag.timestamp_reference else ""
            story.append(_para(f"{marker} {flag.description}{ref}", styles["Normal"]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    # Uncompressed so a reviewer can strings(1) the Summary, QA Scores, and Compliance Flags sections.
    doc.pageCompression = 0
    doc.build(story)
    return buf.getvalue()
