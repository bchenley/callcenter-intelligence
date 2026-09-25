# callcenter-intelligence
# tests/unit/test_report.py

from __future__ import annotations

import pytest
from sqlalchemy import select

from src.agents.report import (
    compile_report,
    generate_report_json,
    generate_report_pdf,
    persist_report,
)
from src.database.connection import session_scope
from src.database.models import CallRecord
from src.graph.state import (
    ActionItem,
    CallReport,
    CallStatus,
    ComplianceFlag,
    Entity,
    IntakeResult,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    Severity,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)
from src.utils.formatters import DIMENSION_LABELS, NO_FLAGS_TEXT

CALL_ID = "call-abc-123"


def _intake() -> IntakeResult:
    return IntakeResult(
        call_id=CALL_ID, validation_passed=True, filename="billing.wav", audio_format="wav"
    )


def _transcript(call_id: str = CALL_ID) -> TranscriptionResult:
    return TranscriptionResult(
        call_id=call_id,
        full_text="Thank you for calling. I was charged twice.",
        segments=[
            TranscriptionSegment(
                start=0.0, end=4.0, text="Thank you for calling.", speaker="Agent", confidence=0.91
            )
        ],
    )


def _summary(call_id: str = CALL_ID) -> SummaryResult:
    return SummaryResult(
        call_id=call_id,
        call_purpose="Duplicate charge on the March invoice.",
        key_discussion_points=["Charged twice on 3 March"],
        action_items=[
            ActionItem(description="Process refund", owner="Agent", deadline="3 business days")
        ],
        resolution_status=ResolutionStatus.RESOLVED,
        sentiment_trajectory="frustrated at open, satisfied at close",
        entities=[Entity(name="March invoice", type="document")],
    )


def _qa(call_id: str = CALL_ID, flags: list[ComplianceFlag] | None = None) -> QAScoreResult:
    dim = QADimensionScore(score=4, justification="At 00:12 the agent greeted by name.")
    return QAScoreResult(
        call_id=call_id,
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        compliance_flags=flags or [],
        overall_score=4.0,
    )


def _compiled(**kwargs) -> CallReport:
    return compile_report(_intake(), _transcript(), _summary(), _qa(), **kwargs)


def test_compile_report_propagates_call_id_throughout() -> None:
    """compile_report assembles all upstream results with the correct call_id propagated throughout"""
    report = _compiled()
    assert report.call_id == CALL_ID
    assert report.transcript is not None and report.transcript.call_id == CALL_ID
    assert report.summary is not None and report.summary.call_id == CALL_ID
    assert report.qa_scores is not None and report.qa_scores.call_id == CALL_ID
    assert report.filename == "billing.wav"
    assert report.summary.call_purpose == "Duplicate charge on the March invoice."


def test_compile_report_overwrites_empty_nested_call_ids() -> None:
    report = compile_report(_intake(), _transcript(""), _summary(""), _qa(""))
    assert report.call_id == CALL_ID
    assert report.transcript is not None and report.transcript.call_id == CALL_ID
    assert report.summary is not None and report.summary.call_id == CALL_ID
    assert report.qa_scores is not None and report.qa_scores.call_id == CALL_ID


@pytest.mark.parametrize("which", ["transcription", "summary", "qa_scores"])
def test_compile_report_raises_on_nonempty_call_id_mismatch(which: str) -> None:
    transcription, summary, qa = _transcript(), _summary(), _qa()
    if which == "transcription":
        transcription = _transcript("other")
    elif which == "summary":
        summary = _summary("other")
    else:
        qa = _qa("other")
    with pytest.raises(ValueError, match="other"):
        compile_report(_intake(), transcription, summary, qa)


def test_compile_report_does_not_mutate_inputs() -> None:
    summary, qa = _summary(""), _qa("")
    compile_report(_intake(), _transcript(""), summary, qa)
    assert summary.call_id == ""
    assert qa.call_id == ""


def test_compile_report_carries_trace_id_and_processing_seconds() -> None:
    report = _compiled(
        status=CallStatus.FLAGGED_FOR_REVIEW, trace_id="trace-1", processing_seconds=12.5
    )
    assert report.status == CallStatus.FLAGGED_FOR_REVIEW
    assert report.trace_id == "trace-1"
    assert report.processing_seconds == 12.5


def test_generate_report_pdf_magic_bytes() -> None:
    """generate_report_pdf returns bytes where result[:5] == b"%PDF-" """
    result = generate_report_pdf(_compiled())
    assert result[:5] == b"%PDF-"


def test_pdf_contains_summary_qa_and_flags() -> None:
    pdf = generate_report_pdf(_compiled())
    for token in (
        b"Duplicate charge",
        b"resolved",
        b"frustrated",
        b"Process refund",
        b"March invoice",
        NO_FLAGS_TEXT.encode("ascii"),
    ):
        assert token in pdf
    for label in DIMENSION_LABELS.values():
        assert label.encode("ascii") in pdf


def test_pdf_includes_compliance_flags() -> None:
    flag = ComplianceFlag(
        description="Account accessed before identity verification",
        severity=Severity.CRITICAL,
        timestamp_reference="02:15-02:45",
    )
    pdf = generate_report_pdf(
        compile_report(_intake(), _transcript(), _summary(), _qa(flags=[flag]))
    )
    assert b"[CRITICAL]" in pdf
    assert b"Account accessed before identity verification" in pdf
    assert b"02:15-02:45" in pdf
    assert NO_FLAGS_TEXT.encode("ascii") not in pdf


def test_pdf_requires_summary() -> None:
    with pytest.raises(ValueError, match="summary"):
        generate_report_pdf(CallReport(call_id=CALL_ID))


def test_pdf_requires_qa_scores() -> None:
    with pytest.raises(ValueError, match="qa_scores"):
        generate_report_pdf(CallReport(call_id=CALL_ID, summary=_summary()))


def test_pdf_survives_xml_special_chars() -> None:
    summary = _summary().model_copy(update={"call_purpose": "Charge < $50 & more"})
    pdf = generate_report_pdf(compile_report(_intake(), _transcript(), summary, _qa()))
    assert pdf[:5] == b"%PDF-"
    assert b"Charge" in pdf


def test_generate_report_json_contains_call_id_and_summary() -> None:
    """generate_report_json returns a string containing both '"call_id"' and '"summary"'"""
    payload = generate_report_json(_compiled())
    assert '"call_id"' in payload
    assert '"summary"' in payload


def test_generate_report_json_includes_nulls() -> None:
    payload = generate_report_json(CallReport(call_id="c-null"))
    assert '"transcript": null' in payload
    assert '"summary": null' in payload
    assert '"trace_id": null' in payload


def test_persist_report_then_query_by_call_id(engine) -> None:
    """persist_report writes to the database; querying by call_id retrieves the record in the same session

    Queries with a new SQLAlchemy session after commit, not the same Session object.
    """
    report = _compiled(trace_id="trace-1", processing_seconds=12.5)
    persist_report(report, engine)
    with session_scope(engine) as session:
        rec = session.scalar(select(CallRecord).where(CallRecord.call_id == CALL_ID))
        assert rec is not None
        assert rec.status == "completed"
        assert rec.audio_filename == "billing.wav"
        assert rec.transcript_text == report.transcript.full_text
        assert rec.trace_id == "trace-1"
        assert rec.summary_json is not None
        assert rec.qa_scores_json is not None
        assert rec.report_json == generate_report_json(report)
        assert rec.processed_at is not None
