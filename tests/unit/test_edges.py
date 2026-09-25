# callcenter-intelligence
# tests/unit/test_edges.py

from __future__ import annotations

import pytest

from src.graph.edges import (
    ERROR,
    REDACT,
    REPORT,
    SUMMARIZE,
    SUPERVISOR_REVIEW,
    TRANSCRIBE,
    route_after_injection_check,
    route_after_intake,
    route_after_qa,
    route_after_transcription,
)
from src.graph.state import ComplianceFlag, IntakeResult, QADimensionScore, QAScoreResult, Severity


def _intake(passed: bool) -> IntakeResult:
    return IntakeResult(
        call_id="c1",
        validation_passed=passed,
        validation_error=None if passed else "Empty file",
    )


def _qa(flags: list[ComplianceFlag]) -> QAScoreResult:
    dim = QADimensionScore(score=3, justification="at 00:10")
    return QAScoreResult(
        call_id="c1",
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        compliance_flags=flags,
    )


# ---- route_after_intake ------------------------------------------------------


def test_valid_intake_goes_to_transcribe() -> None:
    assert route_after_intake({"intake": _intake(True)}) == TRANSCRIBE


def test_failed_intake_goes_to_error() -> None:
    assert route_after_intake({"intake": _intake(False)}) == ERROR


def test_missing_intake_goes_to_error() -> None:
    """An absent intake is a failure, not a pass. Defaulting the other way would
    send an unvalidated file to Whisper."""
    assert route_after_intake({}) == ERROR


# ---- route_after_transcription ----------------------------------------------


def test_transcription_always_forwards() -> None:
    assert route_after_transcription({}) == SUMMARIZE
    assert route_after_transcription({"error": "anything"}) == SUMMARIZE


# ---- route_after_injection_check --------------------------------------------


def test_clean_transcript_goes_to_redaction() -> None:
    assert route_after_injection_check({"status": "in_progress"}) == REDACT


def test_flagged_transcript_goes_to_error() -> None:
    assert route_after_injection_check({"status": "flagged_for_review"}) == ERROR


def test_error_in_state_goes_to_error() -> None:
    assert route_after_injection_check({"error": "injection detected"}) == ERROR


def test_injection_branch_precedes_every_llm_node() -> None:
    """The security guarantee is an ordering guarantee: the only destinations from
    the injection check are the redactor or the error terminal - never an LLM node."""
    for state in ({}, {"status": "flagged_for_review"}, {"error": "x"}):
        assert route_after_injection_check(state) in {REDACT, ERROR}


# ---- route_after_qa ----------------------------------------------------------


def test_no_flags_goes_to_report() -> None:
    assert route_after_qa({"qa_scores": _qa([])}) == REPORT


def test_critical_flag_goes_to_supervisor_review() -> None:
    flag = ComplianceFlag(description="No identity verification", severity=Severity.CRITICAL)
    assert route_after_qa({"qa_scores": _qa([flag])}) == SUPERVISOR_REVIEW


@pytest.mark.parametrize("severity", [Severity.LOW, Severity.MEDIUM, Severity.HIGH])
def test_non_critical_flags_still_report(severity: Severity) -> None:
    """Only CRITICAL escalates. If HIGH diverted too, every imperfect call would
    queue for a supervisor and the escalation path would lose its meaning."""
    flag = ComplianceFlag(description="Missing disclosure", severity=severity)
    assert route_after_qa({"qa_scores": _qa([flag])}) == REPORT


def test_critical_among_many_still_escalates() -> None:
    flags = [
        ComplianceFlag(description="a", severity=Severity.LOW),
        ComplianceFlag(description="b", severity=Severity.CRITICAL),
        ComplianceFlag(description="c", severity=Severity.MEDIUM),
    ]
    assert route_after_qa({"qa_scores": _qa(flags)}) == SUPERVISOR_REVIEW


def test_error_beats_escalation() -> None:
    flag = ComplianceFlag(description="x", severity=Severity.CRITICAL)
    assert route_after_qa({"error": "llm down", "qa_scores": _qa([flag])}) == ERROR


def test_missing_qa_goes_to_report() -> None:
    assert route_after_qa({}) == REPORT


def test_routing_functions_are_pure() -> None:
    """No mocks, no graph, no database in any test above - that is the point of
    keeping routing out of the node bodies."""
    state = {"intake": _intake(True), "qa_scores": _qa([])}
    before = dict(state)
    route_after_intake(state)
    route_after_qa(state)
    route_after_injection_check(state)
    assert state == before
