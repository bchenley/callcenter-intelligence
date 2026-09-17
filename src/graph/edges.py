# callcenter-intelligence
# src/graph/edges.py

from __future__ import annotations

from src.graph.state import PipelineState, Severity

# Routing lives here, separate from the node bodies, for one reason: a routing rule
# is a pure function of state and can be tested without building a graph, mocking an
# LLM or touching a database. Every branch below has a unit test.

TRANSCRIBE = "transcribe"
SUMMARIZE = "summarize"
REDACT = "redact"
REPORT = "report"
SUPERVISOR_REVIEW = "supervisor_review"
ERROR = "error"


def route_after_intake(state: PipelineState) -> str:
    """Validation is the gate: nothing reaches Whisper or an LLM until the file is
    a real, supported, in-limits audio file."""
    intake = state.get("intake")
    if intake is None or not intake.validation_passed:
        return ERROR
    return TRANSCRIBE


def route_after_transcription(state: PipelineState) -> str:
    """Always forward. Transcription failure is raised inside the node and lands in
    state['error'], so there is no second branch to take here - the constant return
    is the spec's, and it keeps the edge explicit rather than implied."""
    return SUMMARIZE


def route_after_injection_check(state: PipelineState) -> str:
    """Injection routes to error BEFORE redaction and before any LLM call. Ordering
    is the whole control: a transcript carrying 'ignore previous instructions' must
    never reach a model, so this branch precedes every LLM node in the graph."""
    if state.get("error") or state.get("status") == "flagged_for_review":
        return ERROR
    return REDACT


def route_after_qa(state: PipelineState) -> str:
    """A critical compliance flag diverts to a human instead of producing a report.
    Only CRITICAL diverts - high/medium/low are recorded in the report and coached,
    not escalated, or every imperfect call would queue for a supervisor."""
    if state.get("error"):
        return ERROR
    qa = state.get("qa_scores")
    if qa is not None and any(f.severity == Severity.CRITICAL for f in qa.compliance_flags):
        return SUPERVISOR_REVIEW
    return REPORT
