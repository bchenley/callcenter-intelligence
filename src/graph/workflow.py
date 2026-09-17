# callcenter-intelligence
# src/graph/workflow.py

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph
from sqlalchemy import Engine

from src.agents.intake import run_intake
from src.agents.qa_scoring import run_qa_scoring
from src.agents.report import compile_report, persist_report
from src.agents.summarization import run_summarization
from src.agents.transcription import run_transcription
from src.graph import edges
from src.graph.state import CallStatus, PipelineState, TranscriptionSegment
from src.security.audit import AuditLogger
from src.security.injection_detector import detect_injection
from src.security.pii_redactor import redact_pii
from src.utils.config import Config
from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - langsmith is optional per the spec

    def traceable(func: Callable[..., Any]) -> Callable[..., Any]:
        return func


GENERIC_ERROR = "Processing failed for an unknown reason."


def isolate(name: str) -> Callable[[Callable[..., dict]], Callable[..., dict]]:
    """Per-node error isolation: a node that raises records the reason in state and
    the graph routes to the error terminal. Without this one bad node takes the
    whole pipeline down and the user learns nothing about which stage failed.

    This is the ONE place a broad except is correct (STYLE_GUIDE 12 allows what you
    can handle): the handler converts any failure into a typed pipeline outcome
    rather than swallowing it - the reason is preserved and surfaced."""

    def decorate(func: Callable[..., dict]) -> Callable[..., dict]:
        def wrapper(state: PipelineState, *args: Any, **kwargs: Any) -> dict:
            try:
                return func(state, *args, **kwargs)
            except Exception as exc:
                logger.exception("node %s failed", name)
                return {"error": f"{name}: {exc}", "status": CallStatus.FAILED.value}

        wrapper.__name__ = func.__name__
        return wrapper

    return decorate


def _audit(audit: AuditLogger | None, call_id: str, action: str, **details: Any) -> None:
    if audit is None or not call_id:
        return
    audit.log(call_id, action, details=details or None)


def make_intake_step(audit: AuditLogger | None = None) -> Callable[[PipelineState], dict]:
    @traceable
    @isolate("intake")
    def intake_step(state: PipelineState) -> dict:
        audio_input = state["audio_input"]
        result = run_intake(audio_input)
        _audit(
            audit,
            result.call_id,
            "intake_passed" if result.validation_passed else "intake_rejected",
            filename=audio_input.filename,
            reason=result.validation_error,
        )
        if not result.validation_passed:
            return {
                "intake": result,
                "error": result.validation_error,
                "status": CallStatus.FAILED.value,
            }
        return {"intake": result, "status": "in_progress"}

    return intake_step


def make_transcription_node(
    config: Config, engine: Engine, audit: AuditLogger | None = None
) -> Callable[[PipelineState], dict]:
    @traceable
    @isolate("transcription")
    def transcription_node(state: PipelineState) -> dict:
        intake = state["intake"]
        result = run_transcription(
            intake,
            engine,
            model_size=config.whisper_model_size,
            confidence_threshold=config.confidence_threshold,
            low_confidence_halt_ratio=config.low_confidence_halt_ratio,
        )
        _audit(
            audit,
            result.call_id,
            "transcribed",
            cached=result.cached,
            segments=len(result.segments),
        )
        return {"transcription": result}

    return transcription_node


def make_injection_check_node(audit: AuditLogger | None = None) -> Callable[[PipelineState], dict]:
    @traceable
    @isolate("injection_check")
    def injection_check_node(state: PipelineState) -> dict:
        transcription = state["transcription"]
        result = detect_injection(transcription.full_text)
        if result.injection_detected:
            # Return early and flag. Nothing downstream of here is an LLM call for
            # this call_id, which is the entire point of the stage.
            _audit(
                audit,
                transcription.call_id,
                "injection_detected",
                patterns=list(result.matched_patterns),
            )
            return {
                "status": CallStatus.FLAGGED_FOR_REVIEW.value,
                "error": (
                    "Prompt injection detected in transcript; blocked before any LLM call. "
                    f"Matched patterns: {', '.join(result.matched_patterns)}"
                ),
            }
        _audit(audit, transcription.call_id, "injection_check_passed")
        return {"status": "in_progress"}

    return injection_check_node


def make_pii_redaction_node(audit: AuditLogger | None = None) -> Callable[[PipelineState], dict]:
    @traceable
    @isolate("pii_redaction")
    def pii_redaction_node(state: PipelineState) -> dict:
        transcription = state["transcription"]
        full = redact_pii(transcription.full_text)
        segments: list[TranscriptionSegment] = []
        types: set[str] = set(full.pii_types)
        for seg in transcription.segments:
            red = redact_pii(seg.text)
            types.update(red.pii_types)
            segments.append(seg.model_copy(update={"text": red.redacted_text}))
        # Both the joined text and each segment: the UI renders segments while the
        # LLM reads full_text, so redacting one and not the other leaks through
        # whichever path was missed.
        redacted = transcription.model_copy(
            update={"full_text": full.redacted_text, "segments": segments}
        )
        _audit(
            audit,
            transcription.call_id,
            "pii_redacted",
            pii_found=bool(types),
            types=sorted(types),
        )
        return {"transcription": redacted}

    return pii_redaction_node


def make_summarize_and_qa_node(
    config: Config, audit: AuditLogger | None = None
) -> Callable[[PipelineState], dict]:
    @traceable
    @isolate("summarize_and_qa")
    def summarize_and_qa_node(state: PipelineState) -> dict:
        transcription = state["transcription"]
        llm = get_llm(
            provider=config.llm_provider,
            timeout=config.llm_timeout_seconds,
            api_key=_key_for(config),
        )
        summary = run_summarization(transcription, llm, max_retries=config.max_retries_per_node)
        # Summary first, then QA with the summary as context - QA judges resolution
        # and compliance against what the call was actually for.
        qa = run_qa_scoring(transcription, summary, llm, max_retries=config.max_retries_per_node)
        _audit(audit, transcription.call_id, "analyzed", overall_score=qa.overall_score)
        return {"summary": summary, "qa_scores": qa}

    return summarize_and_qa_node


def _key_for(config: Config) -> str | None:
    return {
        "openai": config.openai_api_key,
        "gemini": config.google_api_key,
        "groq": config.groq_api_key,
    }.get(config.llm_provider)


def make_report_node(
    engine: Engine, audit: AuditLogger | None = None, status: CallStatus = CallStatus.COMPLETED
) -> Callable[[PipelineState], dict]:
    @traceable
    @isolate("report")
    def report_node(state: PipelineState) -> dict:
        report = compile_report(
            state["intake"],
            state["transcription"],
            state["summary"],
            state["qa_scores"],
            status=status,
        )
        persist_report(report, engine)
        # Supervisor review is not a successful close. Logging "completed" made the
        # audit trail contradict CallRecord.status and the PDF header.
        action = "completed" if status is CallStatus.COMPLETED else status.value
        _audit(audit, report.call_id, action, status=status.value)
        return {"report": report, "status": status.value}

    return report_node


def make_supervisor_review_node(
    engine: Engine, audit: AuditLogger | None = None
) -> Callable[[PipelineState], dict]:
    """Same report, different terminal status. A critical compliance flag still
    produces the full artifact - a supervisor needs the evidence to review."""
    inner = make_report_node(engine, audit, status=CallStatus.FLAGGED_FOR_REVIEW)

    @traceable
    def supervisor_review_node(state: PipelineState) -> dict:
        return inner(state)

    return supervisor_review_node


def make_error_node(audit: AuditLogger | None = None) -> Callable[[PipelineState], dict]:
    @traceable
    def error_node(state: PipelineState) -> dict:
        # Three-level fallback. Without it an intake failure surfaces as a bare
        # "Validation failed" and the user never learns the file was 61 minutes long.
        intake = state.get("intake")
        message = (
            state.get("error")
            or (intake.validation_error if intake is not None else None)
            or GENERIC_ERROR
        )
        call_id = intake.call_id if intake is not None else ""
        status = state.get("status")
        final = (
            CallStatus.FLAGGED_FOR_REVIEW.value
            if status == CallStatus.FLAGGED_FOR_REVIEW.value
            else CallStatus.FAILED.value
        )
        _audit(audit, call_id, "pipeline_failed", reason=message)
        return {"error": message, "status": final}

    return error_node


# ---- graph wiring ------------------------------------------------------------

# Graph node names. These are the strings the routing functions in edges.py return,
# so the two files cannot drift without a KeyError at compile time rather than a
# silent misroute at runtime.
INTAKE = "intake"
TRANSCRIBE = edges.TRANSCRIBE
INJECTION_CHECK = "injection_check"
REDACT = edges.REDACT
SUMMARIZE_AND_QA = "summarize_and_qa"
REPORT = edges.REPORT
SUPERVISOR_REVIEW = edges.SUPERVISOR_REVIEW
ERROR = edges.ERROR

STAGES: tuple[str, ...] = (
    INTAKE,
    TRANSCRIBE,
    INJECTION_CHECK,
    REDACT,
    SUMMARIZE_AND_QA,
    REPORT,
    SUPERVISOR_REVIEW,
    ERROR,
)

# Compiled graphs are immutable and reusable; building one costs real time and it
# does not depend on the call. Cached by engine id so app.py compiles once at
# startup and every request reuses it. Keyed by id(engine), not URL - see
# database/connection.py for why.
_compiled: dict[int, object] = {}


def build_workflow(
    config: Config, db_engine: Engine, audit: AuditLogger | None = None
) -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node(INTAKE, make_intake_step(audit))
    graph.add_node(TRANSCRIBE, make_transcription_node(config, db_engine, audit))
    graph.add_node(INJECTION_CHECK, make_injection_check_node(audit))
    graph.add_node(REDACT, make_pii_redaction_node(audit))
    graph.add_node(SUMMARIZE_AND_QA, make_summarize_and_qa_node(config, audit))
    graph.add_node(REPORT, make_report_node(db_engine, audit))
    graph.add_node(SUPERVISOR_REVIEW, make_supervisor_review_node(db_engine, audit))
    graph.add_node(ERROR, make_error_node(audit))

    graph.set_entry_point(INTAKE)

    graph.add_conditional_edges(
        INTAKE, edges.route_after_intake, {edges.TRANSCRIBE: TRANSCRIBE, edges.ERROR: ERROR}
    )
    # Unconditional: transcription failure is raised inside the node and lands in
    # state['error'], so there is no second destination to choose between here.
    graph.add_edge(TRANSCRIBE, INJECTION_CHECK)
    graph.add_conditional_edges(
        INJECTION_CHECK,
        edges.route_after_injection_check,
        {edges.REDACT: REDACT, edges.ERROR: ERROR},
    )
    graph.add_edge(REDACT, SUMMARIZE_AND_QA)
    graph.add_conditional_edges(
        SUMMARIZE_AND_QA,
        edges.route_after_qa,
        {
            edges.REPORT: REPORT,
            edges.SUPERVISOR_REVIEW: SUPERVISOR_REVIEW,
            edges.ERROR: ERROR,
        },
    )

    for terminal in (REPORT, SUPERVISOR_REVIEW, ERROR):
        graph.add_edge(terminal, END)

    return graph


def compile_workflow(
    config: Config, db_engine: Engine, audit: AuditLogger | None = None, use_cache: bool = True
):
    """Returns the compiled graph, building it at most once per engine."""
    key = id(db_engine)
    if use_cache and key in _compiled:
        return _compiled[key]
    compiled = build_workflow(config, db_engine, audit).compile()
    if use_cache:
        _compiled[key] = compiled
    logger.info("workflow compiled with %d stages", len(STAGES))
    return compiled


def reset_workflow_cache() -> None:
    """Tests only: drop compiled graphs so a fresh engine or config takes effect."""
    _compiled.clear()
