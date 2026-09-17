# callcenter-intelligence
# tests/unit/test_summarization.py

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.summarization import (
    SummarizationError,
    backoff_seconds,
    format_transcript,
    run_summarization,
)
from src.graph.state import (
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def _transcript(call_id: str = "call-1") -> TranscriptionResult:
    return TranscriptionResult(
        call_id=call_id,
        full_text="Thank you for calling. My account was charged twice.",
        segments=[
            TranscriptionSegment(
                start=0.0, end=2.0, text="Thank you for calling.", speaker="Agent", confidence=0.9
            ),
            TranscriptionSegment(
                start=2.5,
                end=95.0,
                text="My account was charged twice.",
                speaker="Customer",
                confidence=0.9,
            ),
        ],
    )


def _summary(call_id: str = "") -> SummaryResult:
    return SummaryResult(
        call_id=call_id,
        call_purpose="Duplicate charge",
        resolution_status=ResolutionStatus.RESOLVED,
    )


def _llm(result=None, side_effect=None) -> MagicMock:
    structured = MagicMock()
    if side_effect is not None:
        structured.invoke.side_effect = side_effect
    else:
        structured.invoke.return_value = result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def test_returns_structured_result() -> None:
    out = run_summarization(_transcript(), _llm(_summary()), sleep=lambda _s: None)
    assert isinstance(out, SummaryResult)
    assert out.call_purpose == "Duplicate charge"


def test_call_id_overwritten_from_transcript() -> None:
    """The LLM does not know the call_id; whatever it echoes is discarded."""
    out = run_summarization(
        _transcript("call-real"), _llm(_summary("hallucinated")), sleep=lambda _s: None
    )
    assert out.call_id == "call-real"


def test_structured_output_requested_with_the_model() -> None:
    llm = _llm(_summary())
    run_summarization(_transcript(), llm, sleep=lambda _s: None)
    llm.with_structured_output.assert_called_once_with(SummaryResult)


def test_raises_after_exactly_three_attempts() -> None:
    llm = _llm(side_effect=Exception("provider down"))
    with pytest.raises(SummarizationError):
        run_summarization(_transcript(), llm, max_retries=3, sleep=lambda _s: None)
    assert llm.with_structured_output.return_value.invoke.call_count == 3


def test_succeeds_on_retry_after_transient_failure() -> None:
    llm = _llm(side_effect=[Exception("429"), _summary()])
    out = run_summarization(_transcript(), llm, max_retries=3, sleep=lambda _s: None)
    assert out.call_purpose == "Duplicate charge"


def test_error_message_names_the_cause() -> None:
    llm = _llm(side_effect=Exception("rate limited"))
    with pytest.raises(SummarizationError, match="rate limited"):
        run_summarization(_transcript(), llm, max_retries=2, sleep=lambda _s: None)


def test_backoff_is_exponential_and_capped() -> None:
    assert [backoff_seconds(i) for i in range(6)] == [1, 2, 4, 8, 10, 10]


def test_backoff_actually_slept_between_attempts() -> None:
    slept: list[int] = []
    llm = _llm(side_effect=Exception("boom"))
    with pytest.raises(SummarizationError):
        run_summarization(_transcript(), llm, max_retries=3, sleep=slept.append)
    assert slept == [1, 2], "no sleep after the final attempt"


def test_transcript_formatted_with_mmss_and_speaker() -> None:
    text = format_transcript(_transcript())
    assert "[00:00-00:02] Agent: Thank you for calling." in text
    assert "[00:02-01:35] Customer: My account was charged twice." in text
