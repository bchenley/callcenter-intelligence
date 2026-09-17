# callcenter-intelligence
# tests/unit/test_qa_scoring.py

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.qa_scoring import QAScoringError, compute_overall_score, run_qa_scoring
from src.graph.state import (
    DIMENSION_WEIGHTS,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def _transcript() -> TranscriptionResult:
    return TranscriptionResult(
        call_id="call-1",
        full_text="hello",
        segments=[
            TranscriptionSegment(start=0.0, end=2.0, text="hello", speaker="Agent", confidence=0.9)
        ],
    )


def _summary() -> SummaryResult:
    return SummaryResult(
        call_id="call-1", call_purpose="Billing", resolution_status=ResolutionStatus.RESOLVED
    )


def _qa(scores: dict[str, int], llm_overall: float) -> QAScoreResult:
    return QAScoreResult(
        call_id="whatever",
        professionalism=QADimensionScore(score=scores["professionalism"], justification="at 00:10"),
        empathy=QADimensionScore(score=scores["empathy"], justification="at 00:20"),
        problem_resolution=QADimensionScore(
            score=scores["problem_resolution"], justification="at 00:30"
        ),
        compliance=QADimensionScore(score=scores["compliance"], justification="at 00:40"),
        communication_clarity=QADimensionScore(
            score=scores["communication_clarity"], justification="at 00:50"
        ),
        overall_score=llm_overall,
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


ALL_FIVES = dict.fromkeys(DIMENSION_WEIGHTS, 5)
ALL_THREES = dict.fromkeys(DIMENSION_WEIGHTS, 3)


def test_llm_overall_score_is_discarded_and_recomputed() -> None:
    """The spec's headline check: five 5s with the LLM claiming 3.0 must return 5.0."""
    out = run_qa_scoring(
        _transcript(), _summary(), _llm(_qa(ALL_FIVES, llm_overall=3.0)), sleep=lambda _s: None
    )
    assert out.overall_score == 5.0


def test_recomputation_also_corrects_upward_claims() -> None:
    out = run_qa_scoring(
        _transcript(), _summary(), _llm(_qa(ALL_THREES, llm_overall=5.0)), sleep=lambda _s: None
    )
    assert out.overall_score == 3.0


def test_weights_match_the_spec() -> None:
    assert DIMENSION_WEIGHTS == {
        "professionalism": 0.15,
        "empathy": 0.20,
        "problem_resolution": 0.30,
        "compliance": 0.20,
        "communication_clarity": 0.15,
    }
    assert sum(DIMENSION_WEIGHTS.values()) == pytest.approx(1.0)


def test_weighting_is_not_a_flat_average() -> None:
    """Problem Resolution carries 30%, so moving it must move the total more than
    moving Professionalism by the same amount."""
    base = compute_overall_score(_qa(ALL_THREES, 3.0))
    res_up = compute_overall_score(_qa({**ALL_THREES, "problem_resolution": 5}, 3.0))
    prof_up = compute_overall_score(_qa({**ALL_THREES, "professionalism": 5}, 3.0))
    assert res_up - base == pytest.approx(0.60)
    assert prof_up - base == pytest.approx(0.30)


def test_overall_score_within_model_bounds() -> None:
    for scores in (dict.fromkeys(DIMENSION_WEIGHTS, 1), ALL_FIVES):
        assert 1.0 <= compute_overall_score(_qa(scores, 3.0)) <= 5.0


def test_call_id_taken_from_transcript() -> None:
    out = run_qa_scoring(
        _transcript(), _summary(), _llm(_qa(ALL_THREES, 3.0)), sleep=lambda _s: None
    )
    assert out.call_id == "call-1"


def test_summary_is_passed_to_the_model_as_context() -> None:
    llm = _llm(_qa(ALL_THREES, 3.0))
    run_qa_scoring(_transcript(), _summary(), llm, sleep=lambda _s: None)
    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    human = next(content for role, content in messages if role == "human")
    assert "Billing" in human, "the QA agent must see the summary"
    assert "Transcript:" in human


def test_system_prompt_carries_the_grounding_rules() -> None:
    llm = _llm(_qa(ALL_THREES, 3.0))
    run_qa_scoring(_transcript(), _summary(), llm, sleep=lambda _s: None)
    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    system = next(content for role, content in messages if role == "system")
    assert "3 is the baseline" in system
    assert "MM:SS" in system
    assert "Short calls are efficient" in system
    assert "never style" in system or "Never flag style" in system
    assert "next-time action" in system


def test_raises_after_three_attempts() -> None:
    llm = _llm(side_effect=Exception("provider down"))
    with pytest.raises(QAScoringError):
        run_qa_scoring(_transcript(), _summary(), llm, max_retries=3, sleep=lambda _s: None)
    assert llm.with_structured_output.return_value.invoke.call_count == 3


def test_backoff_slept_between_attempts() -> None:
    slept: list[int] = []
    llm = _llm(side_effect=Exception("boom"))
    with pytest.raises(QAScoringError):
        run_qa_scoring(_transcript(), _summary(), llm, max_retries=3, sleep=slept.append)
    assert slept == [1, 2]
