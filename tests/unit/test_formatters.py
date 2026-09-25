# callcenter-intelligence
# tests/unit/test_formatters.py

from __future__ import annotations

import pytest

from src.graph.state import (
    ActionItem,
    ComplianceFlag,
    Entity,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    Severity,
    SummaryResult,
)
from src.utils.formatters import NO_FLAGS_TEXT, format_qa, format_summary, secs_to_mmss


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00"),
        (90.0, "01:30"),
        (59.9, "00:59"),
        (600.0, "10:00"),
        (3599.0, "59:59"),
        (3600.0, "60:00"),
    ],
)
def test_secs_to_mmss(seconds: float, expected: str) -> None:
    assert secs_to_mmss(seconds) == expected


def test_negative_seconds_raises() -> None:
    with pytest.raises(ValueError):
        secs_to_mmss(-1.0)


def _qa(flags: list[ComplianceFlag] | None = None, score: int = 3) -> QAScoreResult:
    dim = QADimensionScore(score=score, justification="At 01:15 the agent confirmed the charge.")
    return QAScoreResult(
        call_id="c1",
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        compliance_flags=flags or [],
        overall_score=float(score),
    )


def test_format_qa_no_flags_text() -> None:
    assert NO_FLAGS_TEXT in format_qa(_qa())
    assert NO_FLAGS_TEXT == "No compliance issues detected."


def test_format_qa_lists_all_five_dimensions() -> None:
    out = format_qa(_qa())
    for label in (
        "Professionalism",
        "Empathy",
        "Problem Resolution",
        "Compliance",
        "Communication Clarity",
    ):
        assert label in out


def test_format_qa_shows_overall_score() -> None:
    assert "3.00" in format_qa(_qa())


def test_format_qa_renders_flags_with_severity_and_timestamp() -> None:
    flag = ComplianceFlag(
        description="Account accessed before identity verification",
        severity=Severity.CRITICAL,
        timestamp_reference="02:15-02:45",
    )
    out = format_qa(_qa(flags=[flag]))
    assert "[CRITICAL]" in out
    assert "02:15-02:45" in out
    assert NO_FLAGS_TEXT not in out


def test_format_qa_is_ascii_only() -> None:
    """The whole rendered card must survive an ASCII-only pipe."""
    flag = ComplianceFlag(description="Missing disclosure", severity=Severity.HIGH)
    format_qa(_qa(flags=[flag])).encode("ascii")


def test_format_qa_escapes_pipes_in_justification() -> None:
    dim = QADimensionScore(score=4, justification="said a | b at 00:10")
    qa = QAScoreResult(
        call_id="c",
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
    )
    table_rows = [ln for ln in format_qa(qa).splitlines() if ln.startswith("| Professionalism")]
    assert table_rows and table_rows[0].count("|") == 4, "a raw pipe would break the markdown table"


def _summary() -> SummaryResult:
    return SummaryResult(
        call_id="c1",
        call_purpose="Duplicate charge on the March invoice.",
        key_discussion_points=["Charged twice on 3 March", "Refund timeline explained"],
        action_items=[
            ActionItem(description="Process refund", owner="Agent", deadline="3 business days")
        ],
        resolution_status=ResolutionStatus.RESOLVED,
        sentiment_trajectory="frustrated at open, satisfied at close",
        entities=[Entity(name="March invoice", type="document")],
    )


def test_format_summary_contains_every_section() -> None:
    out = format_summary(_summary())
    for token in (
        "Duplicate charge",
        "Charged twice",
        "Process refund",
        "resolved",
        "frustrated",
        "March invoice",
    ):
        assert token in out


def test_format_summary_is_ascii_only() -> None:
    format_summary(_summary()).encode("ascii")


def test_format_summary_omits_empty_sections() -> None:
    bare = SummaryResult(call_id="c", call_purpose="Quick question about hours.")
    out = format_summary(bare)
    assert "Action items" not in out and "Entities" not in out
