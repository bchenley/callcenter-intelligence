# callcenter-intelligence
# tests/integration/conftest.py

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.graph.state import (
    ActionItem,
    ComplianceFlag,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    Severity,
    SummaryResult,
)
from src.utils.config import Config


@dataclass
class FakeSeg:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.05


@dataclass
class FakeInfo:
    language: str = "en"
    duration: float = 12.0


DEFAULT_SEGMENTS = [
    FakeSeg(0.0, 2.0, "Thank you for calling Acme support, how can I help you today?"),
    FakeSeg(2.4, 6.0, "My account was charged twice this month and I need help with it."),
    FakeSeg(6.3, 9.0, "Let me check that for you right away."),
]


@pytest.fixture
def config(tmp_path) -> Config:
    """No real keys anywhere. The whole suite must run for a reviewer with none."""
    return Config(
        llm_provider="openai",
        openai_api_key=None,
        google_api_key=None,
        groq_api_key=None,
        llm_timeout_seconds=5,
        max_retries_per_node=2,
        whisper_model_size="tiny",
        confidence_threshold=0.6,
        low_confidence_halt_ratio=0.5,
        db_path=str(tmp_path / "calls.db"),
        db_encryption_key=None,
    )


def make_summary(call_id: str = "") -> SummaryResult:
    return SummaryResult(
        call_id=call_id,
        call_purpose="Customer was charged twice for the March invoice.",
        key_discussion_points=["Duplicate charge on 3 March", "Refund timeline explained"],
        action_items=[ActionItem(description="Process refund", owner="Agent", deadline="3 days")],
        resolution_status=ResolutionStatus.RESOLVED,
        sentiment_trajectory="frustrated at open, satisfied at close",
    )


def make_qa(severity: Severity | None = None, score: int = 4) -> QAScoreResult:
    dim = QADimensionScore(score=score, justification="At 02:15 the agent confirmed the charge.")
    flags = (
        [
            ComplianceFlag(
                description="Account accessed before verification",
                severity=severity,
                timestamp_reference="02:15-02:45",
            )
        ]
        if severity is not None
        else []
    )
    return QAScoreResult(
        call_id="",
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        compliance_flags=flags,
        overall_score=1.0,
    )


def fake_llm(summary: SummaryResult | None = None, qa: QAScoreResult | None = None) -> MagicMock:
    """Dispatches on the schema requested, the way a real structured-output call
    does, so one mock serves both agents in a single pipeline run."""
    summary = summary if summary is not None else make_summary()
    qa = qa if qa is not None else make_qa()

    def with_structured_output(schema):
        stub = MagicMock()
        stub.invoke.return_value = summary if schema is SummaryResult else qa
        return stub

    llm = MagicMock()
    llm.with_structured_output.side_effect = with_structured_output
    return llm


def fake_whisper(segments=None) -> MagicMock:
    model = MagicMock()
    model.transcribe.return_value = (iter(segments or DEFAULT_SEGMENTS), FakeInfo())
    return model
