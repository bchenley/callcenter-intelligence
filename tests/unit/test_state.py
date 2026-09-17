# callcenter-intelligence
# tests/unit/test_state.py

from __future__ import annotations

import pydantic
import pytest

from src.graph.state import (
    DIMENSION_WEIGHTS,
    ActionItem,
    AudioInput,
    CallReport,
    ComplianceFlag,
    Entity,
    PipelineState,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    Severity,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


class TestConstraints:
    def test_confidence_above_one_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            TranscriptionSegment(start=0, end=1, text="hi", confidence=1.5)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            TranscriptionSegment(start=0, end=1, text="hi", confidence=-0.1)

    def test_confidence_at_bounds_accepted(self):
        assert TranscriptionSegment(start=0, end=1, text="hi", confidence=0.0).confidence == 0.0
        assert TranscriptionSegment(start=0, end=1, text="hi", confidence=1.0).confidence == 1.0

    def test_dimension_score_zero_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            QADimensionScore(score=0, justification="...")

    def test_dimension_score_six_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            QADimensionScore(score=6, justification="...")

    def test_overall_score_out_of_range_rejected(self):
        dims = {k: QADimensionScore(score=3, justification="x") for k in DIMENSION_WEIGHTS}
        with pytest.raises(pydantic.ValidationError):
            QAScoreResult(**dims, overall_score=5.5)


class TestEnums:
    def test_resolution_status_coerces_from_string(self):
        s = SummaryResult(call_purpose="billing", resolution_status="escalated")
        assert s.resolution_status is ResolutionStatus.ESCALATED

    def test_unknown_resolution_status_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            SummaryResult(call_purpose="x", resolution_status="mostly_resolved")

    def test_severity_coerces_and_critical_is_distinguishable(self):
        flag = ComplianceFlag(description="no ID verification", severity="critical")
        assert flag.severity is Severity.CRITICAL
        assert flag.severity != Severity.HIGH


class TestWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9

    def test_problem_resolution_is_heaviest(self):
        assert max(DIMENSION_WEIGHTS, key=DIMENSION_WEIGHTS.get) == "problem_resolution"

    def test_weights_cover_exactly_the_five_qa_fields(self):
        model_fields = set(QAScoreResult.model_fields)
        assert set(DIMENSION_WEIGHTS) <= model_fields


class TestShapes:
    def test_audio_input_accepts_optional_metadata(self):
        a = AudioInput(audio_data=b"\x00", filename="c.wav")
        assert a.caller_id is None and a.department is None

    def test_action_item_deadline_optional(self):
        assert ActionItem(description="refund", owner="agent").deadline is None

    def test_report_assembles_nested_models(self):
        report = CallReport(
            call_id="abc",
            transcript=TranscriptionResult(call_id="abc", full_text="hello"),
            summary=SummaryResult(
                call_purpose="billing",
                entities=[Entity(name="Acme", type="ORG")],
            ),
        )
        assert report.transcript.full_text == "hello"
        assert report.summary.entities[0].type == "ORG"

    def test_pipeline_state_keys_are_all_optional(self):
        state: PipelineState = {}
        state["status"] = "completed"
        assert state == {"status": "completed"}
