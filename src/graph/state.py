# callcenter-intelligence
# src/graph/state.py

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, Field


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ESCALATED = "escalated"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    # Only CRITICAL diverts the graph to supervisor review; see route_after_qa.
    CRITICAL = "critical"


class CallStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    FLAGGED_FOR_REVIEW = "flagged_for_review"


class AudioInput(BaseModel):
    # Bytes, not a path: nothing upstream gets to choose where we write to disk.
    audio_data: bytes
    filename: str
    caller_id: str | None = None
    department: str | None = None
    timestamp: datetime | None = None


class AudioProperties(BaseModel):
    duration_seconds: float = Field(default=0.0, ge=0.0)
    sample_rate: int = Field(default=0, ge=0)
    channels: int = Field(default=0, ge=0)


class PIIScanResult(BaseModel):
    pii_detected: bool = False
    affected_fields: list[str] = Field(default_factory=list)
    pii_types: list[str] = Field(default_factory=list)


class IntakeResult(BaseModel):
    call_id: str
    validation_passed: bool
    validation_error: str | None = None
    audio_format: str | None = None
    temp_path: str | None = None
    properties: AudioProperties = Field(default_factory=AudioProperties)
    pii_scan: PIIScanResult = Field(default_factory=PIIScanResult)
    filename: str | None = None


class TranscriptionSegment(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    speaker: str = "Agent"
    # Derived from avg_logprob and no_speech_prob, both bounded; out of range means
    # the confidence formula is wrong and should fail here, not skew an average.
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptionResult(BaseModel):
    call_id: str
    full_text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    language: str = "en"
    duration_seconds: float = Field(default=0.0, ge=0.0)
    low_confidence: bool = False
    flagged_for_review: bool = False
    cached: bool = False
    model_size: str | None = None


class ActionItem(BaseModel):
    description: str
    owner: str
    deadline: str | None = None


class Entity(BaseModel):
    name: str
    type: str


class SummaryResult(BaseModel):
    call_id: str = ""
    call_purpose: str
    key_discussion_points: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    sentiment_trajectory: str = ""
    entities: list[Entity] = Field(default_factory=list)


class QADimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class ComplianceFlag(BaseModel):
    description: str
    severity: Severity
    timestamp_reference: str | None = None


class QAScoreResult(BaseModel):
    call_id: str = ""
    professionalism: QADimensionScore
    empathy: QADimensionScore
    problem_resolution: QADimensionScore
    compliance: QADimensionScore
    communication_clarity: QADimensionScore
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)
    # The LLM fills this and run_qa_scoring discards it, recomputing from
    # DIMENSION_WEIGHTS. Kept on the model so structured output has a slot and a
    # test can prove the recomputation overrode it.
    overall_score: float = Field(default=1.0, ge=1.0, le=5.0)


class CallReport(BaseModel):
    call_id: str
    status: CallStatus = CallStatus.COMPLETED
    filename: str | None = None
    transcript: TranscriptionResult | None = None
    summary: SummaryResult | None = None
    qa_scores: QAScoreResult | None = None
    processed_at: datetime | None = None
    processing_seconds: float | None = None
    error: str | None = None
    trace_id: str | None = None


class PipelineState(TypedDict, total=False):
    # TypedDict, not BaseModel: LangGraph merges partial dicts returned by nodes.
    # total=False because a node writes only the keys it changed. Validation lives
    # on the payloads above, not on the envelope.
    audio_input: AudioInput
    intake: IntakeResult
    transcription: TranscriptionResult
    summary: SummaryResult
    qa_scores: QAScoreResult
    report: CallReport
    error: str
    status: str
    # Monotonic clock stamped by intake, read by the report node. Subtracting
    # wall-clock timestamps would be wrong twice over: the report is built after
    # the pipeline ends, and the system clock can step mid-run.
    started_at: float


DIMENSION_WEIGHTS: dict[str, float] = {
    "professionalism": 0.15,
    "empathy": 0.20,
    "problem_resolution": 0.30,
    "compliance": 0.20,
    "communication_clarity": 0.15,
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, "dimension weights must sum to 1.0"
