# callcenter-intelligence
# src/agents/intake.py

from __future__ import annotations

import io
import re
import tempfile
import uuid
import wave

from src.graph.state import AudioInput, AudioProperties, IntakeResult, PIIScanResult
from src.utils.audio import (
    MAX_DURATION_SECONDS,
    AudioValidationError,
    detect_audio_format,
    extract_audio_properties,
    validate_audio_file,
)

# Detection only; the transcript redactor in src/security replaces text and needs
# position handling this does not.
_METADATA_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")),
]

_EMPTY_AUDIO_PROPS = AudioProperties()
_EMPTY_PII = PIIScanResult()


def _make_failed_result(call_id: str, error: str, filename: str | None = None) -> IntakeResult:
    return IntakeResult(
        call_id=call_id,
        validation_passed=False,
        validation_error=error,
        filename=filename,
        properties=_EMPTY_AUDIO_PROPS,
        pii_scan=_EMPTY_PII,
    )


def _wav_duration_from_header(data: bytes) -> float | None:
    # Returns None rather than raising: an unreadable header is the validator's
    # news to deliver, and this runs before validation.
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / float(rate) if rate > 0 else None
    except (wave.Error, EOFError):
        return None


def _scan_metadata_for_pii(audio_input: AudioInput) -> PIIScanResult:
    # caller_id and department are agent-typed free text that reaches the database
    # before any transcript redaction stage exists.
    affected: list[str] = []
    kinds: set[str] = set()
    for field_name in ("caller_id", "department"):
        value = getattr(audio_input, field_name, None)
        if not value:
            continue
        for kind, pattern in _METADATA_PII_PATTERNS:
            if pattern.search(value):
                kinds.add(kind)
                if field_name not in affected:
                    affected.append(field_name)
    return PIIScanResult(
        pii_detected=bool(affected), affected_fields=affected, pii_types=sorted(kinds)
    )


def run_intake(audio_input: AudioInput) -> IntakeResult:
    # Returns a typed failure instead of raising: this is a graph node, and the
    # error node needs the reason to say something true to the user.
    call_id = str(uuid.uuid4())
    data, filename = audio_input.audio_data, audio_input.filename

    if not data:
        return _make_failed_result(call_id, f"Empty file: {filename}", filename)

    # Duration before the size gate. A 61-minute WAV is also over 50 MB, so order
    # decides which reason the user gets, and "too large" sends them to compress a
    # file that is the wrong length.
    if detect_audio_format(data) == "wav":
        duration = _wav_duration_from_header(data)
        if duration is not None and duration > MAX_DURATION_SECONDS:
            return _make_failed_result(
                call_id,
                f"Audio duration exceeds maximum of {MAX_DURATION_SECONDS // 60} minutes "
                f"(got {duration / 60:.1f} minutes)",
                filename,
            )

    validation = validate_audio_file(data, filename)
    if not validation.is_valid:
        return _make_failed_result(call_id, validation.error or "Validation failed", filename)
    audio_format = validation.detected_format

    try:
        props = extract_audio_properties(data, audio_format)
    except AudioValidationError as exc:
        return _make_failed_result(call_id, str(exc), filename)

    properties = AudioProperties(
        duration_seconds=props.duration_seconds,
        sample_rate=props.sample_rate,
        channels=props.channels,
    )
    # Non-WAV duration lands here because mutagen supplies it, not a header we can
    # read for free beforehand.
    if properties.duration_seconds > MAX_DURATION_SECONDS:
        return _make_failed_result(
            call_id,
            f"Audio duration exceeds maximum of {MAX_DURATION_SECONDS // 60} minutes "
            f"(got {properties.duration_seconds / 60:.1f} minutes)",
            filename,
        )

    # delete=False: transcription opens this by path in the next node. The rolling
    # cleanup in src/services/pipeline.py owns removal.
    with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as handle:
        handle.write(data)
        temp_path = handle.name

    return IntakeResult(
        call_id=call_id,
        validation_passed=True,
        audio_format=audio_format,
        temp_path=temp_path,
        properties=properties,
        pii_scan=_scan_metadata_for_pii(audio_input),
        filename=filename,
    )
