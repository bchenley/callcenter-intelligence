# callcenter-intelligence
# src/agents/transcription.py

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select

from src.database.connection import session_scope
from src.database.models import TranscriptionCache
from src.graph.state import IntakeResult, TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

# Module-level singleton. Loading a Whisper model costs 5-30 seconds; doing it per
# request is the single biggest avoidable latency in the pipeline. app.py warms
# this at startup so the first real call does not pay for it.
_model: Any = None
_model_size: str | None = None

HASH_CHUNK_BYTES = 8192
SPEAKER_GAP_SECONDS = 1.2
SHORT_AFFIRMATION_WORDS = 3
LONG_SEGMENT_WORDS = 8

_AGENT_PATTERNS = re.compile(
    r"\b(?:thank you for calling|how (?:can|may) i (?:help|assist)|my name is \w+ and i"
    r"|let me (?:check|look|pull up|transfer)|one moment|one second"
    r"|for (?:security|verification) purposes"
    r"|is there anything else|i apologize for|i'?m sorry(?: about|,? i didn'?t catch)"
    r"|i can (?:help|assist) you with|(?:may|can) i (?:have|ask)"
    r"|before i (?:pull|look|verify|check)|that verifies"
    r"|i(?:'?m| am) reversing|happy to help|i can see)\b",
    re.IGNORECASE,
)
_CUSTOMER_PATTERNS = re.compile(
    r"\b(?:i need help|i'?m having (?:a |an )?(?:problem|issue|trouble)|my (?:account|bill|order|card)"
    r"|i was charged|it (?:doesn'?t|does not) work|can you (?:help|tell) me|i want to"
    r"|i'?d like to (?:cancel|return|change))\b",
    re.IGNORECASE,
)

# Whisper hallucinates boilerplate on silence and low-quality audio. BLANK_AUDIO
# labels, repeated dots, YouTube footers, non-speech tags and repeated phrases
# are removed rather than trusted as speech.
_BLANK_AUDIO = re.compile(r"\[?\s*BLANK_AUDIO\s*\]?", re.IGNORECASE)
_REPEATED_DOTS = re.compile(r"\.{4,}")
_YOUTUBE_FOOTER = re.compile(
    r"\b(?:thanks? (?:for|to) watching|please subscribe|like and subscribe|see you next time)\b[.!]?",
    re.IGNORECASE,
)
_NON_SPEECH_LABEL = re.compile(
    r"[\[\(]\s*(?:music|applause|laughter|silence|noise|inaudible|foreign)\s*[\]\)]", re.IGNORECASE
)
_REPEATED_PHRASE = re.compile(r"\b(\w+(?:\s+\w+){0,3})(?:[,\s]+\1\b)+", re.IGNORECASE)


class TranscriptionError(RuntimeError):
    pass


def _resolve_device() -> tuple[str, str]:
    """CUDA when present, else CPU with int8. Apple MPS is deliberately treated as
    CPU: CTranslate2 has no MPS backend, so claiming it would fail at load time."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        logger.info("torch not installed; using CPU int8")
    return "cpu", "int8"


def _get_whisper_model(model_size: str = "base") -> Any:
    global _model, _model_size
    if _model is not None and _model_size == model_size:
        return _model
    from faster_whisper import WhisperModel

    device, compute_type = _resolve_device()
    logger.info(
        "loading whisper model=%s device=%s compute_type=%s", model_size, device, compute_type
    )
    _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_size = model_size
    return _model


def reset_model_cache() -> None:
    """Tests only: drops the singleton so a fresh mock can be installed."""
    global _model, _model_size
    _model = None
    _model_size = None


def _clean_transcript_text(text: str) -> str:
    text = _BLANK_AUDIO.sub(" ", text)
    text = _NON_SPEECH_LABEL.sub(" ", text)
    text = _YOUTUBE_FOOTER.sub(" ", text)
    text = _REPEATED_DOTS.sub("...", text)
    text = _REPEATED_PHRASE.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _segment_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    """Two independent signals: how sure the decoder was of the tokens, and how
    sure the VAD was that this was speech at all. Weighted 0.7 logprob / 0.3 speech probability."""
    logprob_conf = max(0.0, min(1.0, 1.0 + avg_logprob))
    speech_conf = 1.0 - no_speech_prob
    return round(logprob_conf * 0.7 + speech_conf * 0.3, 4)


class SpeakerDiarizer:
    """Heuristic two-speaker labelling. Signals are applied in priority order, and
    the first that fires decides - content evidence beats timing evidence, because
    a phrase like "thank you for calling" identifies the speaker outright while a
    pause only suggests a handover."""

    def __init__(self, gap_seconds: float = SPEAKER_GAP_SECONDS) -> None:
        self.gap_seconds = gap_seconds

    @staticmethod
    def _other(speaker: str) -> str:
        return "Customer" if speaker == "Agent" else "Agent"

    def assign(self, segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
        if not segments:
            return segments
        # First segment defaults to Agent: the agent opens a call center call.
        current = "Agent"
        segments[0].speaker = current
        for prev, seg in itertools.pairwise(segments):
            current = self._next_speaker(prev, seg, current)
            seg.speaker = current
        return segments

    def _next_speaker(
        self, prev: TranscriptionSegment, seg: TranscriptionSegment, current: str
    ) -> str:
        # 1. Content patterns - strongest evidence, decides outright.
        if _AGENT_PATTERNS.search(seg.text):
            return "Agent"
        if _CUSTOMER_PATTERNS.search(seg.text):
            return "Customer"
        # 2. Gap-based switching.
        if seg.start - prev.end > self.gap_seconds:
            return self._other(current)
        # 3. Question-answer switching.
        if prev.text.rstrip().endswith("?"):
            return self._other(current)
        # 4. Short affirmation following a long segment ("mm-hm", "yes, exactly").
        if (
            len(seg.text.split()) <= SHORT_AFFIRMATION_WORDS
            and len(prev.text.split()) >= LONG_SEGMENT_WORDS
        ):
            return self._other(current)
        return current


def compute_audio_hash(path: str | Path) -> str:
    """SHA-256 over the file in 8 KB chunks - a 50 MB upload should not be held in
    memory twice just to be fingerprinted."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def check_cache(engine: Engine, audio_hash: str) -> dict[str, Any] | None:
    with session_scope(engine) as session:
        row = session.scalar(
            select(TranscriptionCache).where(TranscriptionCache.audio_hash == audio_hash)
        )
        return json.loads(row.transcription_json) if row else None


def save_cache(engine: Engine, audio_hash: str, result: TranscriptionResult) -> None:
    with session_scope(engine) as session:
        session.add(
            TranscriptionCache(audio_hash=audio_hash, transcription_json=result.model_dump_json())
        )


def run_transcription(
    intake: IntakeResult,
    engine: Engine,
    model_size: str = "base",
    confidence_threshold: float = 0.6,
    low_confidence_halt_ratio: float = 0.5,
) -> TranscriptionResult:
    if not intake.validation_passed or not intake.temp_path:
        raise TranscriptionError(f"cannot transcribe a failed intake: {intake.validation_error}")

    audio_hash = compute_audio_hash(intake.temp_path)
    cached = check_cache(engine, audio_hash)
    if cached is not None:
        # Same audio, different call: everything is reusable except the identity.
        result = TranscriptionResult.model_validate(cached)
        return result.model_copy(update={"call_id": intake.call_id, "cached": True})

    model = _get_whisper_model(model_size)
    segments_iter, info = model.transcribe(
        intake.temp_path,
        beam_size=1,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    segments: list[TranscriptionSegment] = []
    for raw in segments_iter:
        text = _clean_transcript_text(raw.text)
        if not text:
            continue
        segments.append(
            TranscriptionSegment(
                start=float(raw.start),
                end=float(raw.end),
                text=text,
                confidence=_segment_confidence(float(raw.avg_logprob), float(raw.no_speech_prob)),
            )
        )

    segments = SpeakerDiarizer().assign(segments)
    low_conf = [s for s in segments if s.confidence < confidence_threshold]
    low_ratio = len(low_conf) / len(segments) if segments else 0.0

    result = TranscriptionResult(
        call_id=intake.call_id,
        full_text=" ".join(s.text for s in segments),
        segments=segments,
        language=getattr(info, "language", "en") or "en",
        duration_seconds=float(getattr(info, "duration", 0.0) or 0.0),
        low_confidence=bool(low_conf),
        flagged_for_review=low_ratio >= low_confidence_halt_ratio and bool(segments),
        cached=False,
        model_size=model_size,
    )
    save_cache(engine, audio_hash, result)
    return result
