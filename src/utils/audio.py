# callcenter-intelligence
# src/utils/audio.py

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from mutagen import MutagenError
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MAX_DURATION_SECONDS = 3600
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "m4a"}

_PARSERS = {"mp3": MP3, "flac": FLAC, "m4a": MP4}


class AudioValidationError(Exception):
    pass


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error: str | None = None
    detected_format: str | None = None


@dataclass(frozen=True)
class AudioProps:
    duration_seconds: float
    sample_rate: int
    channels: int


def detect_audio_format(data: bytes) -> str | None:
    # Format comes from content, never the filename: filenames are caller-supplied.
    # 12 bytes is the widest header any supported format needs (WAV's RIFF....WAVE).
    if len(data) < 12:
        return None
    h = data[:12]
    if h[0:4] == b"RIFF" and h[8:12] == b"WAVE":
        return "wav"
    if h[0:4] == b"fLaC":
        return "flac"
    if h[4:8] == b"ftyp":
        return "m4a"
    if h[0:3] == b"ID3" or (h[0] == 0xFF and (h[1] & 0xE0) == 0xE0):
        return "mp3"
    return None


def validate_audio_file(data: bytes, filename: str) -> ValidationResult:
    # Size before format: detection on a huge buffer is wasted work, and "too large"
    # is the more useful message when an oversized file also has a bad header.
    # Duration is not checked here; intake reads it first (see run_intake).
    if not data:
        return ValidationResult(False, f"Empty file: {filename}")
    if len(data) > MAX_FILE_SIZE_BYTES:
        mb = len(data) / (1024 * 1024)
        return ValidationResult(False, f"File exceeds maximum size of 50 MB (got {mb:.1f} MB)")
    detected = detect_audio_format(data)
    if detected not in SUPPORTED_FORMATS:
        return ValidationResult(
            False,
            f"Unsupported audio format for {filename}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )
    return ValidationResult(True, detected_format=detected)


def extract_audio_properties(data: bytes, audio_format: str) -> AudioProps:
    # Both branches read container metadata only, so neither decodes audio.
    if audio_format == "wav":
        try:
            with wave.open(io.BytesIO(data), "rb") as handle:
                frames, rate, channels = (
                    handle.getnframes(),
                    handle.getframerate(),
                    handle.getnchannels(),
                )
        except (wave.Error, EOFError) as exc:
            raise AudioValidationError(f"Corrupt or unreadable WAV file: {exc}") from exc
        if rate <= 0:
            raise AudioValidationError("WAV header reports a sample rate of zero")
        return AudioProps(frames / float(rate), rate, channels)

    try:
        info = _PARSERS[audio_format](io.BytesIO(data)).info
    except (MutagenError, KeyError, ValueError) as exc:
        raise AudioValidationError(f"Corrupt or unreadable {audio_format} file: {exc}") from exc
    return AudioProps(
        float(getattr(info, "length", 0.0)),
        int(getattr(info, "sample_rate", 0)),
        int(getattr(info, "channels", 1)),
    )
