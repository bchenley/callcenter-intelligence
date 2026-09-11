"""Audio format detection, validation and property extraction.

GOAL: decide, before anything expensive happens, whether these bytes are audio we can process.

PROBLEM without it: a .wav file containing MP3 data (or a renamed .txt) reaches faster-whisper
and fails with a decoder traceback five stages downstream. The user sees a stack trace instead
of "unsupported format". Worse, a 400MB upload gets written to a temp file first.

CONCEPT: function = a gate that answers yes/no with a reason, using only the bytes.
         form   = three pure functions over `bytes`, plus a ValidationResult dataclass.

The rule that makes this a security control and not just a convenience: the format is read
from the CONTENT (the first 12 bytes), never from the filename. Filenames are attacker-supplied.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB
MAX_DURATION_SECONDS = 3600              # 60 minutes
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "m4a"}


class AudioValidationError(Exception):
    """Raised when a file claims to be audio but cannot be parsed."""


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
    """Identify the container from magic bytes. Returns None if nothing matches.

    Only the first 12 bytes are consulted — that is all four of these formats need, and
    reading no further means a 400MB file costs the same as a 400-byte one.

      wav  : b"RIFF" at 0..3  AND  b"WAVE" at 8..11   (bytes 4..7 are the chunk size)
      flac : b"fLaC" at 0..3
      m4a  : b"ftyp" at 4..7                          (ISO-BMFF box: size, then type)
      mp3  : b"ID3"  at 0..2  (ID3v2 tag)  OR  a frame sync: 0xFF then top 3 bits of
             the next byte set (0xE0 mask). The sync is why b"\\xff\\xfb..." is an mp3
             even with no ID3 tag at all.
    """
    if len(data) < 12:
        return None

    header = data[:12]

    if header[0:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header[0:4] == b"fLaC":
        return "flac"
    if header[4:8] == b"ftyp":
        return "m4a"
    if header[0:3] == b"ID3":
        return "mp3"
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "mp3"

    return None


def validate_audio_file(data: bytes, filename: str) -> ValidationResult:
    """Gate an upload. Order of checks is deliberate — see below.

    empty -> size -> format. Size is checked before format because format detection on a
    huge buffer is pointless work, and because "file exceeds maximum" is the more useful
    message for an oversized file whose header happens to be unrecognisable too.

    NOTE: duration is NOT checked here. A WAV's duration has to come from its RIFF header,
    which the intake agent reads before this size gate so an oversized long WAV reports
    "duration exceeds 60 minutes" rather than the misleading "file too large".
    """
    if not data:
        return ValidationResult(is_valid=False, error=f"Empty file: {filename}")

    if len(data) > MAX_FILE_SIZE_BYTES:
        mb = len(data) / (1024 * 1024)
        return ValidationResult(
            is_valid=False,
            error=f"File exceeds maximum size of 50 MB (got {mb:.1f} MB)",
        )

    detected = detect_audio_format(data)
    if detected is None or detected not in SUPPORTED_FORMATS:
        return ValidationResult(
            is_valid=False,
            error=(
                f"Unsupported audio format for {filename}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            ),
        )

    return ValidationResult(is_valid=True, detected_format=detected)


def extract_audio_properties(data: bytes, audio_format: str) -> AudioProps:
    """Duration, sample rate and channel count, without decoding the audio.

    WAV goes through the stdlib `wave` module: the RIFF header already carries frame count,
    rate and channels, so duration = n_frames / framerate is exact and costs no decoding.

    Everything else goes through mutagen, which parses container metadata only — again no
    decode. mutagen is imported lazily so that WAV-only paths (and the unit tests) do not
    require it to be installed.
    """
    if audio_format == "wav":
        try:
            with wave.open(io.BytesIO(data), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                channels = handle.getnchannels()
        except (wave.Error, EOFError) as exc:
            raise AudioValidationError(f"Corrupt or unreadable WAV file: {exc}") from exc

        if rate <= 0:
            raise AudioValidationError("WAV header reports a sample rate of zero")

        return AudioProps(
            duration_seconds=frames / float(rate),
            sample_rate=rate,
            channels=channels,
        )

    try:
        from mutagen.flac import FLAC
        from mutagen.mp3 import MP3
        from mutagen.mp4 import MP4
    except ImportError as exc:  # pragma: no cover
        raise AudioValidationError(f"mutagen is required for {audio_format} files") from exc

    parser = {"mp3": MP3, "flac": FLAC, "m4a": MP4}[audio_format]

    try:
        meta = parser(io.BytesIO(data))
        info = meta.info
    except Exception as exc:
        raise AudioValidationError(f"Corrupt or unreadable {audio_format} file: {exc}") from exc

    return AudioProps(
        duration_seconds=float(getattr(info, "length", 0.0)),
        sample_rate=int(getattr(info, "sample_rate", 0)),
        # MP4/M4A exposes `channels`; MP3 exposes `channels` too but older mutagen
        # versions use `mode` — default to 1 rather than crashing on a missing attr.
        channels=int(getattr(info, "channels", 1)),
    )
