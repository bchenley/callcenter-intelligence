"""Shared fixtures and audio factories.

make_wav_bytes() is the workhorse of this whole test suite: it builds a real, parseable WAV
in memory so no test ever needs a fixture file on disk or a network call.
"""

from __future__ import annotations

import io
import wave

import pytest


def make_wav_bytes(duration: float = 1.0, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """A silent but structurally valid WAV of the requested duration.

    Silence is fine — every check in this milestone reads the RIFF header, not the samples.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)  # 16-bit
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(duration * sample_rate) * channels)
    return buffer.getvalue()


def make_mp3_bytes(payload_len: int = 100) -> bytes:
    """An MP3 frame sync header. Not decodable audio — enough to exercise magic-byte detection."""
    return b"\xff\xfb\x90\x00" + b"\x00" * payload_len


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()
