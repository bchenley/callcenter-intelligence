# callcenter-intelligence
# tests/conftest.py

from __future__ import annotations

import io
import struct
import wave

import pytest


def make_wav_bytes(duration: float = 1.0, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(duration * sample_rate) * channels)
    return buffer.getvalue()


def make_mp3_bytes(payload_len: int = 100) -> bytes:
    return b"\xff\xfb\x90\x00" + b"\x00" * payload_len


def make_wav_header_bytes(duration: float, sample_rate: int = 16000, channels: int = 1) -> bytes:
    # 44-byte header claiming `duration` with no payload. A real 3601s WAV is 115 MB,
    # which trips the size gate too, so the duration test would pass for the wrong
    # reason. `wave` reads nframes from the data chunk size and never checks the bytes.
    block_align = channels * 2
    data_size = int(duration * sample_rate) * block_align
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * block_align, block_align, 16
        )
        + b"data"
        + struct.pack("<I", data_size)
    )


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()


@pytest.fixture
def engine(tmp_path):
    """A file-backed SQLite database per test under tmp_path, per the milestone's
    isolation rule. A file rather than :memory: so the test exercises the same
    path-creation and connection behaviour the app uses."""
    from src.database.connection import get_engine, init_db

    eng = get_engine(str(tmp_path / "test_calls.db"))
    init_db(eng)
    return eng


@pytest.fixture
def audit_logger(engine):
    from src.security.audit import AuditLogger

    return AuditLogger(engine)
