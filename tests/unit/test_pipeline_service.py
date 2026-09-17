# callcenter-intelligence
# tests/unit/test_pipeline_service.py

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.graph.state import CallStatus, TranscriptionResult, TranscriptionSegment
from src.services import pipeline as svc


@pytest.fixture(autouse=True)
def _clean_tracking():
    svc.reset_temp_tracking()
    yield
    svc.reset_temp_tracking()


def _transcription() -> TranscriptionResult:
    return TranscriptionResult(
        call_id="c1",
        full_text="hello there",
        segments=[
            TranscriptionSegment(
                start=0.0, end=2.0, text="Thank you for calling.", speaker="Agent", confidence=0.95
            ),
            TranscriptionSegment(
                start=95.0, end=99.0, text="mumble mumble", speaker="Customer", confidence=0.31
            ),
        ],
    )


# ---- transcript formatting ---------------------------------------------------


def test_transcript_has_mmss_and_speaker() -> None:
    text = svc.format_transcript(_transcription())
    assert "[00:00] Agent: Thank you for calling." in text
    assert "[01:35] Customer:" in text


def test_low_confidence_segments_marked() -> None:
    lines = svc.format_transcript(_transcription()).splitlines()
    assert svc.LOW_CONF_MARKER not in lines[0]
    assert svc.LOW_CONF_MARKER in lines[1]


def test_threshold_controls_the_marker() -> None:
    assert svc.LOW_CONF_MARKER not in svc.format_transcript(
        _transcription(), confidence_threshold=0.1
    )


# ---- rolling temp-file cap ---------------------------------------------------


def test_cap_is_fifty() -> None:
    assert svc.TEMP_FILE_CAP == 50


def test_tracking_evicts_and_deletes_beyond_the_cap(tmp_path) -> None:
    made = []
    for i in range(svc.TEMP_FILE_CAP + 5):
        p = tmp_path / f"f{i}.tmp"
        p.write_bytes(b"x")
        made.append(p)
        svc._track(p)
    assert len(svc.tracked_temp_files()) == svc.TEMP_FILE_CAP
    assert not any(p.exists() for p in made[:5]), "oldest five should be deleted"
    assert all(p.exists() for p in made[5:]), "the newest fifty must survive"


def test_eviction_is_oldest_first(tmp_path) -> None:
    for i in range(svc.TEMP_FILE_CAP + 1):
        p = tmp_path / f"f{i}.tmp"
        p.write_bytes(b"x")
        svc._track(p)
    remaining = [p.name for p in svc.tracked_temp_files()]
    assert "f0.tmp" not in remaining and "f50.tmp" in remaining


def test_missing_file_does_not_break_eviction(tmp_path) -> None:
    for i in range(svc.TEMP_FILE_CAP + 1):
        p = tmp_path / f"f{i}.tmp"
        if i > 0:
            p.write_bytes(b"x")
        svc._track(p)
    assert len(svc.tracked_temp_files()) == svc.TEMP_FILE_CAP


# ---- gradio input handling ---------------------------------------------------


def test_numpy_tuple_written_as_real_wav() -> None:
    np = pytest.importorskip("numpy")
    path = svc.numpy_tuple_to_wav((16000, np.zeros(16000, dtype="float32")))
    raw = Path(path).read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"


def test_int16_microphone_input_normalised() -> None:
    np = pytest.importorskip("numpy")
    data = (np.ones(8000) * 32767).astype("int16")
    path = svc.numpy_tuple_to_wav((16000, data))
    assert Path(path).stat().st_size > 44


def test_none_audio_returns_a_message_not_an_exception() -> None:
    result = svc.process_call(None, MagicMock())
    assert result.status == CallStatus.FAILED.value
    assert "No audio" in result.error


def test_unreadable_path_returns_a_message(tmp_path) -> None:
    result = svc.process_call(str(tmp_path / "missing.wav"), MagicMock())
    assert result.status == CallStatus.FAILED.value
    assert "Could not read" in result.error


# ---- result shape ------------------------------------------------------------


def test_failed_run_carries_the_reason(tmp_path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE" + b"\x00" * 100)
    wf = MagicMock()
    wf.invoke.return_value = {"status": CallStatus.FAILED.value, "error": "Empty file"}
    result = svc.process_call(str(wav), wf)
    assert result.ok is False
    assert result.error == "Empty file"
    assert result.pdf_path is None


def test_halt_without_report_still_returns_the_transcript(tmp_path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE" + b"\x00" * 100)
    wf = MagicMock()
    wf.invoke.return_value = {
        "status": CallStatus.FLAGGED_FOR_REVIEW.value,
        "error": "Prompt injection detected in transcript; blocked before any LLM call.",
        "transcription": _transcription(),
        "intake": MagicMock(call_id="c1"),
    }
    result = svc.process_call(str(wav), wf)
    assert result.pdf_path is None
    assert "Thank you for calling" in result.transcript
    assert "injection" in result.error.lower()


def test_ok_property_tracks_status() -> None:
    assert svc.PipelineResult(status=CallStatus.COMPLETED.value).ok is True
    assert svc.PipelineResult(status=CallStatus.FLAGGED_FOR_REVIEW.value).ok is False
