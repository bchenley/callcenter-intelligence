# callcenter-intelligence
# tests/unit/test_transcription.py

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.agents import transcription as tx
from src.graph.state import IntakeResult, TranscriptionSegment
from tests.conftest import make_wav_bytes


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


def _fake_model(segments: list[FakeSeg]) -> MagicMock:
    model = MagicMock()
    model.transcribe.return_value = (iter(segments), FakeInfo())
    return model


@pytest.fixture
def intake(tmp_path) -> IntakeResult:
    path = tmp_path / "call.wav"
    path.write_bytes(make_wav_bytes(3.0))
    return IntakeResult(
        call_id="call-abc",
        validation_passed=True,
        audio_format="wav",
        temp_path=str(path),
        filename="call.wav",
    )


DEFAULT_SEGS = [
    FakeSeg(0.0, 2.0, "Thank you for calling Acme support, how can I help you today?"),
    FakeSeg(2.3, 5.0, "My account was charged twice this month and I need help with it."),
]


# ---- call kwargs -------------------------------------------------------------


def test_whisper_called_with_required_kwargs(intake, engine, monkeypatch) -> None:
    model = _fake_model(DEFAULT_SEGS)
    monkeypatch.setattr(tx, "_get_whisper_model", lambda _s="tiny": model)
    tx.run_transcription(intake, engine)
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["beam_size"] == 1
    assert kwargs["language"] == "en"
    assert kwargs["vad_filter"] is True
    assert kwargs["vad_parameters"] == {"min_silence_duration_ms": 300}
    assert kwargs["word_timestamps"] is True
    assert kwargs["condition_on_previous_text"] is False


def test_call_id_propagates(intake, engine, monkeypatch) -> None:
    monkeypatch.setattr(tx, "_get_whisper_model", lambda _s="tiny": _fake_model(DEFAULT_SEGS))
    assert tx.run_transcription(intake, engine).call_id == intake.call_id


def test_failed_intake_raises(engine) -> None:
    bad = IntakeResult(call_id="c", validation_passed=False, validation_error="Empty file")
    with pytest.raises(tx.TranscriptionError):
        tx.run_transcription(bad, engine)


# ---- SHA-256 cache -----------------------------------------------------------


def test_second_call_hits_cache_and_transcribes_once(intake, engine, monkeypatch) -> None:
    model = _fake_model(DEFAULT_SEGS)
    monkeypatch.setattr(tx, "_get_whisper_model", lambda _s="tiny": model)
    first = tx.run_transcription(intake, engine)
    model.transcribe.return_value = (iter(DEFAULT_SEGS), FakeInfo())
    second = tx.run_transcription(intake, engine)
    assert model.transcribe.call_count == 1
    assert second.cached is True and first.cached is False
    assert second.full_text == first.full_text


def test_cache_hit_substitutes_new_call_id(intake, engine, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tx, "_get_whisper_model", lambda _s="tiny": _fake_model(DEFAULT_SEGS))
    tx.run_transcription(intake, engine)
    again = IntakeResult(
        call_id="call-different",
        validation_passed=True,
        audio_format="wav",
        temp_path=intake.temp_path,
        filename="call.wav",
    )
    assert tx.run_transcription(again, engine).call_id == "call-different"


def test_hash_is_content_addressed(tmp_path) -> None:
    a, b, c = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"
    data = make_wav_bytes(2.0)
    a.write_bytes(data)
    b.write_bytes(data)
    c.write_bytes(make_wav_bytes(3.0))
    assert tx.compute_audio_hash(a) == tx.compute_audio_hash(b)
    assert tx.compute_audio_hash(a) != tx.compute_audio_hash(c)
    assert len(tx.compute_audio_hash(a)) == 64


def test_hash_reads_in_chunks() -> None:
    assert tx.HASH_CHUNK_BYTES == 8192


# ---- confidence --------------------------------------------------------------


@pytest.mark.parametrize(
    ("avg_logprob", "no_speech", "expected"),
    [
        (0.0, 0.0, 1.0),
        (-1.0, 1.0, 0.0),
        (-0.2, 0.05, round(0.8 * 0.7 + 0.95 * 0.3, 4)),
        (-5.0, 0.0, round(0.0 * 0.7 + 1.0 * 0.3, 4)),
        (0.5, 0.0, 1.0),
    ],
)
def test_confidence_formula(avg_logprob, no_speech, expected) -> None:
    assert tx._segment_confidence(avg_logprob, no_speech) == expected


def test_confidence_always_in_unit_interval() -> None:
    for lp in (-9.0, -1.0, 0.0, 2.0):
        for ns in (0.0, 0.5, 1.0):
            assert 0.0 <= tx._segment_confidence(lp, ns) <= 1.0


def test_low_confidence_flags_call(intake, engine, monkeypatch) -> None:
    noisy = [FakeSeg(0.0, 2.0, "mumbled words here", avg_logprob=-5.0, no_speech_prob=0.9)]
    monkeypatch.setattr(tx, "_get_whisper_model", lambda _s="tiny": _fake_model(noisy))
    result = tx.run_transcription(intake, engine)
    assert result.low_confidence is True
    assert result.flagged_for_review is True


# ---- artifact cleaning -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "must_not_contain"),
    [
        ("[BLANK_AUDIO] hello there", "BLANK_AUDIO"),
        ("hello ..... there", "....."),
        ("great, thanks for watching", "thanks for watching"),
        ("[MUSIC] the account is closed", "MUSIC"),
        ("(applause) we are done", "applause"),
        ("thank you thank you thank you for that", "thank you thank you"),
    ],
)
def test_clean_removes_artifacts(raw: str, must_not_contain: str) -> None:
    assert must_not_contain.lower() not in tx._clean_transcript_text(raw).lower()


def test_clean_preserves_real_speech() -> None:
    text = "I was charged twice on my credit card in March."
    assert tx._clean_transcript_text(text) == text


def test_blank_only_segment_dropped(intake, engine, monkeypatch) -> None:
    segs = [FakeSeg(0.0, 1.0, "[BLANK_AUDIO]"), FakeSeg(1.0, 3.0, "I need help with my bill.")]
    monkeypatch.setattr(tx, "_get_whisper_model", lambda _s="tiny": _fake_model(segs))
    assert len(tx.run_transcription(intake, engine).segments) == 1


# ---- diarization -------------------------------------------------------------


def _seg(start: float, end: float, text: str) -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text, confidence=0.9)


def test_first_segment_defaults_to_agent() -> None:
    out = tx.SpeakerDiarizer().assign([_seg(0, 1, "Okay.")])
    assert out[0].speaker == "Agent"


def test_content_pattern_beats_timing() -> None:
    """No gap, no question - only the phrase identifies the speaker."""
    segs = [_seg(0, 2, "Okay."), _seg(2.0, 4.0, "My account was charged twice.")]
    assert tx.SpeakerDiarizer().assign(segs)[1].speaker == "Customer"


def test_agent_pattern_recognised() -> None:
    segs = [_seg(0, 2, "Hello."), _seg(2.0, 4.0, "Let me check that for you.")]
    assert tx.SpeakerDiarizer().assign(segs)[1].speaker == "Agent"


def test_gap_switches_speaker() -> None:
    segs = [_seg(0, 2, "Right."), _seg(4.0, 5.0, "Understood.")]
    out = tx.SpeakerDiarizer().assign(segs)
    assert out[0].speaker == "Agent" and out[1].speaker == "Customer"


def test_gap_below_threshold_does_not_switch() -> None:
    segs = [_seg(0, 2, "Right."), _seg(2.5, 3.5, "Understood.")]
    assert tx.SpeakerDiarizer().assign(segs)[1].speaker == "Agent"


def test_question_switches_speaker() -> None:
    segs = [_seg(0, 2, "Could you confirm the last four digits?"), _seg(2.1, 3.0, "Sure it is.")]
    assert tx.SpeakerDiarizer().assign(segs)[1].speaker == "Customer"


def test_short_affirmation_after_long_segment_switches() -> None:
    long_text = "So what I will do is process the refund and it should land in three days."
    segs = [_seg(0, 5, long_text), _seg(5.1, 5.6, "Okay great.")]
    assert tx.SpeakerDiarizer().assign(segs)[1].speaker == "Customer"


def test_short_affirmation_after_short_segment_does_not_switch() -> None:
    segs = [_seg(0, 1, "Sure thing."), _seg(1.1, 1.6, "Okay great.")]
    assert tx.SpeakerDiarizer().assign(segs)[1].speaker == "Agent"


def test_empty_segments_handled() -> None:
    assert tx.SpeakerDiarizer().assign([]) == []


def test_alternating_conversation_labels_both_speakers() -> None:
    segs = [
        _seg(0, 2, "Thank you for calling Acme, how can I help?"),
        _seg(2.2, 5, "My account was charged twice this month."),
        _seg(5.2, 8, "Let me check that for you."),
    ]
    out = tx.SpeakerDiarizer().assign(segs)
    assert [s.speaker for s in out] == ["Agent", "Customer", "Agent"]


# ---- model singleton ---------------------------------------------------------


def test_device_falls_back_to_cpu_int8_without_cuda(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert tx._resolve_device() == ("cpu", "int8")


def test_singleton_reused_for_same_size(monkeypatch) -> None:
    tx.reset_model_cache()
    built: list[str] = []

    class FakeWhisper:
        def __init__(self, size, device, compute_type):
            built.append(size)

    import sys
    import types

    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = FakeWhisper
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)
    tx._get_whisper_model("tiny")
    tx._get_whisper_model("tiny")
    assert built == ["tiny"], "model must load once, not per call"
    tx.reset_model_cache()
