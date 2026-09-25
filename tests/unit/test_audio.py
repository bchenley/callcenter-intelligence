# callcenter-intelligence
# tests/unit/test_audio.py

from __future__ import annotations

import pytest

from src.utils.audio import (
    MAX_FILE_SIZE_BYTES,
    AudioValidationError,
    detect_audio_format,
    extract_audio_properties,
    validate_audio_file,
)
from tests.conftest import make_mp3_bytes, make_wav_bytes


class TestFormatDetection:
    def test_wav_detected_from_riff_wave(self):
        assert detect_audio_format(make_wav_bytes()) == "wav"

    def test_mp3_detected_from_frame_sync(self):
        assert detect_audio_format(make_mp3_bytes()) == "mp3"

    def test_mp3_detected_from_id3_tag(self):
        assert detect_audio_format(b"ID3\x04\x00\x00" + b"\x00" * 100) == "mp3"

    def test_flac_detected(self):
        assert detect_audio_format(b"fLaC" + b"\x00" * 100) == "flac"

    def test_m4a_detected_from_ftyp_box(self):
        assert detect_audio_format(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 100) == "m4a"

    def test_unknown_bytes_return_none(self):
        assert detect_audio_format(b"OggS" + b"\x00" * 100) is None

    def test_too_short_returns_none(self):
        assert detect_audio_format(b"RIFF") is None

    def test_extension_does_not_influence_detection(self):
        assert detect_audio_format(make_mp3_bytes()) == "mp3"


class TestValidation:
    def test_valid_wav_passes(self):
        result = validate_audio_file(make_wav_bytes(), "call.wav")
        assert result.is_valid is True
        assert result.detected_format == "wav"

    def test_empty_file_rejected(self):
        result = validate_audio_file(b"", "empty.wav")
        assert result.is_valid is False
        assert "Empty file" in result.error

    def test_oversized_file_rejected(self):
        result = validate_audio_file(b"\x00" * (MAX_FILE_SIZE_BYTES + 1), "big.wav")
        assert result.is_valid is False
        assert "exceeds maximum" in result.error

    def test_unsupported_format_rejected(self):
        result = validate_audio_file(b"OggS" + b"\x00" * 100, "sound.ogg")
        assert result.is_valid is False
        assert "Unsupported" in result.error

    def test_text_renamed_to_wav_rejected(self):
        result = validate_audio_file(b"this is not audio at all" * 10, "spoofed.wav")
        assert result.is_valid is False
        assert "Unsupported" in result.error


class TestPropertyExtraction:
    def test_wav_duration_is_accurate(self):
        props = extract_audio_properties(make_wav_bytes(5.0, 16000), "wav")
        assert 4.9 <= props.duration_seconds <= 5.1

    def test_wav_sample_rate_and_channels(self):
        props = extract_audio_properties(make_wav_bytes(1.0, 44100, channels=2), "wav")
        assert props.sample_rate == 44100
        assert props.channels == 2

    def test_corrupt_wav_raises(self):
        with pytest.raises(AudioValidationError):
            extract_audio_properties(b"RIFF\x00\x00\x00\x00WAVEgarbage", "wav")
