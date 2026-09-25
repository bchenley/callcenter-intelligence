# callcenter-intelligence
# tests/unit/test_intake.py

from __future__ import annotations

import os

from src.agents.intake import run_intake
from src.graph.state import AudioInput
from src.utils.audio import MAX_FILE_SIZE_BYTES
from tests.conftest import make_mp3_bytes, make_wav_bytes, make_wav_header_bytes


def _input(data: bytes, filename: str, **kw) -> AudioInput:
    return AudioInput(audio_data=data, filename=filename, **kw)


class TestHappyPath:
    def test_valid_wav_passes(self):
        result = run_intake(_input(make_wav_bytes(5.0), "call.wav"))
        assert result.validation_passed is True
        assert result.validation_error is None
        assert result.audio_format == "wav"
        assert 4.9 <= result.properties.duration_seconds <= 5.1
        assert result.properties.sample_rate == 16000

    def test_temp_file_written_and_readable(self):
        data = make_wav_bytes(1.0)
        result = run_intake(_input(data, "call.wav"))
        assert result.temp_path and result.temp_path.endswith(".wav")
        assert os.path.exists(result.temp_path)
        with open(result.temp_path, "rb") as fh:
            assert fh.read() == data
        os.unlink(result.temp_path)

    def test_each_call_gets_a_distinct_id(self):
        a = run_intake(_input(make_wav_bytes(1.0), "call.wav"))
        b = run_intake(_input(make_wav_bytes(1.0), "call.wav"))
        assert a.call_id != b.call_id
        for r in (a, b):
            if r.temp_path:
                os.unlink(r.temp_path)

    def test_failed_intake_still_gets_a_call_id(self):
        result = run_intake(_input(b"", "empty.wav"))
        assert result.call_id


class TestRejection:
    def test_empty_file(self):
        result = run_intake(_input(b"", "empty.wav"))
        assert result.validation_passed is False
        assert "Empty file" in result.validation_error

    def test_unsupported_format(self):
        result = run_intake(_input(b"\x00" * 100, "bad.ogg"))
        assert result.validation_passed is False
        assert "Unsupported" in result.validation_error

    def test_oversized_non_wav(self):
        result = run_intake(_input(make_mp3_bytes(MAX_FILE_SIZE_BYTES), "big.mp3"))
        assert result.validation_passed is False
        assert "exceeds maximum" in result.validation_error

    def test_failed_result_carries_no_partial_properties(self):
        result = run_intake(_input(b"\x00" * 100, "bad.ogg"))
        assert result.properties.duration_seconds == 0.0
        assert result.temp_path is None


class TestTheOrderingRule:
    # Both gates would reject a 61-minute WAV; only the order decides the reason given.

    def test_long_wav_reports_duration_not_size(self):
        long_wav = make_wav_header_bytes(3601.0)
        result = run_intake(_input(long_wav, "long.wav"))
        assert result.validation_passed is False
        assert "duration" in result.validation_error.lower()
        assert "exceeds maximum size" not in result.validation_error

    def test_wav_at_exactly_the_limit_is_not_rejected_for_duration(self):
        at_limit = make_wav_header_bytes(3600.0)
        result = run_intake(_input(at_limit, "hour.wav"))
        assert "duration" not in (result.validation_error or "").lower()


class TestMetadataPIIScan:
    def test_ssn_in_caller_id_detected(self):
        result = run_intake(_input(make_wav_bytes(1.0), "call.wav", caller_id="SSN: 123-45-6789"))
        assert result.pii_scan.pii_detected is True
        assert "caller_id" in result.pii_scan.affected_fields
        assert "SSN" in result.pii_scan.pii_types
        if result.temp_path:
            os.unlink(result.temp_path)

    def test_clean_metadata_not_flagged(self):
        result = run_intake(
            _input(make_wav_bytes(1.0), "call.wav", caller_id="CUST-8842", department="Billing")
        )
        assert result.pii_scan.pii_detected is False
        assert result.pii_scan.affected_fields == []
        if result.temp_path:
            os.unlink(result.temp_path)

    def test_email_in_department_detected(self):
        result = run_intake(
            _input(make_wav_bytes(1.0), "call.wav", department="escalations@acme.com")
        )
        assert result.pii_scan.pii_detected is True
        assert "department" in result.pii_scan.affected_fields
        assert "EMAIL" in result.pii_scan.pii_types
        if result.temp_path:
            os.unlink(result.temp_path)

    def test_no_metadata_is_not_pii(self):
        result = run_intake(_input(make_wav_bytes(1.0), "call.wav"))
        assert result.pii_scan.pii_detected is False
        if result.temp_path:
            os.unlink(result.temp_path)
