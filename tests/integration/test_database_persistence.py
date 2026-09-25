# callcenter-intelligence
# tests/integration/test_database_persistence.py

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from src.agents.report import compile_report, persist_report
from src.database.connection import get_engine, init_db, session_scope
from src.database.models import AuditLogEntry, CallRecord, TranscriptionCache
from src.graph.state import (
    CallStatus,
    IntakeResult,
    TranscriptionResult,
    TranscriptionSegment,
)
from src.security.audit import AuditLogger
from tests.integration.conftest import make_qa, make_summary


def _transcription(call_id: str) -> TranscriptionResult:
    return TranscriptionResult(
        call_id=call_id,
        full_text="Thank you for calling. My account was charged twice.",
        segments=[
            TranscriptionSegment(
                start=0.0, end=2.0, text="Thank you for calling.", speaker="Agent", confidence=0.91
            ),
            TranscriptionSegment(
                start=2.4,
                end=5.0,
                text="My account was charged twice.",
                speaker="Customer",
                confidence=0.88,
            ),
        ],
        duration_seconds=5.0,
    )


def _report(call_id: str = "call-persist-1"):
    return compile_report(
        IntakeResult(
            call_id=call_id, validation_passed=True, audio_format="wav", filename="call.wav"
        ),
        _transcription(call_id),
        make_summary(),
        make_qa(),
    )


# ---- CallRecord --------------------------------------------------------------


def test_insert_and_retrieve_call_record(engine) -> None:
    report = _report()
    persist_report(report, engine)
    with session_scope(engine) as s:
        row = s.scalar(select(CallRecord).where(CallRecord.call_id == report.call_id))
    assert row is not None
    assert row.status == CallStatus.COMPLETED.value
    assert row.audio_filename == "call.wav"


def test_call_record_stores_serialised_payloads(engine) -> None:
    """Round-trip fidelity, not scoring: compile_report is a pure assembler and
    persists what it was handed. The deterministic recomputation is run_qa_scoring's
    job and is covered by test_pipeline_end_to_end.py."""
    report = _report("call-json")
    persist_report(report, engine)
    with session_scope(engine) as s:
        row = s.scalar(select(CallRecord).where(CallRecord.call_id == "call-json"))
    assert json.loads(row.summary_json) == json.loads(report.summary.model_dump_json())
    assert json.loads(row.qa_scores_json) == json.loads(report.qa_scores.model_dump_json())
    assert json.loads(row.report_json)["call_id"] == "call-json"
    assert "charged twice" in row.transcript_text


def test_persists_across_connections(engine, tmp_path) -> None:
    """Survives the session closing. A new engine over the same file is the
    closest honest proxy for a restart."""
    report = _report("call-restart")
    persist_report(report, engine)
    reopened = get_engine(str(engine.url.database))
    init_db(reopened)
    with session_scope(reopened) as s:
        assert s.scalar(select(CallRecord).where(CallRecord.call_id == "call-restart")) is not None


def test_duplicate_call_id_rejected(engine) -> None:
    from sqlalchemy.exc import IntegrityError

    persist_report(_report("call-dup"), engine)
    with pytest.raises(IntegrityError):
        persist_report(_report("call-dup"), engine)


# ---- AuditLogEntry -----------------------------------------------------------


def test_insert_and_retrieve_audit_entries(engine) -> None:
    audit = AuditLogger(engine)
    audit.log("call-a", "intake_passed")
    audit.log("call-a", "transcribed", details={"cached": False})
    audit.log("call-b", "intake_rejected", details={"reason": "Empty file"})

    entries = audit.entries_for_call("call-a")
    assert [e.action for e in entries] == ["intake_passed", "transcribed"]
    assert json.loads(entries[1].details) == {"cached": False}
    assert len(audit.entries_for_call("call-b")) == 1


def test_audit_survives_a_new_engine(engine) -> None:
    AuditLogger(engine).log("call-a", "completed")
    reopened = get_engine(str(engine.url.database))
    init_db(reopened)
    with session_scope(reopened) as s:
        rows = list(s.scalars(select(AuditLogEntry).where(AuditLogEntry.call_id == "call-a")))
    assert [r.action for r in rows] == ["completed"]


def test_audit_ordering_is_insertion_order(engine) -> None:
    audit = AuditLogger(engine)
    expected = [
        "intake_passed",
        "transcribed",
        "injection_check_passed",
        "pii_redacted",
        "analyzed",
        "completed",
    ]
    for action in expected:
        audit.log("call-order", action)
    assert [e.action for e in audit.entries_for_call("call-order")] == expected


# ---- TranscriptionCache ------------------------------------------------------


def test_transcription_cache_round_trip(engine) -> None:
    from src.agents.transcription import check_cache, save_cache

    result = _transcription("call-cache")
    assert check_cache(engine, "hash-abc") is None
    save_cache(engine, "hash-abc", result)
    cached = check_cache(engine, "hash-abc")
    assert cached is not None
    assert cached["full_text"] == result.full_text


def test_cache_hash_is_unique(engine) -> None:
    from sqlalchemy.exc import IntegrityError

    from src.agents.transcription import save_cache

    save_cache(engine, "hash-same", _transcription("c1"))
    with pytest.raises(IntegrityError):
        save_cache(engine, "hash-same", _transcription("c2"))


def test_three_tables_coexist(engine) -> None:
    from src.agents.transcription import save_cache

    persist_report(_report("call-all"), engine)
    AuditLogger(engine).log("call-all", "completed")
    save_cache(engine, "hash-all", _transcription("call-all"))
    with session_scope(engine) as s:
        assert s.scalar(select(CallRecord).where(CallRecord.call_id == "call-all")) is not None
        assert len(list(s.scalars(select(AuditLogEntry)))) == 1
        assert len(list(s.scalars(select(TranscriptionCache)))) == 1
