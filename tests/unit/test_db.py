# callcenter-intelligence
# tests/unit/test_db.py

from __future__ import annotations

import pytest
from sqlalchemy import inspect, select

from src.database.connection import get_engine, get_session, init_db, ping, session_scope
from src.database.models import AuditLogEntry, CallRecord, TranscriptionCache


def test_three_tables_created(engine) -> None:
    assert {"call_records", "audit_log", "transcription_cache"} <= set(
        inspect(engine).get_table_names()
    )


def test_call_id_unique_index(engine) -> None:
    idx = [
        i for i in inspect(engine).get_indexes("call_records") if i["column_names"] == ["call_id"]
    ]
    assert idx and bool(idx[0]["unique"]) is True  # sqlite reports 1/0, not True/False


def test_audio_hash_unique_index(engine) -> None:
    idx = [
        i
        for i in inspect(engine).get_indexes("transcription_cache")
        if i["column_names"] == ["audio_hash"]
    ]
    assert idx and bool(idx[0]["unique"]) is True  # sqlite reports 1/0, not True/False


def test_call_record_columns(engine) -> None:
    cols = {c["name"] for c in inspect(engine).get_columns("call_records")}
    assert {
        "id",
        "call_id",
        "status",
        "audio_filename",
        "transcript_text",
        "summary_json",
        "qa_scores_json",
        "report_json",
        "processed_at",
        "trace_id",
    } <= cols


def test_transcription_cache_columns(engine) -> None:
    cols = {c["name"] for c in inspect(engine).get_columns("transcription_cache")}
    assert {"id", "audio_hash", "transcription_json", "created_at"} <= cols


def test_sessionmaker_is_cached_per_engine(engine) -> None:
    """Rubric asks for pooling via a cached factory, not a new one per request."""
    from src.database.connection import _session_factories

    get_session(engine).close()
    first = _session_factories[id(engine)]
    get_session(engine).close()
    assert _session_factories[id(engine)] is first


def test_distinct_engines_get_distinct_factories(tmp_path) -> None:
    a = get_engine(str(tmp_path / "a.db"))
    b = get_engine(str(tmp_path / "b.db"))
    init_db(a)
    init_db(b)
    from src.database.connection import _session_factories

    get_session(a).close()
    get_session(b).close()
    assert _session_factories[id(a)] is not _session_factories[id(b)]


def test_session_scope_commits(engine) -> None:
    with session_scope(engine) as s:
        s.add(CallRecord(call_id="c1", status="completed"))
    with session_scope(engine) as s:
        assert s.scalar(select(CallRecord).where(CallRecord.call_id == "c1")) is not None


def test_session_scope_rolls_back_and_reraises(engine) -> None:
    with pytest.raises(RuntimeError), session_scope(engine) as s:
        s.add(CallRecord(call_id="c2", status="completed"))
        raise RuntimeError("node failed")
    with session_scope(engine) as s:
        assert s.scalar(select(CallRecord).where(CallRecord.call_id == "c2")) is None


def test_persist_then_query_by_call_id(engine) -> None:
    with session_scope(engine) as s:
        s.add(CallRecord(call_id="c3", status="completed", audio_filename="call.wav"))
    with session_scope(engine) as s:
        rec = s.scalar(select(CallRecord).where(CallRecord.call_id == "c3"))
        assert rec.audio_filename == "call.wav"


def test_duplicate_call_id_rejected(engine) -> None:
    from sqlalchemy.exc import IntegrityError

    with session_scope(engine) as s:
        s.add(CallRecord(call_id="dup", status="completed"))
    with pytest.raises(IntegrityError), session_scope(engine) as s:
        s.add(CallRecord(call_id="dup", status="failed"))


def test_duplicate_audio_hash_rejected(engine) -> None:
    from sqlalchemy.exc import IntegrityError

    with session_scope(engine) as s:
        s.add(TranscriptionCache(audio_hash="h1", transcription_json="{}"))
    with pytest.raises(IntegrityError), session_scope(engine) as s:
        s.add(TranscriptionCache(audio_hash="h1", transcription_json="{}"))


def test_audit_allows_repeated_call_id(engine) -> None:
    with session_scope(engine) as s:
        s.add(AuditLogEntry(call_id="c4", action="intake"))
        s.add(AuditLogEntry(call_id="c4", action="transcribe"))
    with session_scope(engine) as s:
        assert len(list(s.scalars(select(AuditLogEntry).where(AuditLogEntry.call_id == "c4")))) == 2


def test_parent_directory_created(tmp_path) -> None:
    eng = get_engine(str(tmp_path / "nested" / "deep" / "calls.db"))
    init_db(eng)
    assert (tmp_path / "nested" / "deep" / "calls.db").exists()


def test_ping(engine) -> None:
    assert ping(engine) is True


def test_defaults_populate_timestamps(engine) -> None:
    with session_scope(engine) as s:
        rec = CallRecord(call_id="c5", status="completed")
        s.add(rec)
        s.flush()
        assert rec.processed_at is not None
