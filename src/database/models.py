# callcenter-intelligence
# src/database/models.py

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class CallRecord(Base):
    # One row per processed call. The JSON columns hold the Pydantic payloads
    # verbatim rather than a normalized schema: the pipeline's contracts live in
    # state.py, and duplicating them as columns would give two sources of truth.
    __tablename__ = "call_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    call_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    audio_filename: Mapped[str | None] = mapped_column(String(512))
    transcript_text: Mapped[str | None] = mapped_column(Text)
    summary_json: Mapped[str | None] = mapped_column(Text)
    qa_scores_json: Mapped[str | None] = mapped_column(Text)
    report_json: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    trace_id: Mapped[str | None] = mapped_column(String(128))


class AuditLogEntry(Base):
    # Append-only by discipline: nothing in the codebase updates or deletes a row
    # here. call_id is indexed but NOT unique - a single call writes one row per
    # pipeline event.
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    call_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    user: Mapped[str] = mapped_column(String(64), nullable=False, default="app")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    details: Mapped[str | None] = mapped_column(Text)


class TranscriptionCache(Base):
    # audio_hash is the SHA-256 of the audio bytes, unique so a second identical
    # upload collides here instead of paying for another Whisper pass.
    __tablename__ = "transcription_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audio_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    transcription_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
