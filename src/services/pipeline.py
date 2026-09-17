# callcenter-intelligence
# src/services/pipeline.py

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.report import generate_report_json, generate_report_pdf
from src.graph.state import AudioInput, CallStatus, TranscriptionResult
from src.utils.formatters import format_qa, format_summary, secs_to_mmss

logger = logging.getLogger(__name__)

# Rolling cap. A long session otherwise leaves one PDF, one JSON and one WAV per
# call in the system temp directory until the host runs out of disk. The cap is a
# count rather than an age because the failure mode is volume, not staleness.
TEMP_FILE_CAP = 50
_temp_files: list[Path] = []

LOW_CONF_MARKER = "[LOW CONF]"


@dataclass(frozen=True)
class PipelineResult:
    status: str
    transcript: str = ""
    summary_markdown: str = ""
    qa_markdown: str = ""
    pdf_path: str | None = None
    json_path: str | None = None
    error: str | None = None
    call_id: str | None = None
    elapsed_seconds: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == CallStatus.COMPLETED.value


def _track(path: Path) -> Path:
    """Register a temp file and evict the oldest beyond the cap."""
    _temp_files.append(path)
    while len(_temp_files) > TEMP_FILE_CAP:
        stale = _temp_files.pop(0)
        try:
            stale.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - platform dependent
            logger.warning("could not remove temp file %s: %s", stale, exc)
    return path


def reset_temp_tracking() -> None:
    """Tests only: clears the tracking list without touching disk."""
    _temp_files.clear()


def tracked_temp_files() -> tuple[Path, ...]:
    return tuple(_temp_files)


def _write_temp(data: bytes, suffix: str) -> Path:
    # delete=False: Gradio serves this file after process_call returns, so it must
    # outlive the handle. The rolling cap in _track is what reclaims it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        name = handle.name
    return _track(Path(name))


def numpy_tuple_to_wav(audio: tuple[int, Any]) -> Path:
    """Gradio hands back (sample_rate, ndarray), not a file. soundfile writes the
    array to a real WAV so intake's magic-byte check has actual bytes to inspect -
    the alternative is a special case that skips validation for microphone input,
    which is exactly the path an attacker would choose."""
    import numpy as np
    import soundfile as sf

    sample_rate, array = audio
    data = np.asarray(array)
    if data.dtype.kind in "iu":
        # soundfile wants float in [-1, 1] or a declared integer subtype; normalising
        # by the dtype max keeps microphone int16 from clipping into noise.
        data = data.astype("float32") / float(np.iinfo(data.dtype).max)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        name = handle.name
    path = Path(name)
    sf.write(str(path), data, int(sample_rate), format="WAV")
    return _track(path)


def format_transcript(transcription: TranscriptionResult, confidence_threshold: float = 0.6) -> str:
    """[MM:SS] Speaker: text, with low-confidence segments marked so a reviewer
    knows which lines to distrust rather than silently reading noise as speech."""
    lines: list[str] = []
    for seg in transcription.segments:
        marker = f" {LOW_CONF_MARKER}" if seg.confidence < confidence_threshold else ""
        lines.append(f"[{secs_to_mmss(seg.start)}] {seg.speaker}:{marker} {seg.text}")
    return "\n".join(lines)


def process_call(
    audio: Any,
    workflow: Any,
    caller_id: str | None = None,
    department: str | None = None,
    filename: str = "call.wav",
    confidence_threshold: float = 0.6,
) -> PipelineResult:
    """Gradio's entry point into the graph. Returns a typed result rather than
    raising: the UI must show a clear message, never a traceback."""
    started = time.monotonic()
    try:
        if isinstance(audio, tuple):
            path = numpy_tuple_to_wav(audio)
            filename = path.name
        elif audio is None:
            return PipelineResult(status=CallStatus.FAILED.value, error="No audio provided.")
        else:
            path = Path(audio)
            filename = path.name
        data = path.read_bytes()
    except Exception as exc:  # broad on purpose: any read failure becomes a message
        logger.exception("could not read uploaded audio")
        return PipelineResult(status=CallStatus.FAILED.value, error=f"Could not read audio: {exc}")

    state = workflow.invoke(
        {
            "audio_input": AudioInput(
                audio_data=data,
                filename=filename,
                caller_id=caller_id or None,
                department=department or None,
            )
        }
    )

    elapsed = round(time.monotonic() - started, 2)
    status = state.get("status", CallStatus.FAILED.value)
    report = state.get("report")
    if report is None:
        return PipelineResult(
            status=status,
            error=state.get("error") or "Processing failed.",
            call_id=state["intake"].call_id if state.get("intake") else None,
            elapsed_seconds=elapsed,
        )

    pdf_path = _write_temp(generate_report_pdf(report), ".pdf")
    json_path = _write_temp(generate_report_json(report).encode("utf-8"), ".json")

    return PipelineResult(
        status=status,
        transcript=format_transcript(report.transcript, confidence_threshold)
        if report.transcript
        else "",
        summary_markdown=format_summary(report.summary) if report.summary else "",
        qa_markdown=format_qa(report.qa_scores) if report.qa_scores else "",
        pdf_path=str(pdf_path),
        json_path=str(json_path),
        error=state.get("error"),
        call_id=report.call_id,
        elapsed_seconds=elapsed,
    )


def result_to_json(result: PipelineResult) -> str:
    return json.dumps(
        {k: v for k, v in result.__dict__.items() if v is not None}, indent=2, sort_keys=True
    )
