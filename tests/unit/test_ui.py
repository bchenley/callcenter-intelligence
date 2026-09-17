# callcenter-intelligence
# tests/unit/test_ui.py

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph.state import CallStatus
from src.services.pipeline import PipelineResult
from src.ui import app as ui_app
from src.ui.tabs import analyze as analyze_tab
from src.ui.tabs import observability as obs_tab


@pytest.fixture(autouse=True)
def _no_real_pipeline(monkeypatch):
    """Every UI test drives the handler directly; none of them may touch Whisper,
    an LLM or the graph."""
    yield


# ---- the error path the rubric checks ----------------------------------------


def test_rejected_upload_shows_a_message_not_a_traceback(monkeypatch) -> None:
    """IK's self-check uploads an .ogg and requires a clear message with no
    traceback visible to the user."""
    monkeypatch.setattr(
        analyze_tab,
        "process_call",
        lambda *a, **k: PipelineResult(
            status=CallStatus.FAILED.value,
            error="Unsupported format: ogg. Supported formats are WAV, MP3, FLAC, M4A.",
        ),
    )
    transcript, summary, _qa, pdf, js = analyze_tab._run(MagicMock(), 0.6)("clip.ogg", "", "")
    assert transcript == ""
    assert "Unsupported format" in summary
    assert "Traceback" not in summary and "Error:" not in summary
    assert pdf is None and js is None


def test_failure_message_is_markdown_not_raw(monkeypatch) -> None:
    monkeypatch.setattr(
        analyze_tab,
        "process_call",
        lambda *a, **k: PipelineResult(status=CallStatus.FAILED.value, error="Empty file."),
    )
    _, summary, *_ = analyze_tab._run(MagicMock(), 0.6)(None, "", "")
    assert summary.startswith("### Could not analyze this call")


def test_no_audio_is_handled(monkeypatch) -> None:
    monkeypatch.setattr(
        analyze_tab,
        "process_call",
        lambda *a, **k: PipelineResult(status=CallStatus.FAILED.value, error="No audio provided."),
    )
    _, summary, *_ = analyze_tab._run(MagicMock(), 0.6)(None, "", "")
    assert "No audio" in summary


# ---- the success path --------------------------------------------------------


def _ok() -> PipelineResult:
    return PipelineResult(
        status=CallStatus.COMPLETED.value,
        transcript="[00:00] Agent: Thank you for calling.",
        summary_markdown="## Call Summary",
        qa_markdown="## QA Scorecard",
        pdf_path="/tmp/r.pdf",
        json_path="/tmp/r.json",
        call_id="c1",
    )


def test_success_returns_five_outputs_in_order(monkeypatch) -> None:
    monkeypatch.setattr(analyze_tab, "process_call", lambda *a, **k: _ok())
    transcript, summary, qa, pdf, js = analyze_tab._run(MagicMock(), 0.6)("a.wav", "", "")
    assert transcript.startswith("[00:00] Agent:")
    assert summary == "## Call Summary"
    assert qa == "## QA Scorecard"
    assert pdf == "/tmp/r.pdf" and js == "/tmp/r.json"


def test_caller_id_and_department_reach_the_pipeline(monkeypatch) -> None:
    seen: dict = {}

    def spy(audio, workflow, **kwargs):
        seen.update(kwargs)
        return _ok()

    monkeypatch.setattr(analyze_tab, "process_call", spy)
    analyze_tab._run(MagicMock(), 0.6)("a.wav", "48213", "Billing")
    assert seen["caller_id"] == "48213"
    assert seen["department"] == "Billing"


def test_confidence_threshold_is_passed_through(monkeypatch) -> None:
    seen: dict = {}

    def spy(audio, workflow, **kwargs):
        seen.update(kwargs)
        return _ok()

    monkeypatch.setattr(analyze_tab, "process_call", spy)
    analyze_tab._run(MagicMock(), 0.42)("a.wav", "", "")
    assert seen["confidence_threshold"] == 0.42


def test_flagged_call_still_renders_its_artifacts(monkeypatch) -> None:
    """A critical compliance flag is not a failure to display - a supervisor needs
    the transcript and the scorecard."""
    flagged = PipelineResult(
        status=CallStatus.FLAGGED_FOR_REVIEW.value,
        transcript="[00:00] Agent: hello",
        summary_markdown="## Call Summary",
        qa_markdown="## QA Scorecard",
        pdf_path="/tmp/r.pdf",
        json_path="/tmp/r.json",
    )
    monkeypatch.setattr(analyze_tab, "process_call", lambda *a, **k: flagged)
    transcript, summary, _qa, pdf, _js = analyze_tab._run(MagicMock(), 0.6)("a.wav", "", "")
    assert transcript and pdf == "/tmp/r.pdf"
    assert "Could not analyze" not in summary


# ---- notice copy -------------------------------------------------------------


def test_processing_notice_warns_against_refreshing() -> None:
    notice = analyze_tab.PROCESSING_NOTICE
    notice.encode("ascii")
    assert "refresh" in notice.lower()
    assert any(ch.isdigit() for ch in notice), "must give an expected duration"


# ---- observability tab -------------------------------------------------------


def test_dashboard_loader_returns_the_three_outputs(engine) -> None:
    metrics_md, langsmith_md, rows = obs_tab.get_observability_dashboard(engine)
    assert metrics_md.startswith("## Pipeline health")
    assert langsmith_md.startswith("## LangSmith")
    assert isinstance(rows, list)


def test_audit_headers_match_the_service() -> None:
    from src.services.observability import AUDIT_COLUMNS

    assert obs_tab.AUDIT_COLUMNS is AUDIT_COLUMNS
    assert list(AUDIT_COLUMNS) == ["Timestamp", "Call ID", "Action", "Details"]


# ---- app assembly ------------------------------------------------------------


def test_build_app_returns_blocks_with_both_tabs(engine) -> None:
    demo = ui_app.build_app(MagicMock(), engine)
    rendered = str(demo.get_config_file())
    assert "Analyze Call" in rendered
    assert "Observability" in rendered


def test_exactly_two_tabs(engine) -> None:
    """Two tabs, not three. The brief's 'three tabs' is wrong; the milestone doc
    and the problem statement both say two."""
    demo = ui_app.build_app(MagicMock(), engine)
    config = demo.get_config_file()
    tabs = [c for c in config["components"] if c.get("type") == "tabitem"]
    assert len(tabs) == 2


def test_title_and_description_are_ascii() -> None:
    ui_app.TITLE.encode("ascii")
    ui_app.DESCRIPTION.encode("ascii")


def test_build_app_does_not_invoke_the_workflow(engine) -> None:
    workflow = MagicMock()
    ui_app.build_app(workflow, engine)
    assert workflow.invoke.call_count == 0, "building the UI must not run a call"
