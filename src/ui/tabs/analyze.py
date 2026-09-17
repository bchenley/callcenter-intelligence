# callcenter-intelligence
# src/ui/tabs/analyze.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gradio as gr

from src.graph.state import CallStatus
from src.services.pipeline import PipelineResult, process_call

PROCESSING_NOTICE = (
    "### Processing\n\n"
    "Transcription and analysis typically take 30-90 seconds for a five-minute call. "
    "Do not refresh the page; the run is lost if you do."
)


def _present(result: PipelineResult) -> tuple[str, str, str, Any, Any]:
    """Map a pipeline result onto the five Gradio outputs.

    Rejected uploads (no transcript) stay a single error sentence - the rubric
    checks .ogg that way. Everything else, including an injection halt, shows
    Status so escalation is visible without opening the PDF.
    """
    if not result.ok and not result.transcript:
        message = f"### Could not analyze this call\n\n{result.error}"
        return "", message, "", None, None
    summary = f"### Status: {result.status}"
    if result.error and result.status != CallStatus.COMPLETED.value:
        summary = f"{summary}\n\n{result.error}"
    if result.summary_markdown:
        summary = f"{summary}\n\n{result.summary_markdown}"
    return (
        result.transcript,
        summary,
        result.qa_markdown,
        result.pdf_path,
        result.json_path,
    )


def _run(
    workflow: Any, confidence_threshold: float
) -> Callable[[Any, str, str], tuple[str, str, str, Any, Any]]:
    def handler(audio: Any, caller_id: str, department: str):
        result: PipelineResult = process_call(
            audio,
            workflow,
            caller_id=caller_id,
            department=department,
            confidence_threshold=confidence_threshold,
        )
        return _present(result)

    return handler


def build_analyze_tab(workflow: Any, confidence_threshold: float = 0.6) -> None:
    with gr.Tab("Analyze Call"):
        with gr.Row():
            with gr.Column(scale=2):
                audio = gr.Audio(
                    label="Call recording",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
            with gr.Column(scale=1):
                caller_id = gr.Textbox(label="Caller ID (optional)", placeholder="e.g. 48213")
                department = gr.Textbox(label="Department (optional)", placeholder="e.g. Billing")
        analyze = gr.Button("Analyze Call", variant="primary")
        status = gr.Markdown(PROCESSING_NOTICE, visible=False)
        transcript = gr.Textbox(
            label="Transcript", lines=15, show_copy_button=True, interactive=False
        )
        with gr.Row():
            summary = gr.Markdown(label="Summary")
            qa = gr.Markdown(label="QA scorecard", elem_id="qa-card")
        with gr.Row():
            pdf = gr.File(label="Download PDF report")
            js = gr.File(label="Download JSON report")

        # show notice -> run -> hide notice. Chained so the notice is on screen for
        # the whole run rather than flashing after it.
        analyze.click(lambda: gr.update(visible=True), outputs=status).then(
            _run(workflow, confidence_threshold),
            inputs=[audio, caller_id, department],
            outputs=[transcript, summary, qa, pdf, js],
        ).then(lambda: gr.update(visible=False), outputs=status)
