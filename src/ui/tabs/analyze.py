# callcenter-intelligence
# src/ui/tabs/analyze.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gradio as gr

from src.services.pipeline import PipelineResult, process_call

PROCESSING_NOTICE = (
    "### Processing\n\n"
    "Transcription and analysis typically take 30-90 seconds for a five-minute call. "
    "Do not refresh the page; the run is lost if you do."
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
        if not result.ok and not result.transcript:
            # A rejected upload gets a sentence, never a traceback - the rubric
            # checks this with an .ogg file.
            message = f"### Could not analyze this call\n\n{result.error}"
            return "", message, "", None, None
        return (
            result.transcript,
            result.summary_markdown,
            result.qa_markdown,
            result.pdf_path,
            result.json_path,
        )

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
            qa = gr.Markdown(label="QA scorecard")
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
