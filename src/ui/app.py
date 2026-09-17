# callcenter-intelligence
# src/ui/app.py

from __future__ import annotations

from typing import Any

import gradio as gr
from sqlalchemy import Engine

from src.ui.tabs.analyze import build_analyze_tab
from src.ui.tabs.observability import build_observability_tab

TITLE = "Call Center Intelligence System"
DESCRIPTION = (
    "Upload a call recording for transcription, speaker labelling, PII redaction, "
    "summarization and a five-dimension QA scorecard."
)


def build_app(workflow: Any, engine: Engine, confidence_threshold: float = 0.6) -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")
        build_analyze_tab(workflow, confidence_threshold)
        build_observability_tab(engine)
    return demo
