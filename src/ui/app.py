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

# Markdown tables in a half-width column were wrapping inside words
# ("Profess / ionalis / m"). nowrap on the label and score columns; justification
# wraps at spaces only. Horizontal scroll if the pane is narrower than the labels.
QA_TABLE_CSS = """
#qa-card { overflow-x: auto; }
#qa-card table { table-layout: auto; }
#qa-card th, #qa-card td {
  word-break: normal;
  overflow-wrap: break-word;
  hyphens: none;
  vertical-align: top;
}
#qa-card th:nth-child(1), #qa-card td:nth-child(1),
#qa-card th:nth-child(2), #qa-card td:nth-child(2) {
  white-space: nowrap;
}
"""


def build_app(workflow: Any, engine: Engine, confidence_threshold: float = 0.6) -> gr.Blocks:
    with gr.Blocks(title=TITLE, css=QA_TABLE_CSS) as demo:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")
        build_analyze_tab(workflow, confidence_threshold)
        build_observability_tab(engine)
    return demo
