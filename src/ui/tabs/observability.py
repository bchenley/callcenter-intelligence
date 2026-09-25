# callcenter-intelligence
# src/ui/tabs/observability.py

from __future__ import annotations

import gradio as gr
from sqlalchemy import Engine

from src.services.observability import AUDIT_COLUMNS, get_observability_dashboard


def build_observability_tab(engine: Engine) -> None:
    with gr.Tab("Observability") as tab:
        refresh = gr.Button("Refresh")
        metrics = gr.Markdown()
        langsmith = gr.Markdown()
        audit = gr.Dataframe(
            headers=list(AUDIT_COLUMNS),
            label="20 most recent audit events",
            wrap=True,
            interactive=False,
        )

        def load():
            return get_observability_dashboard(engine)

        outputs = [metrics, langsmith, audit]
        # Auto-refresh on tab click as well as on the button: a dashboard that shows
        # startup state until someone presses refresh is worse than no dashboard.
        tab.select(load, outputs=outputs)
        refresh.click(load, outputs=outputs)
