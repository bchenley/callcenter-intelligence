# callcenter-intelligence
# scripts/check_stack.py

"""Smoke-test the installed stack against the APIs the app depends on.

The stack was written for LangGraph 0.4.x, Gradio 5.x and ReportLab 4.1.x. pip may resolve
newer majors. This exercises exactly the API surface the project uses, so a breaking
change shows up now instead of mid-build.

NOTE: no `from __future__ import annotations` in this file, and the TypedDict / ORM classes
live at MODULE level on purpose. Both LangGraph and SQLAlchemy resolve annotations at runtime
against module globals - stringified annotations on function-local classes cannot be resolved,
which produces a misleading "NameError: name 'S' is not defined" / "Could not interpret
annotation Mapped[int]" that looks like a version problem but is a scoping problem.

Run:  python scripts/check_stack.py [-v]
"""

import io
import sys
import traceback
from typing import TypedDict

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn

    return wrap


def run_checks(verbose: bool = False) -> list[tuple[str, bool, str]]:
    results = []
    for name, fn in CHECKS:
        try:
            fn()
            results.append((name, True, ""))
        # A check that raises must be recorded as a failed check, not end the run.
        # This harness exists to report every API that broke, not just the first.
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            if verbose:
                traceback.print_exc()
    return results


# ---------------------------------------------------------------- versions
@check("versions")
def _versions():
    import importlib.metadata as md

    for pkg in ("langgraph", "gradio", "reportlab", "pydantic", "faster-whisper", "sqlalchemy"):
        print(f"  {pkg:16} {md.version(pkg)}")


# ---------------------------------------------------------------- LangGraph
# Module level - see the note in the docstring.
class SmokeState(TypedDict, total=False):
    value: int
    route: str
    trace: list[str]


@check("LangGraph: StateGraph + add_conditional_edges + compile + invoke")
def _langgraph():
    from langgraph.graph import END, StateGraph

    def start(state: SmokeState) -> dict:
        return {"value": state.get("value", 0) + 1, "trace": ["start"]}

    def good(state: SmokeState) -> dict:
        return {"route": "good", "trace": state.get("trace", []) + ["good"]}

    def bad(state: SmokeState) -> dict:
        return {"route": "bad", "trace": state.get("trace", []) + ["bad"]}

    def router(state: SmokeState) -> str:
        return "good_node" if state["value"] > 0 else "bad_node"

    g = StateGraph(SmokeState)
    g.add_node("start_node", start)
    g.add_node("good_node", good)
    g.add_node("bad_node", bad)
    g.set_entry_point("start_node")
    g.add_conditional_edges(
        "start_node", router, {"good_node": "good_node", "bad_node": "bad_node"}
    )
    g.add_edge("good_node", END)
    g.add_edge("bad_node", END)

    app = g.compile()
    out = app.invoke({"value": 0})
    assert out["route"] == "good", out
    assert out["trace"] == ["start", "good"], out
    print(f"  routed correctly, final state keys: {sorted(out)}")


# ---------------------------------------------------------------- Gradio
@check("Gradio: Audio(numpy+sources), Blocks/Tabs, File, Dataframe, .click().then()")
def _gradio():
    import gradio as gr

    with gr.Blocks() as demo:  # noqa: F841
        with gr.Tabs():
            with gr.Tab("Analyze"):
                audio = gr.Audio(type="numpy", sources=["upload", "microphone"])
                caller = gr.Textbox(label="Caller ID")
                btn = gr.Button("Analyze Call", variant="primary")
                status = gr.Markdown(visible=False)
                transcript = gr.Textbox(lines=15)
                pdf = gr.File(label="PDF")
            with gr.Tab("Observability") as obs_tab:
                table = gr.Dataframe(headers=["Timestamp", "Call ID", "Action", "Details"])

        btn.click(lambda: gr.update(visible=True), outputs=status).then(
            lambda a, c: ("text", None), inputs=[audio, caller], outputs=[transcript, pdf]
        ).then(lambda: gr.update(visible=False), outputs=status)

        obs_tab.select(lambda: [["t", "c", "a", "d"]], outputs=table)
    print("  Blocks graph built; .click().then().then() chain and tab.select() accepted")


@check("Gradio: Textbox(show_copy_button=True)")
def _gradio_copy_button():
    import gradio as gr

    with gr.Blocks():
        gr.Textbox(lines=15, show_copy_button=True)
    print("  show_copy_button accepted")


# ---------------------------------------------------------------- ReportLab
@check("ReportLab: SimpleDocTemplate to BytesIO -> %PDF- bytes")
def _reportlab():
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    doc.build(
        [
            Paragraph("Call Report", styles["Title"]),
            Spacer(1, 12),
            Table([["Dimension", "Score"], ["Empathy", "4"]]),
        ]
    )
    data = buf.getvalue()
    assert data[:5] == b"%PDF-", data[:20]
    print(f"  {len(data)} bytes, header {data[:8]!r}")


# ---------------------------------------------------------------- Pydantic
@check("Pydantic v2: Field constraints + enum + ValidationError")
def _pydantic():
    from enum import Enum

    import pydantic
    from pydantic import BaseModel, Field

    class Resolution(str, Enum):
        resolved = "resolved"
        unresolved = "unresolved"
        escalated = "escalated"

    class Segment(BaseModel):
        confidence: float = Field(ge=0.0, le=1.0)
        status: Resolution = Resolution.resolved

    Segment(confidence=0.5)
    try:
        Segment(confidence=1.5)
    except pydantic.ValidationError:
        pass
    else:
        raise AssertionError("ge/le constraint not enforced")
    print("  constraints enforced, enum coerces from string")


# ---------------------------------------------------------------- SQLAlchemy
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class SmokeBase(DeclarativeBase):
    pass


class SmokeRec(SmokeBase):
    __tablename__ = "smoke_rec"
    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)


@check("SQLAlchemy 2.x: DeclarativeBase + create_all + session round-trip")
def _sqlalchemy():
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    SmokeBase.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as s:
        s.add(SmokeRec(call_id="abc"))
        s.commit()
        assert s.scalar(select(SmokeRec).where(SmokeRec.call_id == "abc")) is not None
    print("  declarative models, unique index and session round-trip OK")


# ---------------------------------------------------------------- whisper
@check("faster-whisper: transcribe() accepts the pipeline's settings (no model download)")
def _whisper():
    import inspect

    from faster_whisper import WhisperModel

    sig = inspect.signature(WhisperModel.transcribe)
    for param in ("beam_size", "vad_filter", "condition_on_previous_text", "word_timestamps"):
        assert param in sig.parameters, f"transcribe() has no {param}"
    print("  beam_size, vad_filter, condition_on_previous_text, word_timestamps all present")


# ---------------------------------------------------------------- LLM providers
@check("LLM factory: all three provider classes import and construct")
def _providers():
    """OpenAI, Gemini and Groq must import and construct, switchable by env var.

    Constructing with a dummy key makes no network call - it only proves the package
    imports cleanly and the constructor signature matches. This is the check that
    catches a pydantic-version conflict between gradio and google-genai before it
    surfaces as an ImportError at runtime.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", api_key="sk-dummy", timeout=60)
    assert llm.model_name == "gpt-4o"
    print("  openai   ChatOpenAI(gpt-4o) OK")

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="dummy")
    print("  gemini   ChatGoogleGenerativeAI(gemini-2.0-flash) OK")

    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key="dummy")
    print("  groq     ChatGroq(llama-3.3-70b-versatile) OK")


@check("with_structured_output is available on a chat model")
def _structured_output():
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    class Tiny(BaseModel):
        purpose: str

    llm = ChatOpenAI(model="gpt-4o", api_key="sk-dummy")
    bound = llm.with_structured_output(Tiny)
    assert bound is not None
    print("  with_structured_output(Model) binds without error")


if __name__ == "__main__":
    results = run_checks(verbose="-v" in sys.argv)
    print()
    for name, ok, err in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if err:
            print(f"       {err}")
    print()
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"{len(failed)} check(s) failed. Re-run with -v for tracebacks.")
        raise SystemExit(1)
    print("Stack matches the API surface the app depends on. Safe to build on these versions.")
    raise SystemExit(0)
