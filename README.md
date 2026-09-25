<!-- callcenter-intelligence -->
<!-- README.md -->

# Call Center Intelligence System

Demo video: TODO_LINK

Upload a call recording. The system transcribes it, labels speakers, strips
prompt-injection and PII before any model sees the text, then writes a structured
summary, a five-dimension QA score, and downloadable PDF and JSON reports.

## Architecture

Five layers under `src/`: `ui`, `services`, `agents`, `graph`, `database`. Security
helpers live in `src/security/`. `app.py` is the entrypoint: it loads config, opens
SQLite, warms the Whisper singleton, compiles the LangGraph workflow once, and
launches Gradio.

### Why a graph, not a straight-line script

Three decisions in the pipeline are data-dependent:

- invalid audio never reaches Whisper
- a prompt-injection hit never reaches an LLM
- a critical compliance flag still produces a report, but the call is marked
  `flagged_for_review` instead of `completed`

A function chain hides those branches in `if` statements and shared mutable
state. LangGraph makes them explicit: each node reads and writes a typed
`PipelineState`, and `compile_workflow` builds the graph once per database
engine so every request reuses the same compiled object.

### Seven stages

The compiled graph registers eight node functions. Error handling and supervisor
review are two terminals of the same last stage.

1. **Intake** validates bytes (empty, size, format, WAV duration), assigns a
   `call_id`, writes a temp file, and scans `caller_id` / `department` for PII.
   Failed intake routes to error. Nothing reaches Whisper until this passes.
2. **Transcription** runs faster-whisper (`beam_size=1`, VAD on, no
   condition-on-previous-text). Heuristic Agent/Customer diarization, SHA-256
   cache keyed by audio hash, per-segment confidence from `avg_logprob` and
   `no_speech_prob`. The compiled graph then always continues to injection
   check; transcription failures are raised inside the node and land in
   `state["error"]`.
3. **Injection check** scans the transcript with 22 named regex patterns. A hit
   sets `flagged_for_review`, writes `injection_detected` to the audit log, and
   routes to error. There is no edge from this node to summarization.
4. **PII redaction** replaces SSN, credit card, email, and phone with labelled
   placeholders, right-to-left so offsets stay valid. It rewrites both
   `full_text` (what the LLM reads) and every segment (what the UI renders).
5. **Summarize and QA** is the first LLM stage. Summarization runs first.
   QA scoring receives that summary as context, then **discards** the model's
   `overall_score` and recomputes
   `sum(dimension.score * weight)` with weights
   Professionalism 0.15, Empathy 0.20, Problem Resolution 0.30, Compliance 0.20,
   Communication Clarity 0.15. Temperature is 0. Retries use
   `min(2**attempt, 10)` seconds of backoff.
6. **Report** compiles a `CallReport`, persists a `CallRecord`, and writes
   `completed` to the audit log.
7. **Error / supervisor review.** Supervisor review is the same persist path
   with status `flagged_for_review` (critical compliance only; low/medium/high
   still complete). The error node surfaces `state["error"]`, else the intake
   validation message, else a generic failure string.

```
intake -> transcribe -> injection_check -> redact -> summarize_and_qa
              |                |                         |
            error            error              report | supervisor | error
```

### Security ordering

Injection check and PII redaction both run after transcription and before
`get_llm`. An injected phrase in the audio cannot reach the summarizer. Raw
account numbers cannot leave the box in an API payload. The audit log is
append-only SQLite (`AuditLogEntry`); rows are not updated or deleted.

### Data

Three tables: `call_records` (per call, JSON payloads), `audit_log` (per event),
`transcription_cache` (per audio hash). Sessions come from a cached
`sessionmaker` keyed by `id(engine)`.

The UI is two Gradio tabs: Analyze Call, and Observability (counts, LangSmith
status, last 20 audit events). Temp PDF/JSON/WAV files are capped at 50.

## Setup

Python 3.11 or 3.12. 3.13 is not supported (Gradio).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env`. Set `LLM_PROVIDER` to `openai`, `gemini`, or `groq`, and the matching
API key. Leave the other keys empty. `WHISPER_MODEL_SIZE=base` is the default
on a laptop. Set `tiny` if you want a faster first load.

```bash
pre-commit install   # also run by `make install`
```

## Run

```bash
make run
# or: python app.py
```

Opens http://127.0.0.1:7860. Upload wav/mp3/flac/m4a, or record from the
microphone. Unsupported formats (for example `.ogg`) return an error string,
not a traceback.

## Tests

No API keys required. Whisper and the LLM are mocked.

```bash
make test              # unit + security
make test-integration  # full graph, mocked Whisper and LLM
make test-all          # everything
# or: PYTHONPATH=. pytest tests/ -q
```

## GPU

`_resolve_device` in `src/agents/transcription.py`:

- NVIDIA CUDA available: `device="cuda"`, `compute_type="float16"`
- otherwise, including Apple MPS: `device="cpu"`, `compute_type="int8"`

CTranslate2 has no MPS backend. Setting the device to `mps` would fail at
model load, so this code never does that.

On a CUDA host set `WHISPER_MODEL_SIZE=large-v3` (or `small`). On CPU the
default is `base`. `large-v3` on CPU is tens of minutes per call.

The model is a process-wide singleton. `app.py` loads it at startup so the first
request does not pay the 5-30s load.

## Docker

```bash
docker build -t callcenter-intelligence .
docker run --rm -p 7860:7860 --env-file .env callcenter-intelligence
```

app.py binds 0.0.0.0 automatically inside a container (it detects /.dockerenv)
or on HuggingFace Spaces (SPACE_ID), and 127.0.0.1 otherwise. Set SERVER_HOST
to override.

Docker `--env-file` does not strip inline comments. Keep comments on their own
lines in `.env` (see `.env.example`).

The image is `python:3.11-slim` plus `ffmpeg` (and `libsndfile1` for
`soundfile`). Config is environment-only; do not bake a `.env` into the image.

## HuggingFace Spaces

Create a Gradio Space. Add the provider API key as a secret. The root
`requirements.txt` is installed automatically. Use `WHISPER_MODEL_SIZE=base` on
the free CPU tier (`tiny` if the Space is memory-constrained).
