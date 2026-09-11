# Provenance

Which code in this repo is mine and which was teacher-forced. Same convention as
`~/dev/ik/agenticai`. This file is the authority; if it disagrees with memory, it wins.

**P1 (teacher-forced)** — written by Claude, run and break-tested by me. Pass 1 of the
scheduled-sampling protocol. Not evidence I can produce this cold.

**P3 (cold)** — reproduced from a blank file, no notes. The only pass that measures anything.

**Mine** — written by me from the start.

| File | Provenance | Date | Notes |
|---|---|---|---|
| `src/utils/config.py` | P1 | Sep 11, 2026 | frozen dataclass, empty-secret→None |
| `src/utils/audio.py` | P1 | Sep 11, 2026 | magic-byte detection, validation gate, wave/mutagen extraction |
| `tests/conftest.py` | P1 | Sep 11, 2026 | `make_wav_bytes` / `make_mp3_bytes` factories |
| `tests/unit/test_audio.py` | P1 | Sep 11, 2026 | Milestone 1 self-checks as tests |
| `tests/unit/test_config.py` | P1 | Sep 11, 2026 | |
| `scripts/check_stack.py` | P1 | Sep 11, 2026 | version smoke test; not part of the graded suite |
| scaffold, pyproject, Makefile, .env.example | P1 | Sep 11, 2026 | config, not logic |

## Scheduled for P3 (cold reproduction, blank file)

These five are 55% of the rubric and all of the interview value. Nothing here counts until
it has been written from empty with no notes open:

1. LangGraph wiring — nodes, the three conditional edges, `PipelineState`.
2. `route_after_intake` / `route_after_qa`.
3. Deterministic `overall_score` recomputation from `DIMENSION_WEIGHTS`.
4. PII redactor right-to-left replacement.
5. Injection detector placement — why *before* the LLM call, not after.

Gradio plumbing, ReportLab layout, Dockerfile and test scaffolding stay P1 permanently.
They are craft, not the thing being measured.

## Environment decisions

- Python 3.11.8, venv at `.venv`.
- **gradio pinned `<6`**: Gradio 6 removed `Textbox(show_copy_button=True)`, which the spec
  names outright in Milestone 6. Verified by `scripts/check_stack.py`.
- LangGraph 1.2.11 (spec assumed 0.4.x) — `StateGraph`, `add_conditional_edges`, `compile()`,
  `invoke()` all verified working. Riding the current API deliberately.
- ReportLab 5.0.1 (spec assumed 4.1) — `SimpleDocTemplate` → `%PDF-` verified.
- `hf-gradio` uninstalled (orphan from the Gradio 6 install).
- pip warns google-genai wants pydantic >=2.12.5 while gradio caps at 2.12.3. **Not a real
  break** — all three provider classes construct. Metadata pessimism; do not "fix" it.
