# callcenter-intelligence
# src/agents/qa_scoring.py

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.prompts import QA_SYSTEM_PROMPT
from src.agents.summarization import backoff_seconds, format_transcript
from src.graph.state import DIMENSION_WEIGHTS, QAScoreResult, SummaryResult, TranscriptionResult
from src.utils.formatters import format_summary

logger = logging.getLogger(__name__)


class QAScoringError(RuntimeError):
    pass


def compute_overall_score(qa: QAScoreResult) -> float:
    """The deterministic guardrail. An LLM asked for a weighted average produces a
    plausible number, not the number - and two runs of the same call produce two
    different ones, which makes the score unusable in a performance review. This is
    arithmetic, so it is done in Python and the model's value is discarded."""
    total = sum(getattr(qa, dim).score * weight for dim, weight in DIMENSION_WEIGHTS.items())
    return round(total, 4)


def run_qa_scoring(
    transcript: TranscriptionResult,
    summary: SummaryResult,
    llm: Any,
    max_retries: int = 3,
    sleep: Any = time.sleep,
) -> QAScoreResult:
    # The summary goes in as context: it tells the scorer what the call was for, so
    # resolution and compliance are judged against the actual purpose.
    messages = [
        ("system", QA_SYSTEM_PROMPT),
        (
            "human",
            (
                f"Call summary for context:\n\n{format_summary(summary)}\n\n"
                f"Transcript:\n\n{format_transcript(transcript)}"
            ),
        ),
    ]
    structured = llm.with_structured_output(QAScoreResult)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            result: QAScoreResult = structured.invoke(messages)
            return result.model_copy(
                update={
                    "call_id": transcript.call_id,
                    "overall_score": compute_overall_score(result),
                }
            )
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise unrelated types
            last_error = exc
            logger.warning("qa scoring attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                sleep(backoff_seconds(attempt))
    raise QAScoringError(
        f"qa scoring failed after {max_retries} attempts: {last_error}"
    ) from last_error
