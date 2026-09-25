# callcenter-intelligence
# src/agents/summarization.py

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.prompts import SUMMARIZATION_SYSTEM_PROMPT
from src.graph.state import SummaryResult, TranscriptionResult
from src.utils.formatters import secs_to_mmss

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 10


class SummarizationError(RuntimeError):
    pass


def format_transcript(transcript: TranscriptionResult) -> str:
    """[MM:SS-MM:SS] Speaker: text - timestamps are in the prompt because the QA
    agent is required to cite them, and it can only cite what it was shown."""
    return "\n".join(
        f"[{secs_to_mmss(s.start)}-{secs_to_mmss(s.end)}] {s.speaker}: {s.text}"
        for s in transcript.segments
    )


def backoff_seconds(attempt: int) -> int:
    """1s, 2s, 4s, 8s, capped at 10. Capped because a node that sleeps longer than
    the LLM timeout has stopped retrying and started hanging."""
    return min(2**attempt, MAX_BACKOFF_SECONDS)


def run_summarization(
    transcript: TranscriptionResult,
    llm: Any,
    max_retries: int = 3,
    sleep: Any = time.sleep,
) -> SummaryResult:
    messages = [
        ("system", SUMMARIZATION_SYSTEM_PROMPT),
        ("human", f"Transcript:\n\n{format_transcript(transcript)}"),
    ]
    structured = llm.with_structured_output(SummaryResult)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            result: SummaryResult = structured.invoke(messages)
            # The LLM does not know the call_id and must not be trusted to echo it.
            return result.model_copy(update={"call_id": transcript.call_id})
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise unrelated types
            last_error = exc
            logger.warning("summarization attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                sleep(backoff_seconds(attempt))
    raise SummarizationError(
        f"summarization failed after {max_retries} attempts: {last_error}"
    ) from last_error
