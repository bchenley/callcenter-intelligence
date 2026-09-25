# callcenter-intelligence
# src/security/pii_redactor.py

from __future__ import annotations

import re
from dataclasses import dataclass

# Declaration order breaks overlap ties at equal start position: an SSN and a
# phone number can both match a run of digits, so the more specific type leads.
PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}[-. ]\d{2}[-. ]\d{4}\b|\b\d{9}\b(?![-.\d])"), "SSN"),
    (
        # Whisper often inserts commas between spoken digit groups
        # ("4111, 1111, 1111, 1111"). Same 4x4 shape as the written card.
        re.compile(r"\b\d{4}(?:[-.,\s]*\d{4}){3}\b|\b3[47]\d{2}[-. ]?\d{6}[-. ]?\d{5}\b"),
        "CREDIT_CARD",
    ),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "EMAIL"),
    # Spoken "jane.do at example.com" - Whisper drops @ and says "at".
    (re.compile(r"\b[A-Za-z0-9._%+-]+\s+at\s+[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "EMAIL"),
    (re.compile(r"(?:\+?1[-. ]?)?\(?\b\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"), "PHONE"),
]

PLACEHOLDER = "[REDACTED_{label}]"


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    pii_found: bool
    pii_types: tuple[str, ...] = ()
    replacements: int = 0
    spans: tuple[tuple[int, int, str], ...] = ()


def _collect_spans(text: str) -> list[tuple[int, int, str]]:
    """All matches from the original text, deduplicated so overlaps keep the one
    starting earlier; ties at the same start go to the earlier-declared pattern."""
    found: list[tuple[int, int, str, int]] = []
    for order, (pattern, label) in enumerate(PII_PATTERNS):
        for m in pattern.finditer(text):
            start, end = m.span()
            found.append((start, end, label, order))
    found.sort(key=lambda s: (s[0], s[3]))
    kept: list[tuple[int, int, str]] = []
    for start, end, label, _order in found:
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, label))
    return kept


def redact_pii(text: str) -> RedactionResult:
    """Replace right-to-left so every not-yet-applied span keeps the offsets found
    on the original string. Left-to-right would shift them the moment a placeholder
    differs in length from what it replaced, and the shift compounds - the second
    replacement lands inside the wrong token. Runs before any LLM call."""
    if not isinstance(text, str):
        raise TypeError(f"redact_pii expects str, got {type(text).__name__}")
    spans = _collect_spans(text)
    if not spans:
        return RedactionResult(redacted_text=text, pii_found=False)
    out = text
    for start, end, label in reversed(spans):
        out = out[:start] + PLACEHOLDER.format(label=label) + out[end:]
    return RedactionResult(
        redacted_text=out,
        pii_found=True,
        pii_types=tuple(dict.fromkeys(label for _, _, label in spans)),
        replacements=len(spans),
        spans=tuple(spans),
    )
